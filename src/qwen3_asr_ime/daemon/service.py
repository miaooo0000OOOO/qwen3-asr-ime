"""Main voice input daemon.

This daemon ties together the global hotkey listener, microphone recorder,
streaming ASR client, IPC server, and X11 text injection.

High-level flow (streaming branch):

1. The user holds the configured hotkey (default: Ctrl).
2. ``VoiceInputDaemon._on_hotkey(press)`` starts the ``Recorder`` and opens a
   WebSocket connection to the Qwen3-ASR server via ``ASRStreamClient``.
3. Audio chunks captured by ``sounddevice`` are pushed into a deque and sent
   over the WebSocket by ``_send_chunks_loop``.
4. The server returns ``partial`` results during recording; these are broadcast
   to any connected IPC clients over the Unix socket.
5. When the user releases the hotkey, ``_finish_stream()`` flushes remaining
   chunks, sends a ``finish`` message, waits for the ``final`` result, and
   injects the recognized text into the active X11 window.

IPC:

- A Unix domain socket (configured by ``ipc_socket_path``) accepts JSON-line
  messages from external clients (e.g. a future GUI or shell extension).
  Currently only ``HotkeyCommand`` is handled, allowing remote triggering of
  recording.
- State updates and recognized text are broadcast back to connected clients as
  JSON-line messages.
"""

from __future__ import annotations

import asyncio
import collections
import os
import threading
import signal
from pathlib import Path

from qwen3_asr_ime.common.config import IMEConfig
from qwen3_asr_ime.common.logger import get_logger
from qwen3_asr_ime.common.protocol import HotkeyCommand, RecognizedText, StateUpdate, parse_message
from qwen3_asr_ime.daemon.asr_client import ASRResult, ASRStreamClient
from qwen3_asr_ime.daemon.hotkey import HotkeyEvent, create_hotkey_listener
from qwen3_asr_ime.daemon.recorder import AudioConfig, Recorder

logger = get_logger(__name__)


def _type_text_x11(text: str) -> None:
    """Type Unicode text into the currently focused X11 window.

    Uses ``pynput.keyboard.Controller.type``, which sends fake key events
    through the X11 XTEST extension. The X Server routes these events to the
    window that currently has keyboard focus, so no window ID is needed.
    """
    from pynput.keyboard import Controller

    Controller().type(text)


def _type_incremental_x11(to_delete: int, text: str) -> None:
    """Delete ``to_delete`` characters then type ``text`` into the active window.

    This implementation avoids ``pynput.keyboard.Controller.type``'s keycode
    borrowing limits, which cause ``InvalidCharacterException`` for long
    Chinese strings. It uses a single scratch keycode and synthetic X events
    with ``state=0`` so that physical modifiers (e.g. the held Ctrl hotkey)
    do not corrupt the injected characters.
    """
    import time

    import Xlib.display
    import Xlib.X
    import Xlib.XK
    from pynput._util.xorg import char_to_keysym

    display = Xlib.display.Display()
    try:
        focus = display.get_input_focus().focus
        root = display.screen().root
        send_event = getattr(focus, "send_event", lambda event: display.send_event(focus, event))

        def _send_keycode(keycode: int, state: int = 0) -> None:
            for is_press in (True, False):
                event_cls = (
                    Xlib.display.event.KeyPress if is_press else Xlib.display.event.KeyRelease
                )
                send_event(
                    event_cls(
                        detail=keycode,
                        state=state,
                        time=0,
                        root=root,
                        window=focus,
                        same_screen=0,
                        child=Xlib.X.NONE,
                        root_x=0,
                        root_y=0,
                        event_x=0,
                        event_y=0,
                    )
                )
            display.sync()

        # Backspace: use an explicit synthetic event to avoid pynput borrowing.
        backspace_kc = display.keysym_to_keycode(Xlib.XK.XK_BackSpace)
        for _ in range(to_delete):
            _send_keycode(backspace_kc)
            time.sleep(0.005)

        if not text:
            return

        # Find an unused scratch keycode to temporarily map Unicode characters.
        min_kc = display.display.info.min_keycode
        max_kc = display.display.info.max_keycode
        mapping = display.get_keyboard_mapping(min_kc, max_kc - min_kc + 1)
        scratch_kc = None
        for i, keysyms in enumerate(mapping):
            if all(k == 0 for k in keysyms):
                scratch_kc = min_kc + i
                break
        if scratch_kc is None:
            scratch_kc = max_kc

        for ch in text:
            keysym = char_to_keysym(ch)
            if not keysym:
                continue
            display.change_keyboard_mapping(scratch_kc, [(keysym,)])
            display.sync()
            _send_keycode(scratch_kc)
            time.sleep(0.01)

        # Restore the scratch keycode.
        display.change_keyboard_mapping(scratch_kc, [(Xlib.XK.NoSymbol,)])
        display.sync()
    finally:
        display.close()


class VoiceInputDaemon:
    """Daemon that orchestrates hotkey, recording, streaming ASR, IPC, and typing.

    Attributes:
        config: Runtime configuration loaded from YAML.
        recorder: ``Recorder`` instance for microphone capture.
        hotkey: Global hotkey listener (pynput-based).
        _clients: Connected IPC clients over the Unix socket.
        _state: Current high-level state: ``"idle"``, ``"recording"``,
            ``"recognizing"``, or ``"error"``.
        _stream_client: Active WebSocket client to the ASR server.
        _streaming_task: Async task that reads partial/final results.
        _sender_task: Async task that drains the chunk deque to the server.
        _pending_chunks: Deque of int16 PCM chunks waiting to be sent.
        _stream_final_event: Set when the streaming reader reaches the final
            result or an error.
    """

    def __init__(self, config: IMEConfig):
        self.config = config
        self.recorder = Recorder(
            AudioConfig(
                sample_rate=config.audio_sample_rate,
                channels=config.audio_channels,
                chunk_ms=config.audio_chunk_ms,
            )
        )
        self.hotkey = create_hotkey_listener(
            config.hotkey_device,
            config.hotkey_key,
            self._on_hotkey,
        )
        self._clients: set[asyncio.StreamWriter] = set()
        self._lock = threading.Lock()
        self._state: str = "idle"
        self._server: asyncio.AbstractServer | None = None

        # Streaming ASR state
        self._stream_client: ASRStreamClient | None = None
        self._streaming_task: asyncio.Task[None] | None = None
        self._sender_task: asyncio.Task[None] | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._pending_chunks: collections.deque[bytes] = collections.deque()
        self._max_pending_chunks = 200  # ~4 seconds at 20ms/chunk
        self._streaming_error: str | None = None
        self._current_text: str = ""
        self._typed_text: str = ""
        self._stream_final_result: ASRResult | None = None
        self._stream_final_event: asyncio.Event | None = None

    async def start(self) -> None:
        """Create the IPC Unix socket and start the hotkey listener."""
        socket_path = Path(self.config.ipc_socket_path)
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        if socket_path.exists():
            socket_path.unlink()
        self._loop = asyncio.get_running_loop()

        self._server = await asyncio.start_unix_server(
            self._on_client_connected,
            path=str(socket_path),
        )
        os.chmod(socket_path, 0o600)
        self.hotkey.start()

    def _on_hotkey_message(self, msg: HotkeyCommand) -> None:
        """Handle hotkey commands received over IPC."""
        logger.debug("IPC hotkey: %s", msg.action)
        self._on_hotkey(HotkeyEvent(action=msg.action))

    async def run_forever(self) -> None:
        """Run the Unix socket server until SIGTERM/SIGINT."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown)
        async with self._server:
            self._serve_task = asyncio.create_task(self._server.serve_forever())
            try:
                await self._serve_task
            except asyncio.CancelledError:
                pass

    def _on_hotkey(self, event: HotkeyEvent) -> None:
        """Handle a hotkey event (may arrive from a non-asyncio thread)."""
        self._loop.call_soon_threadsafe(lambda ev=event: self._handle_hotkey(ev))

    def _handle_hotkey(self, event: HotkeyEvent) -> None:
        """State machine entry point driven by press/release hotkey events.

        Always runs on the asyncio event loop.
        """
        try:
            if event.action == "press" and self._state == "idle":
                self._state = "recording"
                self._current_text = ""
                self._typed_text = ""
                self._streaming_error = None
                self._stream_final_result = None
                self._stream_final_event = asyncio.Event()
                self._pending_chunks.clear()
                self.recorder.start(chunk_callback=self._on_audio_chunk)
                self._stream_client = ASRStreamClient(
                    self.config.asr_endpoint,
                    api_key=self.config.asr_api_key,
                    timeout=self.config.asr_timeout,
                )
                self._streaming_task = asyncio.create_task(self._run_stream())
                self._broadcast_state("recording", "🔴 录音中")
                logger.info("⬇ Ctrl 按下 → 开始录音并建立流式 ASR 连接")
            elif event.action == "release" and self._state == "recording":
                self._state = "recognizing"
                self._broadcast_state("recognizing", "🔄 识别中...")
                audio_bytes = self.recorder.stop()
                dur = len(audio_bytes) / 32000
                logger.info(
                    "⬆ Ctrl 松开 → 停止录音 (%.1f 秒, %d KB)", dur, len(audio_bytes) // 1024
                )
                asyncio.create_task(self._finish_stream())
            elif event.action == "interrupt" and self._state == "recording":
                logger.info("⛔ 检测到组合键，中断语音输入")
                self._state = "idle"
                self._broadcast_state("idle", "🎤 就绪")
                self.recorder.stop()
                self._pending_chunks.clear()
                asyncio.create_task(self._cleanup_streaming())
        except Exception:
            logger.exception("❌ 热键处理错误")

    def _on_audio_chunk(self, chunk_bytes: bytes) -> None:
        """Called from sounddevice thread for each audio chunk.

        Pushes the chunk into a thread-safe deque. ``_send_chunks_loop``
        consumes it on the asyncio event loop.
        """
        if len(self._pending_chunks) >= self._max_pending_chunks:
            # Drop oldest chunk to avoid unbounded growth.
            self._pending_chunks.popleft()
        self._pending_chunks.append(chunk_bytes)

    async def _run_stream(self) -> None:
        """Manage the WebSocket streaming ASR session.

        Connects to the server, starts the chunk sender, and consumes
        partial/final results from ``ASRStreamClient.iterate()``.
        """
        if self._stream_client is None:
            return
        try:
            await self._stream_client.connect()
            # Start sender task to drain pending audio chunks.
            self._sender_task = asyncio.create_task(self._send_chunks_loop())
            async for result in self._stream_client.iterate():
                if result.error:
                    self._streaming_error = result.error
                    logger.error("Streaming ASR error: %s", result.error)
                    break
                if result.final:
                    self._stream_final_result = result
                    break
                # Partial result: update current text, broadcast for live preview,
                # and type the incremental difference when no IPC client is connected.
                if result.text != self._current_text:
                    self._current_text = result.text
                    self._broadcast_recognized(result.text)
                    self._type_text_incremental(result.text)
        except Exception as exc:
            self._streaming_error = str(exc)
            logger.exception("❌ 流式 ASR 会话失败")
        finally:
            if self._sender_task is not None:
                self._sender_task.cancel()
                try:
                    await self._sender_task
                except asyncio.CancelledError:
                    pass
                self._sender_task = None
            if self._stream_final_event is not None:
                self._stream_final_event.set()

    async def _send_chunks_loop(self) -> None:
        """Coroutine that drains the pending-chunk deque and sends over WebSocket."""
        try:
            while True:
                if not self._pending_chunks:
                    await asyncio.sleep(0.005)
                    continue
                chunk = self._pending_chunks.popleft()
                if self._stream_client is not None:
                    await self._stream_client.send_chunk(chunk, fmt="pcm")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Audio chunk sender error: %s", exc)

    async def _finish_stream(self) -> None:
        """Called on hotkey release: flush remaining chunks, send finish, get final result."""
        # Allow a brief moment for queued chunks to drain.
        for _ in range(50):
            if not self._pending_chunks:
                break
            await asyncio.sleep(0.005)

        if self._sender_task is not None and not self._sender_task.done():
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
            self._sender_task = None

        if self._stream_client is not None:
            try:
                await self._stream_client.send_json({"type": "finish"})
            except Exception as exc:
                logger.warning("Failed to send finish: %s", exc)

        # Wait for the streaming reader task to produce the final result.
        if self._streaming_task is not None and self._stream_final_event is not None:
            try:
                await asyncio.wait_for(
                    self._stream_final_event.wait(), timeout=self.config.asr_timeout
                )
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for final streaming result")
                self._streaming_error = "Timeout waiting for final result"
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass
            self._streaming_task = None

        await self._finalize_recognition()
        if self._stream_client is not None:
            await self._stream_client.close()
            self._stream_client = None

    async def _finalize_recognition(self) -> None:
        """Broadcast final result, type any remaining text, and reset state."""
        from typing import Literal

        state: Literal["idle", "recording", "recognizing", "error"] = "idle"
        if self._streaming_error:
            self._state = "idle"
            state = "error"
            self._broadcast_state(state, "⚠️ ASR 错误")
            self._broadcast_recognized("", error=self._streaming_error)
            logger.error("❌ ASR 识别失败: %s", self._streaming_error)
        elif self._stream_final_result:
            result = self._stream_final_result
            self._state = "idle"
            state = "idle"
            self._broadcast_state(state, "🎤 就绪")
            self._broadcast_recognized(result.text)
            logger.info('✅ ASR 识别完成: "%s"', result.text)
            if not self._clients:
                self._type_text_incremental(result.text)
            self._typed_text = ""
        else:
            self._state = "idle"
            state = "error"
            self._broadcast_state(state, "⚠️ 无识别结果")
            logger.error("❌ ASR 未返回最终结果")

    def _type_text_incremental(self, new_text: str) -> None:
        """Type only the characters in ``new_text`` that have not been typed yet.

        If the ASR model revises earlier characters, backspace the changed
        portion and re-type the corrected suffix. This keeps the visual output
        in sync with the latest streaming result.
        """
        if self._clients:
            # An external client is connected; let it handle output.
            return
        if new_text == self._typed_text:
            return
        old = self._typed_text
        common = 0
        for a, b in zip(old, new_text):
            if a != b:
                break
            common += 1
        to_delete = len(old) - common
        to_type = new_text[common:]
        self._typed_text = new_text
        if to_delete or to_type:
            self._loop.run_in_executor(None, _type_incremental_x11, to_delete, to_type)

    async def _cleanup_streaming(self) -> None:
        """Cancel streaming tasks and close the client without typing anything."""
        if self._sender_task is not None and not self._sender_task.done():
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
            self._sender_task = None
        if self._streaming_task is not None and not self._streaming_task.done():
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass
            self._streaming_task = None
        if self._stream_client is not None:
            try:
                await self._stream_client.close()
            except Exception as exc:
                logger.warning("Failed to close stream client during cleanup: %s", exc)
            self._stream_client = None
        self._pending_chunks.clear()

    def _broadcast_state(self, state: str, message: str | None) -> None:
        """Broadcast a state update to all connected IPC clients."""
        msg = StateUpdate(state=state, message=message).to_json()
        self._broadcast(msg)

    def _broadcast_recognized(self, text: str, error: str | None = None) -> None:
        """Broadcast recognized text (or error) to all connected IPC clients."""
        msg = RecognizedText(text=text, error=error).to_json()
        self._broadcast(msg)

    def _broadcast(self, msg: str) -> None:
        """Send a JSON-line message to every connected IPC client."""
        data = (msg + "\n").encode("utf-8")
        loop = self._loop
        with self._lock:
            writers = list(self._clients)
        for writer in writers:
            try:
                writer.write(data)
                loop.call_soon_threadsafe(lambda w=writer: asyncio.create_task(w.drain()))
            except Exception as exc:
                logger.warning("Failed to send to client: %s", exc)

    def _on_client_connected(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a new IPC client connection."""
        with self._lock:
            self._clients.add(writer)
        logger.info("IPC client connected")

        async def read_loop():
            try:
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    text = line.decode("utf-8").strip()
                    if not text:
                        continue
                    try:
                        msg = parse_message(text)
                        if isinstance(msg, HotkeyCommand):
                            self._on_hotkey_message(msg)
                    except Exception:
                        logger.warning("Invalid message from client: %s", text)
            except Exception:
                pass
            finally:
                with self._lock:
                    self._clients.discard(writer)
                try:
                    writer.close()
                except Exception:
                    pass

        asyncio.create_task(read_loop())

    def _shutdown(self) -> None:
        """Cancel tasks and close resources on SIGTERM/SIGINT."""
        logger.info("Shutting down daemon")
        if self._streaming_task is not None:
            self._streaming_task.cancel()
        if self._sender_task is not None:
            self._sender_task.cancel()
        if self._serve_task is not None:
            self._serve_task.cancel()
        if self._server:
            self._server.close()
        self.hotkey.stop()
        self.recorder.close()


async def main() -> None:
    """Entry point: load config and run the daemon forever."""
    config = IMEConfig.load()
    daemon = VoiceInputDaemon(config)
    await daemon.start()
    await daemon.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
