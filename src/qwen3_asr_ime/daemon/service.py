"""Main voice input daemon.

This daemon ties together the global hotkey listener, microphone recorder,
ASR client (streaming or non-streaming), IPC server, and X11 text injection.
It manages backend lifecycle (spawn/sleep/restart) and watches the config
file for live changes.

High-level flow:

- **Non-streaming (offline, default):** User holds hotkey → record. Release →
  POST complete WAV to ``/v1/asr/transcribe`` → type final result once.
  No real-time partial output.

- **Streaming:** User holds → WebSocket to server, send chunks, type
  incremental partial results. Release → finish → final result.

Daemon-level features:
- Auto-sleep: stop backend after idle timeout to save CPU/GPU memory.
- Config watching: poll config mtime every 5s; reload on change; restart
  backend if backend-relevant fields changed.
- Fail-loudly: any unexpected error calls ``sys.exit(1)``.
"""

from __future__ import annotations

import asyncio
import collections
import os
import signal
import sys
import threading
from pathlib import Path

from qwen3_asr_ime.common.config import ConfigWatcher, IMEConfig
from qwen3_asr_ime.common.logger import get_logger
from qwen3_asr_ime.common.protocol import (
    HotkeyCommand,
    RecognizedText,
    StateUpdate,
    parse_message,
)
from qwen3_asr_ime.daemon.asr_client import (
    ASRHttpClient,
    ASRResult,
    ASRStreamClient,
)
from qwen3_asr_ime.daemon.backend_manager import BackendManager
from qwen3_asr_ime.daemon.hotkey import HotkeyEvent, create_hotkey_listener
from qwen3_asr_ime.daemon.recorder import AudioConfig, Recorder

logger = get_logger(__name__)


def _type_incremental_x11(to_delete: int, text: str) -> None:
    """Delete ``to_delete`` characters then type ``text`` into the active window."""
    import time

    import Xlib.display
    import Xlib.X
    import Xlib.XK
    from pynput._util.xorg import char_to_keysym

    display = Xlib.display.Display()
    try:
        focus = display.get_input_focus().focus
        root = display.screen().root
        send_event = getattr(
            focus, "send_event", lambda event: display.send_event(focus, event)
        )

        def _send_keycode(keycode: int, state: int = 0) -> None:
            for is_press in (True, False):
                event_cls = (
                    Xlib.display.event.KeyPress
                    if is_press
                    else Xlib.display.event.KeyRelease
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

        backspace_kc = display.keysym_to_keycode(Xlib.XK.XK_BackSpace)
        for _ in range(to_delete):
            _send_keycode(backspace_kc)
            time.sleep(0.005)

        if not text:
            return

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

        display.change_keyboard_mapping(scratch_kc, [(Xlib.XK.NoSymbol,)])
        display.sync()
    finally:
        display.close()


class VoiceInputDaemon:
    """Daemon that orchestrates hotkey, recording, ASR, IPC, typing, backend lifecycle.

    Attributes:
        _config_watcher: Watches config file for changes.
        _config: Current effective config (may be updated at runtime).
        _backend_mgr: Manages the ASR backend child process.
        recorder: ``Recorder`` instance for microphone capture.
        hotkey: Global hotkey listener (pynput-based).
    """

    def __init__(self, config_watcher: ConfigWatcher | IMEConfig):
        if isinstance(config_watcher, IMEConfig):
            # Backward compatibility for tests — wrap IMEConfig in a dummy.
            self._config_watcher = config_watcher  # type: ignore[assignment]
            self._config = config_watcher
        else:
            self._config_watcher = config_watcher
            self._config = config_watcher.config
        self._backend_mgr = BackendManager()

        self.recorder = Recorder(
            AudioConfig(
                sample_rate=self._config.audio_sample_rate,
                channels=self._config.audio_channels,
                chunk_ms=self._config.audio_chunk_ms,
            )
        )
        self.hotkey = create_hotkey_listener(
            self._config.hotkey_device,
            self._config.hotkey_key,
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
        self._max_pending_chunks = 200
        self._streaming_error: str | None = None
        self._current_text: str = ""
        self._typed_text: str = ""
        self._stream_final_result: ASRResult | None = None
        self._stream_final_event: asyncio.Event | None = None

        # Offline ASR state
        self._offline_recording: bytes = b""

        # Background tasks
        self._config_watch_task: asyncio.Task[None] | None = None
        self._idle_check_task: asyncio.Task[None] | None = None

        # Error counter for consecutive failures
        self._consecutive_errors: int = 0
        self._max_consecutive_errors: int = 5

    async def start(self) -> None:
        """Create IPC socket, start backend, launch hotkey, begin config/idle watchers."""
        socket_path = Path(self._config.ipc_socket_path)
        socket_path.parent.mkdir(parents=True, exist_ok=True)
        if socket_path.exists():
            socket_path.unlink()
        self._loop = asyncio.get_running_loop()

        # Spawn backend and wait for health
        await self._backend_mgr.spawn(self._config)
        await self._backend_mgr.wait_ready()

        # Start IPC server
        self._server = await asyncio.start_unix_server(
            self._on_client_connected,
            path=str(socket_path),
        )
        os.chmod(socket_path, 0o600)

        # Start hotkey listener
        self.hotkey.start()

        # Launch background watchers (skip in test mode when config_watcher is IMEConfig)
        if isinstance(self._config_watcher, ConfigWatcher):
            self._config_watch_task = asyncio.create_task(
                self._config_watcher.watch_loop(self._on_config_change)
            )
            self._idle_check_task = asyncio.create_task(self._idle_check_loop())

        logger.info(
            "Daemon started (mode=%s, backend=%s, model=%s)",
            self._config.asr_mode,
            self._config.asr_backend,
            self._config.asr_model,
        )

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
        asyncio.run_coroutine_threadsafe(self._handle_hotkey(event), self._loop)

    async def _handle_hotkey(self, event: HotkeyEvent) -> None:
        """State machine driven by press/release hotkey events.

        Routes between offline (non-streaming) and streaming modes based on config.
        """
        try:
            if event.action == "press" and self._state == "idle":
                # Wake up backend if it was sleeping
                if not self._backend_mgr.is_running:
                    await self._backend_mgr.spawn(self._config)
                    await self._backend_mgr.wait_ready()
                else:
                    self._backend_mgr.touch_activity()

                self._state = "recording"
                self._current_text = ""
                self._typed_text = ""
                self._streaming_error = None
                self._stream_final_result = None
                self._offline_recording = b""
                self._pending_chunks.clear()

                if self._config.asr_mode == "streaming":
                    # Existing streaming flow
                    self._stream_final_event = asyncio.Event()
                    self.recorder.start(chunk_callback=self._on_audio_chunk)
                    self._stream_client = ASRStreamClient(
                        self._config.asr_endpoint,
                        api_key=self._config.asr_api_key,
                        timeout=self._config.asr_timeout,
                    )
                    self._streaming_task = asyncio.create_task(self._run_stream())
                    self._broadcast_state("recording", "🔴 录音中 (流式)")
                    logger.info("⬇ Ctrl 按下 → 开始录音并建立流式 ASR 连接")
                else:
                    # Offline mode: just record, no WebSocket
                    self.recorder.start()  # no chunk_callback needed
                    self._broadcast_state("recording", "🔴 录音中 (离线)")
                    logger.info("⬇ Ctrl 按下 → 开始录音 (离线模式)")

            elif event.action == "release" and self._state == "recording":
                self._state = "recognizing"
                self._broadcast_state("recognizing", "🔄 识别中...")
                audio_bytes = self.recorder.stop()
                dur = len(audio_bytes) / 32000
                logger.info(
                    "⬆ Ctrl 松开 → 停止录音 (%.1f 秒, %d KB)",
                    dur,
                    len(audio_bytes) // 1024,
                )
                if self._config.asr_mode == "offline":
                    self._offline_recording = audio_bytes
                    asyncio.create_task(self._run_offline_recognition())
                else:
                    asyncio.create_task(self._finish_stream())

            elif event.action == "interrupt" and self._state == "recording":
                logger.info("⛔ 检测到组合键，中断语音输入")
                self._state = "idle"
                self._broadcast_state("idle", "🎤 就绪")
                self.recorder.stop()
                self._pending_chunks.clear()
                if self._config.asr_mode == "streaming":
                    asyncio.create_task(self._cleanup_streaming())
        except Exception:
            logger.exception("❌ 热键处理错误")
            self._consecutive_errors += 1
            if self._consecutive_errors >= self._max_consecutive_errors:
                logger.critical("连续 %d 次错误，退出程序", self._consecutive_errors)
                sys.exit(1)

    def _on_audio_chunk(self, chunk_bytes: bytes) -> None:
        """Called from sounddevice thread for each audio chunk."""
        if len(self._pending_chunks) >= self._max_pending_chunks:
            self._pending_chunks.popleft()
        self._pending_chunks.append(chunk_bytes)

    # ── Streaming ASR (existing, mostly unchanged) ──────────────────────

    async def _run_stream(self) -> None:
        """Manage the WebSocket streaming ASR session."""
        if self._stream_client is None:
            return
        try:
            await self._stream_client.connect()
            self._sender_task = asyncio.create_task(self._send_chunks_loop())
            async for result in self._stream_client.iterate():
                if result.error:
                    self._streaming_error = result.error
                    logger.error("Streaming ASR error: %s", result.error)
                    break
                if result.final:
                    self._stream_final_result = result
                    break
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

        if self._streaming_task is not None and self._stream_final_event is not None:
            try:
                await asyncio.wait_for(
                    self._stream_final_event.wait(), timeout=self._config.asr_timeout
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
        if self._streaming_error:
            self._state = "idle"
            self._broadcast_state("error", "⚠️ ASR 错误")
            self._broadcast_recognized("", error=self._streaming_error)
            logger.error("❌ ASR 识别失败: %s", self._streaming_error)
            self._consecutive_errors += 1
        elif self._stream_final_result:
            result = self._stream_final_result
            self._state = "idle"
            self._broadcast_state("idle", "🎤 就绪")
            self._broadcast_recognized(result.text)
            logger.info('✅ ASR 识别完成: "%s"', result.text)
            if not self._clients:
                self._type_text_incremental(result.text)
            self._typed_text = ""
            self._consecutive_errors = 0
        else:
            self._state = "idle"
            self._broadcast_state("error", "⚠️ 无识别结果")
            logger.error("❌ ASR 未返回最终结果")
            self._consecutive_errors += 1

        if self._consecutive_errors >= self._max_consecutive_errors:
            logger.critical("连续 %d 次错误，退出程序", self._consecutive_errors)
            sys.exit(1)

    def _type_text_incremental(self, new_text: str) -> None:
        """Type only the characters in ``new_text`` that have not been typed yet."""
        if self._clients:
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
                logger.warning(
                    "Failed to close stream client during cleanup: %s", exc
                )
            self._stream_client = None
        self._pending_chunks.clear()

    # ── Offline (non-streaming) ASR ────────────────────────────────────

    async def _run_offline_recognition(self) -> None:
        """Send the complete recording to the HTTP ASR endpoint and handle result."""
        if not self._offline_recording:
            self._state = "idle"
            self._broadcast_state("idle", "🎤 就绪")
            return

        try:
            client = ASRHttpClient(
                self._config.asr_endpoint,
                api_key=self._config.asr_api_key,
                timeout=self._config.asr_timeout,
            )
            result = await client.transcribe(self._offline_recording)

            if result.error:
                self._streaming_error = result.error
                self._consecutive_errors += 1
                logger.error("❌ 离线 ASR 识别失败: %s", result.error)
                self._state = "idle"
                self._broadcast_state("error", "⚠️ ASR 错误")
                self._broadcast_recognized("", error=result.error)
                if self._consecutive_errors >= self._max_consecutive_errors:
                    logger.critical("连续 %d 次识别失败，退出程序", self._consecutive_errors)
                    sys.exit(1)
            else:
                self._consecutive_errors = 0
                self._state = "idle"
                self._broadcast_state("idle", "🎤 就绪")
                self._broadcast_recognized(result.text)
                logger.info('✅ ASR 识别完成: "%s"', result.text)
                if not self._clients:
                    self._type_text_final(result.text)
        except Exception as exc:
            logger.exception("❌ 离线 ASR 未预期错误")
            self._consecutive_errors += 1
            self._state = "idle"
            self._broadcast_state("error", "⚠️ ASR 错误")
            if self._consecutive_errors >= self._max_consecutive_errors:
                sys.exit(1)
        finally:
            self._backend_mgr.touch_activity()

    def _type_text_final(self, text: str) -> None:
        """Type the final recognized text into the active X11 window.

        Uses ``_type_incremental_x11`` with zero deletion because
        ``Controller().type()`` has keycode borrowing limits that fail
        for long Chinese strings.
        """
        if self._clients:
            return
        if not text:
            return
        self._typed_text = text
        self._loop.run_in_executor(None, _type_incremental_x11, 0, text)

    # ── Config watching and idle management ────────────────────────────

    def _on_config_change(self, new_config: IMEConfig) -> None:
        """Handle config file changes detected by ConfigWatcher.

        Compares old and new config. If backend-relevant fields changed,
        schedules a backend restart. Daemon-local fields are applied
        immediately.
        """
        old = self._config
        self._config = new_config

        # Check if backend-relevant config has changed
        backend_keys = (
            "asr_mode",
            "asr_model",
            "asr_backend",
            "asr_device",
            "asr_endpoint",
            "asr_auto_sleep_time",
            "asr_backend_wait_timeout",
        )
        needs_restart = any(
            getattr(old, k) != getattr(new_config, k) for k in backend_keys
        )

        if needs_restart:
            logger.info("Backend-relevant config changed; scheduling restart")
            asyncio.create_task(self._backend_mgr.restart(new_config))

    async def _idle_check_loop(self) -> None:
        """Periodically check if the backend should be put to sleep."""
        while True:
            await asyncio.sleep(5.0)
            try:
                await self._backend_mgr.check_idle()
            except Exception as exc:
                logger.warning("Idle check error: %s", exc)

    # ── Broadcasting and IPC ────────────────────────────────────────────

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
                loop.call_soon_threadsafe(
                    lambda w=writer: asyncio.create_task(w.drain())
                )
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

    # ── Shutdown ────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        """Cancel tasks and close resources on SIGTERM/SIGINT."""
        logger.info("Shutting down daemon")
        if self._streaming_task is not None:
            self._streaming_task.cancel()
        if self._sender_task is not None:
            self._sender_task.cancel()
        if self._serve_task is not None:
            self._serve_task.cancel()
        if self._config_watch_task is not None and not self._config_watch_task.done():
            self._config_watch_task.cancel()
        if self._idle_check_task is not None and not self._idle_check_task.done():
            self._idle_check_task.cancel()
        if self._server:
            self._server.close()
        self.hotkey.stop()
        self.recorder.close()
        # Stop backend asynchronously
        asyncio.ensure_future(self._backend_mgr.stop())


async def main() -> None:
    """Entry point: init ConfigWatcher, create daemon, run forever."""
    watcher = ConfigWatcher()
    daemon = VoiceInputDaemon(watcher)
    await daemon.start()
    await daemon.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
