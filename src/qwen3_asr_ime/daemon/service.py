from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from qwen3_asr_ime.common.config import IMEConfig
from qwen3_asr_ime.common.logger import get_logger
from qwen3_asr_ime.common.protocol import HotkeyCommand, RecognizedText, StateUpdate, parse_message
from qwen3_asr_ime.daemon.asr_client import ASRClient
from qwen3_asr_ime.daemon.hotkey import HotkeyEvent, create_hotkey_listener
from qwen3_asr_ime.daemon.recorder import AudioConfig, Recorder

logger = get_logger(__name__)


def _type_text_uinput(text: str) -> None:
    """Type text using xdotool (reliable, no clipboard)."""
    import subprocess
    try:
        subprocess.run(
            ["xdotool", "type", "--delay", "10", text],
            check=True, timeout=10, capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("xdotool failed: %s", exc)

class VoiceInputDaemon:
    def __init__(self, config: IMEConfig):
        self.config = config
        self.recorder = Recorder(
            AudioConfig(
                sample_rate=config.audio_sample_rate,
                channels=config.audio_channels,
                chunk_ms=config.audio_chunk_ms,
            )
        )
        self.asr = ASRClient(config.asr_endpoint, api_key=config.asr_api_key)
        self.hotkey = create_hotkey_listener(
            config.hotkey_device,
            config.hotkey_key,
            self._on_hotkey,
        )
        self._clients: set[asyncio.StreamWriter] = set()
        self._state: str = "idle"
        self._server = None

    async def start(self) -> None:
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
        """Handle hotkey commands from GNOME Shell extension over IPC."""
        logger.debug("IPC hotkey: %s", msg.action)
        self._on_hotkey(HotkeyEvent(action=msg.action))

    async def run_forever(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown)
        async with self._server:
            await self._server.serve_forever()

    def _on_hotkey(self, event: HotkeyEvent) -> None:
        try:
            if event.action == "press" and self._state == "idle":
                self._state = "recording"
                self.recorder.start()
                self._broadcast_state("recording", "🔴 录音中")
                logger.info("⬇ Ctrl 按下 → 开始录音")
            elif event.action == "release" and self._state == "recording":
                self._state = "recognizing"
                self._broadcast_state("recognizing", "🔄 识别中...")
                audio_bytes = self.recorder.stop()
                dur = len(audio_bytes) / 32000  # 16kHz 16bit 单声道
                logger.info(
                    "⬆ Ctrl 松开 → 停止录音 (%.1f 秒, %d KB)", dur, len(audio_bytes) // 1024
                )
                logger.info("➡ 调用 ASR 模型: %s", self.config.asr_endpoint)
                self._loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self._recognize(audio_bytes))
                )
        except Exception:
            logger.exception("❌ 热键处理错误")

    async def _recognize(self, audio_bytes: bytes) -> None:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        result = await loop.run_in_executor(None, self.asr.recognize, audio_bytes)
        elapsed = loop.time() - t0
        if result.error:
            self._state = "idle"
            self._broadcast_state("error", "⚠️ ASR 错误")
            logger.error("❌ ASR 识别失败 (%d ms): %s", int(elapsed * 1000), result.error)
        else:
            self._state = "idle"
            self._broadcast_state("idle", "🎤 就绪")
            self._broadcast_recognized(result.text)
            logger.info('✅ ASR 识别完成 (%d ms): "%s"', int(elapsed * 1000), result.text)
            # Fallback: type text directly via uinput if no IBus client is connected
            if not self._clients:
                loop.run_in_executor(None, _type_text_uinput, result.text)

    def _broadcast_state(self, state: str, message: str | None) -> None:
        msg = StateUpdate(state=state, message=message).to_json()
        self._broadcast(msg)

    def _broadcast_recognized(self, text: str, error: str | None = None) -> None:
        msg = RecognizedText(text=text, error=error).to_json()
        self._broadcast(msg)

    def _broadcast(self, msg: str) -> None:
        data = (msg + "\n").encode("utf-8")
        loop = self._loop
        for writer in list(self._clients):
            try:
                writer.write(data)
                loop.call_soon_threadsafe(lambda w=writer: asyncio.create_task(w.drain()))
            except Exception as exc:
                logger.warning("Failed to send to client: %s", exc)

    def _on_client_connected(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._clients.add(writer)
        logger.info("IBus engine connected")

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
                self._clients.discard(writer)
                try:
                    writer.close()
                except Exception:
                    pass

        asyncio.create_task(read_loop())

    def _shutdown(self) -> None:
        logger.info("Shutting down daemon")
        if self._server:
            self._server.close()
        self.hotkey.stop()
        self.recorder.close()


async def main():
    config = IMEConfig.load()
    daemon = VoiceInputDaemon(config)
    await daemon.start()
    await daemon.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
