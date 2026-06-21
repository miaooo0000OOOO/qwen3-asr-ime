from __future__ import annotations

import asyncio
from pathlib import Path

import gi

gi.require_version("IBus", "1.0")  # noqa: E402
from gi.repository import GLib, IBus  # noqa: E402

from qwen3_asr_ime.common.config import IMEConfig  # noqa: E402
from qwen3_asr_ime.common.logger import get_logger  # noqa: E402
from qwen3_asr_ime.common.protocol import (  # noqa: E402
    RecognizedText,
    StateUpdate,
    parse_message,
)

logger = get_logger(__name__)

# Status → emoji label displayed in IBus panel
STATUS_LABELS: dict[str, str] = {
    "idle": "🎤 就绪",
    "recording": "🔴 录音中",
    "recognizing": "🔄 识别中",
    "error": "⚠️ 错误",
}


class Qwen3ASREngine(IBus.Engine):
    __gtype_name__ = "Qwen3ASREngine"

    def __init__(self):
        super().__init__()
        self.config = IMEConfig.load()
        self._reader = None
        self._writer = None
        self._prop_list = IBus.PropList()
        self._prop_list.append(self._make_prop("idle"))
        GLib.timeout_add(100, self._connect_to_daemon)

    @staticmethod
    def _make_prop(state: str) -> IBus.Property:
        return IBus.Property(
            key="status",
            type=IBus.PropType.NORMAL,
            label=IBus.Text.new_from_string(STATUS_LABELS.get(state, state)),
        )

    def _set_state(self, state: str) -> None:
        self.update_property("status", self._make_prop(state))

    def _connect_to_daemon(self):
        socket_path = Path(self.config.ipc_socket_path)
        if not socket_path.exists():
            return True
        try:
            asyncio.ensure_future(self._connect(socket_path))
            asyncio.ensure_future(self._read_loop())
            return False
        except Exception as exc:
            logger.warning("Failed to connect to daemon: %s", exc)
            return True

    async def _connect(self, socket_path: Path):
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        self._reader = reader
        self._writer = writer
        self._set_state("idle")
        logger.info("Connected to daemon")

    async def _read_loop(self):
        while True:
            try:
                line = await self._reader.readline()
                if not line:
                    break
                msg = parse_message(line.decode("utf-8"))
                GLib.idle_add(self._handle_message, msg)
            except Exception as exc:
                logger.error("Read loop error: %s", exc)
                break

    def _handle_message(self, msg):
        if isinstance(msg, RecognizedText):
            if msg.error:
                self._set_state("error")
            elif msg.text:
                self.commit_text(IBus.Text.new_from_string(msg.text))
                self._set_state("idle")
        elif isinstance(msg, StateUpdate):
            self._set_state(msg.state)
        return False

    def do_focus_in(self):
        self.register_properties(self._prop_list)

    def do_focus_out(self):
        pass

    def do_property_activate(self, prop_name: str, prop_state: int):
        logger.info("Property activated: %s", prop_name)
