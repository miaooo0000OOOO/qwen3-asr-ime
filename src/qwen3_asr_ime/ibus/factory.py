import gi

gi.require_version("IBus", "1.0")  # noqa: E402
from gi.repository import IBus  # noqa: E402

from qwen3_asr_ime.ibus.engine import Qwen3ASREngine  # noqa: E402


class EngineFactory(IBus.Factory):
    __gtype_name__ = "Qwen3ASREngineFactory"

    def __init__(self, bus):
        super().__init__(connection=bus.get_connection())
        self.bus = bus

    def do_create_engine(self, engine_name):
        if engine_name == "qwen3-asr-ime":
            return Qwen3ASREngine()
        return None
