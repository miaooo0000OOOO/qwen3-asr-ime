import sys

import gi

gi.require_version("IBus", "1.0")  # noqa: E402
from gi.repository import GLib, GObject, IBus  # noqa: E402

from qwen3_asr_ime.common.logger import get_logger  # noqa: E402
from qwen3_asr_ime.ibus.factory import EngineFactory  # noqa: E402

logger = get_logger(__name__)


def main():
    IBus.init()
    bus = IBus.Bus()
    if not bus.is_connected():
        logger.error("Cannot connect to IBus daemon")
        sys.exit(1)

    factory = EngineFactory(bus)
    factory.add_engine("qwen3-asr-ime", GObject.type_from_name("Qwen3ASREngine"))

    loop = GLib.MainLoop()
    logger.info("Qwen3-ASR IME engine started")
    loop.run()


if __name__ == "__main__":
    main()
