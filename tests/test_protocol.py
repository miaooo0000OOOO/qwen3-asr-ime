import pytest

from qwen3_asr_ime.common.protocol import (
    RecognizedText,
    StateUpdate,
    parse_message,
)


def test_recognized_text_roundtrip():
    msg = RecognizedText(text="你好 world", confidence=0.95)
    parsed = parse_message(msg.to_json())
    assert isinstance(parsed, RecognizedText)
    assert parsed.text == "你好 world"
    assert parsed.confidence == pytest.approx(0.95)


def test_state_update_roundtrip():
    msg = StateUpdate(state="recording", message="开始录音")
    parsed = parse_message(msg.to_json())
    assert isinstance(parsed, StateUpdate)
    assert parsed.state == "recording"
    assert parsed.message == "开始录音"


def test_parse_unknown_message_type():
    with pytest.raises(ValueError, match="Unknown message type"):
        parse_message('{"type": "unknown"}')


def test_parse_non_object_message():
    with pytest.raises(ValueError, match="Message must be a JSON object"):
        parse_message("[]")


def test_parse_invalid_state_value():
    with pytest.raises(ValueError, match="Invalid state"):
        StateUpdate.from_dict({"type": "state", "state": "invalid"})
