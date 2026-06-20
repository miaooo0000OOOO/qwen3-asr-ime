import responses

from qwen3_asr_ime.daemon.asr_client import ASRClient


@responses.activate
def test_recognize_success():
    responses.post(
        "http://localhost:8000/v1/audio/transcriptions",
        json={"text": "你好"},
        status=200,
    )
    client = ASRClient("http://localhost:8000")
    result = client.recognize(b"fake wav data")
    assert result.text == "你好"
    assert result.error is None


@responses.activate
def test_recognize_failure():
    responses.post(
        "http://localhost:8000/v1/audio/transcriptions",
        status=500,
    )
    client = ASRClient("http://localhost:8000", timeout=2.0)
    result = client.recognize(b"fake wav data")
    assert result.text == ""
    assert result.error is not None
