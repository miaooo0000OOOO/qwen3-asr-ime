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


@responses.activate
def test_recognize_with_base_url():
    """Test with a full base URL (like default config) — no path doubling."""
    responses.post(
        "http://127.0.0.1:8000/v1/audio/transcriptions",
        json={"text": "你好"},
        status=200,
    )
    # Default config uses just base URL now
    client = ASRClient("http://127.0.0.1:8000")
    result = client.recognize(b"fake test data")
    assert result.text == "你好"
    assert result.error is None


@responses.activate
def test_recognize_url_not_doubled():
    """Verify URL is exactly endpoint + /v1/audio/transcriptions, not doubled."""
    responses.post(
        "http://127.0.0.1:8000/v1/audio/transcriptions",
        json={"text": "ok"},
        status=200,
    )
    client = ASRClient("http://127.0.0.1:8000")
    client.recognize(b"data")
    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == "http://127.0.0.1:8000/v1/audio/transcriptions"
