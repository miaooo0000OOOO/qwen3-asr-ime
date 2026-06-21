import responses

from qwen3_asr_ime.daemon.asr_client import ASRClient


@responses.activate
def test_recognize_success():
    responses.post(
        "http://127.0.0.1:8000/v1/chat/completions",
        json={
            "choices": [{
                "message": {
                    "content": "<|language|>Chinese<|/language|><|text|>你好世界<|/text|>"
                }
            }]
        },
        status=200,
    )
    client = ASRClient("http://127.0.0.1:8000")
    result = client.recognize(b"fake wav data")
    assert result.text == "你好世界"
    assert result.language == "Chinese"
    assert result.error is None


@responses.activate
def test_recognize_failure():
    responses.post(
        "http://127.0.0.1:8000/v1/chat/completions",
        status=500,
    )
    client = ASRClient("http://127.0.0.1:8000", timeout=2.0)
    result = client.recognize(b"fake wav data")
    assert result.text == ""
    assert result.error is not None


@responses.activate
def test_recognize_url_not_doubled():
    responses.post(
        "http://127.0.0.1:8000/v1/chat/completions",
        json={
            "choices": [{"message": {"content": "<|text|>ok<|/text|>"}}]
        },
        status=200,
    )
    client = ASRClient("http://127.0.0.1:8000")
    client.recognize(b"data")
    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == "http://127.0.0.1:8000/v1/chat/completions"
