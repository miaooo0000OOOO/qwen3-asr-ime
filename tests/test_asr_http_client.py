"""Tests for ASRHttpClient."""
from unittest.mock import patch

import pytest

from qwen3_asr_ime.daemon.asr_client import ASRHttpClient


class _FakeResponse:
    """A fake aiohttp response for testing."""

    def __init__(self, status=200, body=None, error_text=None):
        self.status = status
        self._body = body or {}
        self._error_text = error_text or ""

    async def json(self):
        return self._body

    async def text(self):
        return self._error_text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeSession:
    """A fake aiohttp ClientSession that returns controlled responses."""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def post(self, url, data=None, headers=None, params=None, timeout=None):
        return self._response


class TestASRHttpClient:
    """Unit tests for ASRHttpClient."""

    @pytest.fixture
    def client(self):
        return ASRHttpClient("http://127.0.0.1:8000", api_key="test", timeout=5.0)

    def test_transcribe_url(self, client):
        """URL is constructed correctly from endpoint."""
        assert client._transcribe_url == "http://127.0.0.1:8000/v1/asr/transcribe"

    @pytest.mark.asyncio
    async def test_transcribe_success(self, client):
        """Successful transcription returns text with final=True."""
        fake_resp = _FakeResponse(status=200, body={"text": "你好世界", "language": "zh"})
        fake_session = _FakeSession(fake_resp)

        with patch("aiohttp.ClientSession", return_value=fake_session):
            result = await client.transcribe(b"fake_wav_data")
            assert result.text == "你好世界"
            assert result.language == "zh"
            assert result.final is True
            assert result.error is None

    @pytest.mark.asyncio
    async def test_transcribe_server_error(self, client):
        """Server error returns result with error set."""
        fake_resp = _FakeResponse(status=500, error_text="Internal Server Error")
        fake_session = _FakeSession(fake_resp)

        with patch("aiohttp.ClientSession", return_value=fake_session):
            result = await client.transcribe(b"fake_wav_data")
            assert result.error is not None
            assert "500" in result.error
            assert result.final is False

    @pytest.mark.asyncio
    async def test_transcribe_network_error(self, client):
        """Network error returns result with error set."""
        with patch("aiohttp.ClientSession", side_effect=ConnectionError("Connection refused")):
            result = await client.transcribe(b"fake_wav_data")
            assert result.error is not None
            assert "Connection refused" in result.error
