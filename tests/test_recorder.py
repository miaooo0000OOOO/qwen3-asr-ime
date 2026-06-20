from unittest.mock import MagicMock, patch

from qwen3_asr_ime.daemon.recorder import AudioConfig, Recorder


@patch("qwen3_asr_ime.daemon.recorder.pyaudio.PyAudio")
def test_recorder_start_stop(mock_pyaudio_cls):
    mock_audio = MagicMock()
    mock_stream = MagicMock()
    mock_pyaudio_cls.return_value = mock_audio
    mock_audio.open.return_value = mock_stream
    mock_audio.get_sample_size.return_value = 2

    recorder = Recorder(AudioConfig(sample_rate=16000, chunk_ms=20))
    recorder.start()
    assert recorder._is_recording is True

    mock_stream.stop_stream.assert_not_called()
    recorder.stop()
    assert recorder._is_recording is False
    mock_stream.stop_stream.assert_called_once()


@patch("qwen3_asr_ime.daemon.recorder.pyaudio.PyAudio")
def test_recorder_wav_output(mock_pyaudio_cls):
    mock_audio = MagicMock()
    mock_stream = MagicMock()
    mock_pyaudio_cls.return_value = mock_audio
    mock_audio.open.return_value = mock_stream
    mock_audio.get_sample_size.return_value = 2

    recorder = Recorder(AudioConfig(sample_rate=16000, chunk_ms=20))
    recorder.start()
    callback = mock_audio.open.call_args[1]["stream_callback"]
    callback(b"\x00\x01", None, None, None)
    wav_bytes = recorder.stop()

    assert wav_bytes.startswith(b"RIFF")
