from unittest.mock import MagicMock, patch, sentinel

import numpy as np

from qwen3_asr_ime.daemon.recorder import AudioConfig, Recorder


@patch("qwen3_asr_ime.daemon.recorder.sd.InputStream")
def test_recorder_start_stop(mock_stream_cls):
    mock_stream = MagicMock()
    mock_stream_cls.return_value = mock_stream

    recorder = Recorder(AudioConfig(sample_rate=16000, chunk_ms=20))
    recorder.start()
    assert recorder._is_recording is True

    mock_stream.stop.assert_not_called()
    recorder.stop()
    assert recorder._is_recording is False
    mock_stream.stop.assert_called_once()
    mock_stream.close.assert_called_once()


@patch("qwen3_asr_ime.daemon.recorder.sd.InputStream")
@patch("qwen3_asr_ime.daemon.recorder.np.concatenate")
def test_recorder_wav_output(mock_concat, mock_stream_cls):
    mock_stream = MagicMock()
    mock_stream_cls.return_value = mock_stream
    mock_concat.return_value = np.frombuffer(b"\x00\x01" * 20, dtype=np.int16)

    recorder = Recorder(AudioConfig(sample_rate=16000, chunk_ms=20))
    recorder.start()
    callback = mock_stream_cls.call_args[1]["callback"]
    callback(np.zeros((10, 1), dtype=np.int16), 10, None, None)
    wav_bytes = recorder.stop()

    assert wav_bytes.startswith(b"RIFF")
