from unittest.mock import MagicMock, patch

import pytest

import filetranscribe
import transcribe

CONFIG = {
    "whisper_binary": "/opt/homebrew/bin/whisper-cli",
    "whisper_model_path": "/config/models/ggml-medium.bin",
    "ffmpeg_binary": "/opt/homebrew/bin/ffmpeg",
}


@patch("filetranscribe.subprocess.run", side_effect=FileNotFoundError("no such file"))
def test_transcribe_file_raises_error_when_ffmpeg_missing(mock_run):
    with pytest.raises(filetranscribe.FileTranscribeError):
        filetranscribe.transcribe_file("/tmp/some.mp4", CONFIG)


@patch("filetranscribe.subprocess.run")
def test_transcribe_file_raises_error_when_ffmpeg_conversion_fails(mock_run):
    mock_run.return_value = MagicMock(returncode=1, stderr="invalid data found")
    with pytest.raises(filetranscribe.FileTranscribeError):
        filetranscribe.transcribe_file("/tmp/some.mp4", CONFIG)


@patch("filetranscribe.transcribe")
@patch("filetranscribe.subprocess.run")
def test_transcribe_file_calls_transcribe_on_converted_wav(mock_run, mock_transcribe):
    mock_run.return_value = MagicMock(returncode=0, stderr="")
    mock_transcribe.return_value = ("hello world", "en")

    text, lang = filetranscribe.transcribe_file("/tmp/some.mp4", CONFIG)

    assert text == "hello world"
    assert lang == "en"
    # ffmpeg must have been invoked with the input path
    ffmpeg_cmd = mock_run.call_args[0][0]
    assert "/tmp/some.mp4" in ffmpeg_cmd
    # transcribe() must have been called with a wav path, not the original file
    called_wav_path = mock_transcribe.call_args[0][0]
    assert called_wav_path.endswith(".wav")


@patch("filetranscribe.transcribe", side_effect=transcribe.TranscribeError("boom"))
@patch("filetranscribe.subprocess.run")
def test_transcribe_file_wraps_transcribe_error(mock_run, mock_transcribe):
    mock_run.return_value = MagicMock(returncode=0, stderr="")
    with pytest.raises(filetranscribe.FileTranscribeError):
        filetranscribe.transcribe_file("/tmp/some.mp4", CONFIG)
