import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import transcribe

FIXTURES = Path(__file__).parent / "fixtures"

CONFIG = {
    "whisper_binary": "/opt/homebrew/bin/whisper-cli",
    "whisper_model_path": "/config/models/ggml-medium.bin",
}


def test_parse_whisper_json_single_segment():
    text, lang = transcribe._parse_whisper_json(str(FIXTURES / "whisper_output_en.json"))
    assert text == "Hello, this is a test of the transcription pipeline."
    assert lang == "en"


def test_parse_whisper_json_joins_multiple_segments():
    text, lang = transcribe._parse_whisper_json(str(FIXTURES / "whisper_output_multi.json"))
    assert text == "Bugün Kubernetes üzerinde çalıştım."
    assert lang == "tr"


@patch("transcribe.subprocess.run", side_effect=FileNotFoundError("no such file"))
def test_transcribe_raises_transcribe_error_when_binary_missing(mock_run):
    # subprocess.run raises FileNotFoundError (not TranscribeError) when the
    # whisper-cli binary itself doesn't exist - this must not escape as a
    # bare FileNotFoundError.
    with pytest.raises(transcribe.TranscribeError):
        transcribe.transcribe("/tmp/some.wav", CONFIG)


def _fake_run_writing(content_writer):
    def fake_run(cmd, capture_output, text):
        out_prefix = cmd[cmd.index("-of") + 1]
        content_writer(out_prefix + ".json")
        return MagicMock(returncode=0, stderr="")

    return fake_run


def test_transcribe_raises_transcribe_error_on_malformed_output_json():
    fake_run = _fake_run_writing(lambda path: Path(path).write_text("{not valid json"))
    with patch("transcribe.subprocess.run", side_effect=fake_run):
        with pytest.raises(transcribe.TranscribeError):
            transcribe.transcribe("/tmp/some.wav", CONFIG)


def test_transcribe_raises_transcribe_error_on_segment_missing_text_key():
    def write_bad_segment(path):
        with open(path, "w") as f:
            json.dump(
                {"result": {"language": "en"}, "transcription": [{"not_text": "oops"}]}, f
            )

    fake_run = _fake_run_writing(write_bad_segment)
    with patch("transcribe.subprocess.run", side_effect=fake_run):
        with pytest.raises(transcribe.TranscribeError):
            transcribe.transcribe("/tmp/some.wav", CONFIG)
