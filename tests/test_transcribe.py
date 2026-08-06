import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

import transcribe
import whisper_server

FIXTURES = Path(__file__).parent / "fixtures"

CONFIG = {
    "whisper_binary": "/opt/homebrew/bin/whisper-cli",
    "whisper_model_path": "/config/models/ggml-medium.bin",
    "whisper_server_binary": "/opt/homebrew/bin/whisper-server",
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
def test_transcribe_via_subprocess_raises_transcribe_error_when_binary_missing(mock_run):
    # subprocess.run raises FileNotFoundError (not TranscribeError) when the
    # whisper-cli binary itself doesn't exist - this must not escape as a
    # bare FileNotFoundError.
    with pytest.raises(transcribe.TranscribeError):
        transcribe._transcribe_via_subprocess("/tmp/some.wav", CONFIG)


def _fake_run_writing(content_writer):
    def fake_run(cmd, capture_output, text):
        out_prefix = cmd[cmd.index("-of") + 1]
        content_writer(out_prefix + ".json")
        return MagicMock(returncode=0, stderr="")

    return fake_run


def test_transcribe_via_subprocess_raises_transcribe_error_on_malformed_output_json():
    fake_run = _fake_run_writing(lambda path: Path(path).write_text("{not valid json"))
    with patch("transcribe.subprocess.run", side_effect=fake_run):
        with pytest.raises(transcribe.TranscribeError):
            transcribe._transcribe_via_subprocess("/tmp/some.wav", CONFIG)


def test_transcribe_via_subprocess_raises_transcribe_error_on_segment_missing_text_key():
    def write_bad_segment(path):
        with open(path, "w") as f:
            json.dump(
                {"result": {"language": "en"}, "transcription": [{"not_text": "oops"}]}, f
            )

    fake_run = _fake_run_writing(write_bad_segment)
    with patch("transcribe.subprocess.run", side_effect=fake_run):
        with pytest.raises(transcribe.TranscribeError):
            transcribe._transcribe_via_subprocess("/tmp/some.wav", CONFIG)


def _fake_run_capturing_cmd(captured, content_writer):
    def fake_run(cmd, capture_output, text):
        captured.append(cmd)
        out_prefix = cmd[cmd.index("-of") + 1]
        content_writer(out_prefix + ".json")
        return MagicMock(returncode=0, stderr="")

    return fake_run


def _write_minimal_output(path):
    with open(path, "w") as f:
        json.dump({"result": {"language": "en"}, "transcription": [{"text": "hi"}]}, f)


def test_transcribe_via_subprocess_defaults_to_auto_language_when_not_configured():
    captured = []
    fake_run = _fake_run_capturing_cmd(captured, _write_minimal_output)
    with patch("transcribe.subprocess.run", side_effect=fake_run):
        transcribe._transcribe_via_subprocess("/tmp/some.wav", CONFIG)
    cmd = captured[0]
    assert cmd[cmd.index("-l") + 1] == "auto"


def test_transcribe_via_subprocess_uses_configured_language_override():
    captured = []
    fake_run = _fake_run_capturing_cmd(captured, _write_minimal_output)
    config_with_language = dict(CONFIG, language="tr")
    with patch("transcribe.subprocess.run", side_effect=fake_run):
        transcribe._transcribe_via_subprocess("/tmp/some.wav", config_with_language)
    cmd = captured[0]
    assert cmd[cmd.index("-l") + 1] == "tr"


def test_transcribe_via_subprocess_omits_prompt_flag_when_glossary_empty():
    captured = []
    fake_run = _fake_run_capturing_cmd(captured, _write_minimal_output)
    with patch("transcribe.subprocess.run", side_effect=fake_run):
        transcribe._transcribe_via_subprocess("/tmp/some.wav", CONFIG)
    cmd = captured[0]
    assert "--prompt" not in cmd


def test_transcribe_via_subprocess_passes_glossary_as_prompt_hint():
    captured = []
    fake_run = _fake_run_capturing_cmd(captured, _write_minimal_output)
    config_with_glossary = dict(CONFIG, glossary=["Kubernetes", "PyQt", "Grafana"])
    with patch("transcribe.subprocess.run", side_effect=fake_run):
        transcribe._transcribe_via_subprocess("/tmp/some.wav", config_with_glossary)
    cmd = captured[0]
    prompt_value = cmd[cmd.index("--prompt") + 1]
    assert "Kubernetes" in prompt_value
    assert "PyQt" in prompt_value
    assert "Grafana" in prompt_value


def _fake_post_response(json_body):
    resp = MagicMock()
    resp.json.return_value = json_body
    resp.raise_for_status = lambda: None
    return resp


def test_transcribe_via_server_parses_text_and_normalizes_known_language(tmp_path):
    wav_path = tmp_path / "some.wav"
    wav_path.write_bytes(b"fake wav data")
    with patch(
        "transcribe.requests.post",
        return_value=_fake_post_response({"text": "  hello there  ", "language": "english"}),
    ) as mock_post:
        text, language = transcribe._transcribe_via_server(
            str(wav_path), "http://127.0.0.1:8090", CONFIG
        )
    assert text == "hello there"
    assert language == "en"
    assert mock_post.call_args[0][0] == "http://127.0.0.1:8090/inference"


def test_transcribe_via_server_passes_through_unknown_language_lowercased(tmp_path):
    wav_path = tmp_path / "some.wav"
    wav_path.write_bytes(b"fake wav data")
    with patch(
        "transcribe.requests.post",
        return_value=_fake_post_response({"text": "bonjour", "language": "French"}),
    ):
        _text, language = transcribe._transcribe_via_server(
            str(wav_path), "http://127.0.0.1:8090", CONFIG
        )
    assert language == "french"


def test_transcribe_via_server_omits_prompt_field_when_glossary_empty(tmp_path):
    wav_path = tmp_path / "some.wav"
    wav_path.write_bytes(b"fake wav data")
    with patch(
        "transcribe.requests.post",
        return_value=_fake_post_response({"text": "hi", "language": "english"}),
    ) as mock_post:
        transcribe._transcribe_via_server(str(wav_path), "http://127.0.0.1:8090", CONFIG)
    sent_data = mock_post.call_args.kwargs["data"]
    assert "prompt" not in sent_data


def test_transcribe_via_server_passes_glossary_as_prompt_field(tmp_path):
    wav_path = tmp_path / "some.wav"
    wav_path.write_bytes(b"fake wav data")
    config_with_glossary = dict(CONFIG, glossary=["Kubernetes", "PyQt"])
    with patch(
        "transcribe.requests.post",
        return_value=_fake_post_response({"text": "hi", "language": "english"}),
    ) as mock_post:
        transcribe._transcribe_via_server(
            str(wav_path), "http://127.0.0.1:8090", config_with_glossary
        )
    sent_data = mock_post.call_args.kwargs["data"]
    assert "Kubernetes" in sent_data["prompt"]
    assert "PyQt" in sent_data["prompt"]


def test_transcribe_dispatches_to_server_when_available(tmp_path, monkeypatch):
    wav_path = tmp_path / "some.wav"
    wav_path.write_bytes(b"fake wav data")
    monkeypatch.setattr(whisper_server, "ensure_running", lambda config: "http://127.0.0.1:8090")
    with patch("transcribe.subprocess.run") as mock_run, patch(
        "transcribe.requests.post",
        return_value=_fake_post_response({"text": "server result", "language": "english"}),
    ):
        text, language = transcribe.transcribe(str(wav_path), CONFIG)
    assert text == "server result"
    assert language == "en"
    mock_run.assert_not_called()


def test_transcribe_falls_back_to_subprocess_when_server_unavailable(tmp_path, monkeypatch):
    wav_path = tmp_path / "some.wav"
    wav_path.write_bytes(b"fake wav data")

    def _raise_server_error(config):
        raise whisper_server.WhisperServerError("cooldown")

    monkeypatch.setattr(whisper_server, "ensure_running", _raise_server_error)
    captured = []
    fake_run = _fake_run_capturing_cmd(captured, _write_minimal_output)
    with patch("transcribe.subprocess.run", side_effect=fake_run):
        text, language = transcribe.transcribe(str(wav_path), CONFIG)
    assert text == "hi"
    assert language == "en"
    assert captured  # subprocess path was actually exercised


def test_transcribe_falls_back_to_subprocess_when_server_request_fails(tmp_path, monkeypatch):
    wav_path = tmp_path / "some.wav"
    wav_path.write_bytes(b"fake wav data")
    monkeypatch.setattr(whisper_server, "ensure_running", lambda config: "http://127.0.0.1:8090")
    captured = []
    fake_run = _fake_run_capturing_cmd(captured, _write_minimal_output)
    with patch(
        "transcribe.requests.post",
        side_effect=requests.exceptions.ConnectionError("reset"),
    ), patch("transcribe.subprocess.run", side_effect=fake_run):
        text, language = transcribe.transcribe(str(wav_path), CONFIG)
    assert text == "hi"
    assert language == "en"
    assert captured


def test_transcribe_falls_back_to_subprocess_when_server_response_is_malformed(
    tmp_path, monkeypatch
):
    wav_path = tmp_path / "some.wav"
    wav_path.write_bytes(b"fake wav data")
    monkeypatch.setattr(whisper_server, "ensure_running", lambda config: "http://127.0.0.1:8090")
    captured = []
    fake_run = _fake_run_capturing_cmd(captured, _write_minimal_output)

    malformed_resp = MagicMock()
    malformed_resp.raise_for_status = lambda: None
    malformed_resp.json.side_effect = ValueError("not valid json")

    with patch(
        "transcribe.requests.post", return_value=malformed_resp
    ), patch("transcribe.subprocess.run", side_effect=fake_run):
        text, language = transcribe.transcribe(str(wav_path), CONFIG)
    assert text == "hi"
    assert language == "en"
    assert captured  # subprocess path was actually exercised
