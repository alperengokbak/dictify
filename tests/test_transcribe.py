import json
import wave
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
        return_value=_fake_post_response(
            {
                "text": "  hello there  ",
                "language": "english",
                "segments": [{"text": "  hello there  "}],
            }
        ),
    ) as mock_post:
        text, language = transcribe._transcribe_via_server(
            str(wav_path), "http://127.0.0.1:8090", CONFIG
        )
    assert text == "hello there"
    assert language == "en"
    assert mock_post.call_args[0][0] == "http://127.0.0.1:8090/inference"


def test_transcribe_via_server_joins_segments_instead_of_using_raw_text_field(tmp_path):
    # Regression test: whisper-server's top-level "text" field embeds
    # literal newlines between (and sometimes mid-word within) segments -
    # e.g. "speech-to-\ntext transcription" - because it's meant for
    # human-readable display, not for feeding straight into a paste buffer.
    # Pasting that raw text splits it across multiple lines/rows instead of
    # one continuous sentence. Must reconstruct from "segments" instead,
    # exactly like _parse_whisper_json already does for the subprocess path.
    wav_path = tmp_path / "some.wav"
    wav_path.write_bytes(b"fake wav data")
    with patch(
        "transcribe.requests.post",
        return_value=_fake_post_response(
            {
                "text": " Hi, this is a test recording to measure how long speech-to-\n"
                "text transcription takes on this machine,\n"
                " so we can figure out where the time is going and reduce it.\n",
                "language": "english",
                "segments": [
                    {"text": " Hi, this is a test recording to measure how long speech-to-"},
                    {"text": "text transcription takes on this machine,"},
                    {"text": " so we can figure out where the time is going and reduce it."},
                ],
            }
        ),
    ):
        text, _language = transcribe._transcribe_via_server(
            str(wav_path), "http://127.0.0.1:8090", CONFIG
        )
    assert "\n" not in text
    assert text == (
        "Hi, this is a test recording to measure how long speech-to- "
        "text transcription takes on this machine, "
        "so we can figure out where the time is going and reduce it."
    )


def test_transcribe_via_server_passes_through_unknown_language_lowercased(tmp_path):
    wav_path = tmp_path / "some.wav"
    wav_path.write_bytes(b"fake wav data")
    with patch(
        "transcribe.requests.post",
        return_value=_fake_post_response(
            {"text": "bonjour", "language": "French", "segments": [{"text": "bonjour"}]}
        ),
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
        return_value=_fake_post_response(
            {"text": "hi", "language": "english", "segments": [{"text": "hi"}]}
        ),
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
        return_value=_fake_post_response(
            {"text": "hi", "language": "english", "segments": [{"text": "hi"}]}
        ),
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
        return_value=_fake_post_response(
            {
                "text": "server result",
                "language": "english",
                "segments": [{"text": "server result"}],
            }
        ),
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


def test_transcribe_falls_back_to_subprocess_on_key_error_from_server_path(
    tmp_path, monkeypatch
):
    # Simulates any KeyError arising while talking to the server (payload
    # shape isn't the only thing that could raise one) - the dispatcher's
    # except tuple must catch it and degrade to the subprocess path rather
    # than letting it propagate.
    wav_path = tmp_path / "some.wav"
    wav_path.write_bytes(b"fake wav data")
    monkeypatch.setattr(whisper_server, "ensure_running", lambda config: "http://127.0.0.1:8090")
    captured = []
    fake_run = _fake_run_capturing_cmd(captured, _write_minimal_output)

    key_error_resp = MagicMock()
    key_error_resp.raise_for_status = lambda: None
    key_error_resp.json.side_effect = KeyError("text")

    with patch(
        "transcribe.requests.post", return_value=key_error_resp
    ), patch("transcribe.subprocess.run", side_effect=fake_run):
        text, language = transcribe.transcribe(str(wav_path), CONFIG)
    assert text == "hi"
    assert language == "en"
    assert captured  # subprocess path was actually exercised


def test_transcribe_falls_back_to_subprocess_when_server_payload_is_not_a_dict(
    tmp_path, monkeypatch
):
    # A response body that's valid JSON but not an object (e.g. a bare
    # list) has no .get() method - payload.get("text", ...) raises
    # AttributeError, which the dispatcher must also catch.
    wav_path = tmp_path / "some.wav"
    wav_path.write_bytes(b"fake wav data")
    monkeypatch.setattr(whisper_server, "ensure_running", lambda config: "http://127.0.0.1:8090")
    captured = []
    fake_run = _fake_run_capturing_cmd(captured, _write_minimal_output)

    with patch(
        "transcribe.requests.post", return_value=_fake_post_response(["unexpected", "list"])
    ), patch("transcribe.subprocess.run", side_effect=fake_run):
        text, language = transcribe.transcribe(str(wav_path), CONFIG)
    assert text == "hi"
    assert language == "en"
    assert captured  # subprocess path was actually exercised


def _write_wav(path, duration_secs, framerate=100):
    """Writes a real, wave-module-readable WAV of a given playing time. The
    deliberately low frame rate keeps the fixture at a few KB while still
    declaring a long duration in its header - the timeout math only ever
    looks at nframes/framerate."""
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(1)
        wav.setframerate(framerate)
        wav.writeframes(b"\x00" * int(duration_secs * framerate))


def test_inference_timeout_uses_the_floor_for_short_dictation_clips(tmp_path):
    wav_path = tmp_path / "short.wav"
    _write_wav(wav_path, duration_secs=5)
    assert transcribe._inference_timeout_secs(str(wav_path)) == (
        transcribe.INFERENCE_TIMEOUT_FLOOR_SECS
    )


def test_inference_timeout_scales_with_long_audio(tmp_path):
    # Regression test: a flat 30s timeout made "Transcribe File..." strictly
    # worse for long audio - it burned the timeout AND then paid the full
    # whisper-cli cost on top.
    wav_path = tmp_path / "long.wav"
    _write_wav(wav_path, duration_secs=600)  # 10 minutes of audio
    timeout = transcribe._inference_timeout_secs(str(wav_path))
    assert timeout > transcribe.INFERENCE_TIMEOUT_FLOOR_SECS
    assert timeout == 600 * transcribe.INFERENCE_TIMEOUT_PER_AUDIO_SEC


def test_inference_timeout_falls_back_to_the_floor_for_an_unreadable_wav(tmp_path):
    wav_path = tmp_path / "not-really.wav"
    wav_path.write_bytes(b"fake wav data")
    assert transcribe._inference_timeout_secs(str(wav_path)) == (
        transcribe.INFERENCE_TIMEOUT_FLOOR_SECS
    )


def test_inference_timeout_falls_back_to_the_floor_for_an_empty_wav(tmp_path):
    # A zero-byte file makes the wave module raise EOFError, not wave.Error.
    wav_path = tmp_path / "empty.wav"
    wav_path.write_bytes(b"")
    assert transcribe._inference_timeout_secs(str(wav_path)) == (
        transcribe.INFERENCE_TIMEOUT_FLOOR_SECS
    )


def test_inference_timeout_falls_back_to_the_floor_for_a_missing_wav(tmp_path):
    assert transcribe._inference_timeout_secs(str(tmp_path / "gone.wav")) == (
        transcribe.INFERENCE_TIMEOUT_FLOOR_SECS
    )


def test_transcribe_via_server_sends_a_duration_scaled_timeout(tmp_path):
    wav_path = tmp_path / "long.wav"
    _write_wav(wav_path, duration_secs=600)
    with patch(
        "transcribe.requests.post",
        return_value=_fake_post_response(
            {"text": "hi", "language": "english", "segments": [{"text": "hi"}]}
        ),
    ) as mock_post:
        transcribe._transcribe_via_server(str(wav_path), "http://127.0.0.1:8090", CONFIG)
    assert mock_post.call_args.kwargs["timeout"] == (
        600 * transcribe.INFERENCE_TIMEOUT_PER_AUDIO_SEC
    )


def test_transcribe_falls_back_to_subprocess_when_the_wav_cannot_be_opened(
    tmp_path, monkeypatch
):
    # open(wav_path, "rb") lives inside _transcribe_via_server but outside
    # requests' own exception hierarchy - a FileNotFoundError/PermissionError
    # there must degrade to the subprocess path, not escape as a bare OSError.
    missing_wav = str(tmp_path / "never-written.wav")
    monkeypatch.setattr(whisper_server, "ensure_running", lambda config: "http://127.0.0.1:8090")
    captured = []
    fake_run = _fake_run_capturing_cmd(captured, _write_minimal_output)
    with patch("transcribe.subprocess.run", side_effect=fake_run):
        text, language = transcribe.transcribe(missing_wav, CONFIG)
    assert text == "hi"
    assert language == "en"
    assert captured  # subprocess path was actually exercised


def test_transcribe_prints_a_diagnostic_before_falling_back(tmp_path, monkeypatch, capsys):
    # A permanently-failing server path is otherwise invisible: the app just
    # silently stays slow forever, with nothing in dictify.err.log to say why.
    wav_path = tmp_path / "some.wav"
    wav_path.write_bytes(b"fake wav data")

    def _raise_server_error(config):
        raise whisper_server.WhisperServerError("cooldown")

    monkeypatch.setattr(whisper_server, "ensure_running", _raise_server_error)
    captured = []
    fake_run = _fake_run_capturing_cmd(captured, _write_minimal_output)
    with patch("transcribe.subprocess.run", side_effect=fake_run):
        transcribe.transcribe(str(wav_path), CONFIG)

    stderr = capsys.readouterr().err
    assert "[dictify diag]" in stderr
    assert "WhisperServerError" in stderr


def test_transcribe_prints_no_diagnostic_when_the_server_path_works(tmp_path, monkeypatch, capsys):
    wav_path = tmp_path / "some.wav"
    wav_path.write_bytes(b"fake wav data")
    monkeypatch.setattr(whisper_server, "ensure_running", lambda config: "http://127.0.0.1:8090")
    with patch(
        "transcribe.requests.post",
        return_value=_fake_post_response(
            {"text": "hi", "language": "english", "segments": [{"text": "hi"}]}
        ),
    ):
        transcribe.transcribe(str(wav_path), CONFIG)
    assert "[dictify diag]" not in capsys.readouterr().err
