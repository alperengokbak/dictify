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
    # The second segment has no leading space: Whisper split mid-word, so
    # it is glued back on ("speech-to-text"), not space-joined.
    assert text == (
        "Hi, this is a test recording to measure how long speech-to-"
        "text transcription takes on this machine, "
        "so we can figure out where the time is going and reduce it."
    )


def test_transcribe_via_server_passes_through_unknown_language_lowercased(tmp_path):
    # An explicitly chosen language is trusted as-is; only "auto" detections
    # are narrowed to English/Turkish (see the language-resolution tests).
    wav_path = tmp_path / "some.wav"
    wav_path.write_bytes(b"fake wav data")
    with patch(
        "transcribe.requests.post",
        return_value=_fake_post_response(
            {"text": "bonjour", "language": "French", "segments": [{"text": "bonjour"}]}
        ),
    ):
        _text, language = transcribe._transcribe_via_server(
            str(wav_path), "http://127.0.0.1:8090", dict(CONFIG, language="fr")
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


def test_parse_whisper_json_drops_non_speech_artifact_segments():
    # Regression: observed live (2026-08-12) via history.jsonl - Whisper
    # emits non-speech annotations as their own segments and they were
    # being joined into the transcript like real speech, then pasted into
    # whatever the user was typing in. Four real dictations were affected:
    # two were nothing but "[BLANK_AUDIO]"/"[inaudible]", one had
    # "(banging)" appended after a full paragraph, one had a trailing
    # "[BLANK_AUDIO]" that only got removed because the cleanup model
    # happened to drop it - which is not a guarantee (see the fallback
    # test below).
    text, lang = transcribe._parse_whisper_json(
        str(FIXTURES / "whisper_output_artifacts.json")
    )
    assert text == "Under the old prompt this was the shape that got collapsed."
    assert lang == "en"


def test_transcribe_via_server_drops_non_speech_artifact_segments(tmp_path):
    # Same artifact stripping must apply to the server path, not just the
    # subprocess path - the server path is the one live dictation actually
    # uses, and it was where the observed "[BLANK_AUDIO]" cases came from.
    wav_path = tmp_path / "some.wav"
    wav_path.write_bytes(b"fake wav data")
    with patch(
        "transcribe.requests.post",
        return_value=_fake_post_response(
            {
                "text": "ignored",
                "language": "english",
                "segments": [
                    {"text": " Can we also see these differences?"},
                    {"text": " (banging)"},
                ],
            }
        ),
    ):
        text, _language = transcribe._transcribe_via_server(
            str(wav_path), "http://127.0.0.1:8090", CONFIG
        )
    assert text == "Can we also see these differences?"


def test_transcribe_returns_empty_when_every_segment_is_an_artifact(tmp_path):
    # Two of the four real cases were a recording containing no speech at
    # all, transcribed as nothing but "[BLANK_AUDIO]" / "[inaudible]" and
    # then pasted verbatim. Stripping must leave an empty string here, not
    # a stray bracket or whitespace - dictate.py's `if not raw_text: return`
    # then makes the whole dictation a clean no-op instead of a bad paste.
    wav_path = tmp_path / "some.wav"
    wav_path.write_bytes(b"fake wav data")
    with patch(
        "transcribe.requests.post",
        return_value=_fake_post_response(
            {
                "text": "ignored",
                "language": "english",
                "segments": [{"text": " [BLANK_AUDIO]"}, {"text": " [inaudible]"}],
            }
        ),
    ):
        text, _language = transcribe._transcribe_via_server(
            str(wav_path), "http://127.0.0.1:8090", CONFIG
        )
    assert text == ""


def test_transcribe_keeps_parentheses_that_are_part_of_real_speech(tmp_path):
    # The stripping rule must only fire on a segment that is ENTIRELY one
    # bracketed annotation. A parenthetical inside a real spoken sentence
    # is genuine content and must survive untouched - dropping every
    # bracketed run would silently eat the user's own words.
    wav_path = tmp_path / "some.wav"
    wav_path.write_bytes(b"fake wav data")
    with patch(
        "transcribe.requests.post",
        return_value=_fake_post_response(
            {
                "text": "ignored",
                "language": "english",
                "segments": [
                    {"text": " I installed English (UK) and English (US)."},
                ],
            }
        ),
    ):
        text, _language = transcribe._transcribe_via_server(
            str(wav_path), "http://127.0.0.1:8090", CONFIG
        )
    assert text == "I installed English (UK) and English (US)."


# --- Auto-detect narrowed to English/Turkish (2026-09-22) -------------------
# Observed live in history.jsonl on 2026-09-21: short clips under
# language "auto" came back as Czech ("Sáv, takže krede.") and English
# speech was tagged Turkish. Reproduced with `say`: a 0.45 s "Evet." is
# detected as English at p=0.66 and transcribed as "Amidst.".


@pytest.fixture(autouse=True)
def _reset_remembered_language():
    transcribe._last_confident_language = None
    yield
    transcribe._last_confident_language = None


def _server_payload(text, language, probability=None, probabilities=None):
    payload = {"text": text, "language": language, "segments": [{"text": text}]}
    if probability is not None:
        payload["detected_language_probability"] = probability
    if probabilities is not None:
        payload["language_probabilities"] = probabilities
    return payload


def _run_server(tmp_path, responses, config=CONFIG, duration_secs=5):
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path, duration_secs=duration_secs)
    with patch(
        "transcribe.requests.post",
        side_effect=[_fake_post_response(r) for r in responses],
    ) as mock_post:
        result = transcribe._transcribe_via_server(
            str(wav_path), "http://127.0.0.1:8090", config
        )
    return result, [c.kwargs["data"] for c in mock_post.call_args_list]


def test_confident_turkish_detection_is_accepted_without_a_rerun(tmp_path):
    (text, language), sent = _run_server(
        tmp_path, [_server_payload("Bunu bir daha dinleyelim.", "turkish", 0.99)]
    )
    assert (text, language) == ("Bunu bir daha dinleyelim.", "tr")
    assert len(sent) == 1


def test_detection_outside_english_and_turkish_is_rerun_in_the_likelier_of_the_two(tmp_path):
    (text, language), sent = _run_server(
        tmp_path,
        [
            _server_payload("Sáv, takže krede.", "czech", 0.4,
                            {"cs": 0.4, "tr": 0.3, "en": 0.1}),
            _server_payload("Tamam, teşekkürler.", "turkish"),
        ],
    )
    assert (text, language) == ("Tamam, teşekkürler.", "tr")
    assert sent[1]["language"] == "tr"


def test_low_confidence_detection_falls_back_to_the_last_confidently_spoken_language(tmp_path):
    _run_server(tmp_path, [_server_payload("Yarın okula gideceğim.", "turkish", 0.99)])
    (text, language), sent = _run_server(
        tmp_path,
        [
            _server_payload("Amidst.", "english", 0.66, {"en": 0.66, "tr": 0.01}),
            _server_payload("Evet.", "turkish"),
        ],
    )
    assert (text, language) == ("Evet.", "tr")
    assert sent[1]["language"] == "tr"


def test_low_confidence_with_no_history_uses_the_likelier_of_english_and_turkish(tmp_path):
    (_text, language), sent = _run_server(
        tmp_path,
        [
            _server_payload("Okay.", "english", 0.7, {"en": 0.7, "tr": 0.05}),
            _server_payload("Okay.", "english"),
        ],
    )
    assert language == "en"
    assert sent[1]["language"] == "en"


def test_an_explicitly_chosen_language_is_never_rerun(tmp_path):
    (_text, language), sent = _run_server(
        tmp_path,
        [_server_payload("Evet.", "turkish", 0.3)],
        config=dict(CONFIG, language="tr"),
    )
    assert language == "tr"
    assert len(sent) == 1


# --- Glossary prompt leaking into short clips (2026-09-22) -----------------
# Reproduced: a 1.7 s "Tamam, teşekkürler." with the glossary as the prompt
# came back as "Code, Xcode, SwiftUI, Alperen Gökbak, DeepL, Arc".

GLOSSARY_CONFIG = dict(CONFIG, glossary=["Claude Code", "Xcode", "SwiftUI", "Alperen Gökbak"])


def test_glossary_prompt_is_skipped_for_clips_under_two_seconds(tmp_path):
    _result, sent = _run_server(
        tmp_path, [_server_payload("Tamam.", "turkish", 0.95)],
        config=GLOSSARY_CONFIG, duration_secs=1.7,
    )
    assert "prompt" not in sent[0]


def test_glossary_prompt_is_kept_for_clips_of_two_seconds_or_more(tmp_path):
    _result, sent = _run_server(
        tmp_path, [_server_payload("Open Xcode.", "english", 0.95)],
        config=GLOSSARY_CONFIG, duration_secs=2.5,
    )
    assert "Xcode" in sent[0]["prompt"]


def test_subprocess_path_skips_the_glossary_prompt_for_short_clips(tmp_path):
    wav_path = tmp_path / "short.wav"
    _write_wav(wav_path, duration_secs=1.0)
    captured = []
    fake_run = _fake_run_capturing_cmd(captured, _write_minimal_output)
    with patch("transcribe.subprocess.run", side_effect=fake_run):
        transcribe._transcribe_via_subprocess(str(wav_path), GLOSSARY_CONFIG)
    assert "--prompt" not in captured[0]


def test_output_that_only_echoes_the_glossary_is_rerun_without_the_prompt(tmp_path):
    (text, _language), sent = _run_server(
        tmp_path,
        [
            _server_payload("Code, Xcode, SwiftUI, Alperen Gökbak", "english", 0.95),
            _server_payload("Please open the project.", "english", 0.95),
        ],
        config=GLOSSARY_CONFIG,
    )
    assert text == "Please open the project."
    assert "prompt" in sent[0]
    assert "prompt" not in sent[1]


def test_a_single_dictated_glossary_word_is_kept(tmp_path):
    (text, _language), sent = _run_server(
        tmp_path, [_server_payload("Xcode.", "english", 0.95)],
        config=GLOSSARY_CONFIG,
    )
    assert text == "Xcode."
    assert len(sent) == 1


# --- Mid-word segment splits (2026-09-22) ----------------------------------
# Whisper marks a segment that starts a new word with a leading space; a
# segment WITHOUT one continues the previous word. Space-joining every
# segment produced "dinley elim" live (history.jsonl, 2026-09-21) and
# "uğ rayacağım" when reproduced.


def test_a_segment_without_a_leading_space_continues_the_previous_word(tmp_path):
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path, duration_secs=5)
    payload = {
        "language": "turkish",
        "detected_language_probability": 0.99,
        "segments": [
            {"text": " Yarın sabah okula gideceğim ve dersten sonra kütüphaneye uğ"},
            {"text": "rayacağım."},
            {"text": " Sonra eve döneceğim."},
        ],
    }
    with patch("transcribe.requests.post", return_value=_fake_post_response(payload)):
        text, _language = transcribe._transcribe_via_server(
            str(wav_path), "http://127.0.0.1:8090", CONFIG
        )
    assert text == (
        "Yarın sabah okula gideceğim ve dersten sonra kütüphaneye uğrayacağım. "
        "Sonra eve döneceğim."
    )
