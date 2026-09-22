import json
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import requests

import whisper_server

_LANGUAGE_NAME_TO_CODE = {"english": "en", "turkish": "tr"}

# Whisper annotates non-speech audio with bracketed tags - "[BLANK_AUDIO]",
# "[inaudible]", "(banging)" - and emits them as their own segments. They are
# not speech and must never reach the paste buffer: observed live via
# history.jsonl, a silent recording came through as a literal "[BLANK_AUDIO]"
# pasted into whatever the user was typing in.
#
# Deliberately only matches a segment that is ENTIRELY one bracketed run.
# Stripping bracketed text anywhere would eat real speech - "I installed
# English (UK) and English (US)" is genuine content, not an annotation.
# Known limitation: an annotation Whisper packs into the same segment as
# real speech survives this. Every case seen so far has been its own
# segment, so this stays conservative rather than guessing at the rest.
_ARTIFACT_SEGMENT_RE = re.compile(r"^[\[(][^\[\]()]*[\])]$")


def _join_segment_text(segments) -> str:
    """Joins Whisper's segments into one line, dropping non-speech
    annotation segments. Returns "" if there was no actual speech, which
    dictate.py already treats as "nothing to paste".

    Whisper starts a segment that begins a new word with a space; one
    without it continues the previous word (Whisper can split mid-word).
    Space-joining every segment turned "dinleyelim" into "dinley elim"."""
    joined = ""
    for seg in segments:
        raw = seg["text"]
        stripped = raw.strip()
        if not stripped or _ARTIFACT_SEGMENT_RE.match(stripped):
            continue
        if joined and raw[:1].isspace():
            joined += " "
        joined += stripped
    return joined

# Server-side inference time scales with audio length, so a flat timeout would
# make "Transcribe File..." strictly worse for long files: it would burn the
# whole timeout and THEN pay the full whisper-cli cost on top.
INFERENCE_TIMEOUT_FLOOR_SECS = 30
INFERENCE_TIMEOUT_PER_AUDIO_SEC = 2

GLOSSARY_PROMPT_MIN_SECS = 2.0

# "auto" is narrowed to the two languages Dictify is for. A detection outside
# them, or below this confidence, is re-run - see _resolve_auto_language.
# Only the server path does this: whisper-cli's JSON carries no probabilities.
SUPPORTED_AUTO_LANGUAGES = ("en", "tr")
AUTO_LANGUAGE_MIN_PROBABILITY = 0.8
_last_confident_language: str | None = None


class TranscribeError(Exception):
    pass


def _parse_whisper_json(json_path: str) -> tuple[str, str]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("transcription", [])
    text = _join_segment_text(segments)
    language = data.get("result", {}).get("language", "unknown")
    return text, language


def transcribe(wav_path: str, config: dict) -> tuple[str, str]:
    try:
        base_url = whisper_server.ensure_running(config)
        return _transcribe_via_server(wav_path, base_url, config)
    except (
        whisper_server.WhisperServerError,
        requests.RequestException,
        OSError,
        ValueError,
        KeyError,
        AttributeError,
    ) as exc:
        # Falls back on ANY server-path failure, not just connection-level
        # ones - a malformed/unexpected JSON response (ValueError/KeyError/
        # AttributeError) or an unreadable wav (OSError) should degrade to
        # the subprocess path exactly like a connection failure would, same
        # defensive stance cleanup.py already takes for its own Ollama
        # response parsing. Logged because a permanently-failing server path
        # is otherwise invisible - the app just silently stays slow.
        print(
            f"[dictify diag] whisper-server path failed ({exc!r}); falling back to subprocess",
            file=sys.stderr,
        )
        return _transcribe_via_subprocess(wav_path, config)


def _wav_duration_secs(wav_path: str) -> float | None:
    """Playing time of the wav, or None if it can't be read."""
    try:
        with wave.open(wav_path, "rb") as wav:
            frame_rate = wav.getframerate()
            return wav.getnframes() / frame_rate if frame_rate else 0
    except (OSError, wave.Error, EOFError):
        return None


def _inference_timeout_secs(wav_path: str) -> float:
    """Scales the server request timeout with the audio's own length - live
    dictation clips keep the 30s floor, a 20-minute file gets 40 minutes."""
    duration = _wav_duration_secs(wav_path)
    if duration is None:
        # Unreadable/truncated/not-a-wav: the timeout is a safety net, not a
        # reason to fail the transcription - fall back to the floor.
        return INFERENCE_TIMEOUT_FLOOR_SECS
    return max(INFERENCE_TIMEOUT_FLOOR_SECS, duration * INFERENCE_TIMEOUT_PER_AUDIO_SEC)


def _glossary_prompt(wav_path: str, config: dict) -> str | None:
    """The glossary as Whisper's prompt hint, or None. Skipped for clips
    shorter than GLOSSARY_PROMPT_MIN_SECS: with too little audio to anchor
    on, Whisper echoes the prompt instead of transcribing - a 1.7 s
    "Tamam, teşekkürler." came back as "Code, Xcode, SwiftUI, Alperen
    Gökbak, DeepL, Arc". An unreadable wav keeps the prompt (old behavior)."""
    glossary = config.get("glossary") or []
    if not glossary:
        return None
    duration = _wav_duration_secs(wav_path)
    if duration is not None and duration < GLOSSARY_PROMPT_MIN_SECS:
        return None
    return ", ".join(glossary)


def _is_glossary_echo(text: str, glossary) -> bool:
    """True when the transcript is nothing but glossary words - the prompt
    leaking through rather than speech. Needs at least two words, so a
    deliberately dictated single term ("Xcode.") is kept."""
    glossary_words = {w.lower() for term in (glossary or []) for w in term.split()}
    words = [w.lower() for w in re.findall(r"[^\W_]+", text)]
    return len(words) >= 2 and all(w in glossary_words for w in words)


def _resolve_auto_language(payload: dict, detected: str) -> str | None:
    """For language "auto": returns None when the detection can be trusted,
    otherwise the language (en/tr) to re-run the clip in.

    Short clips carry too little audio for Whisper's detector - live they
    came back as Czech, and a 0.45 s "Evet." is "English" at p=0.66. Forcing
    both languages and comparing confidence does NOT work: forced English
    on Turkish speech is a confident *translation* ("Tamam, teşekkürler."
    -> "Thank you.", avg_logprob -0.004). So an unsure clip goes to the
    language last spoken with confidence (people dictate in runs of one
    language), else the likelier of en/tr by the detector's own odds."""
    global _last_confident_language
    probability = payload.get("detected_language_probability")
    if detected in SUPPORTED_AUTO_LANGUAGES and (
        probability is None or probability >= AUTO_LANGUAGE_MIN_PROBABILITY
    ):
        _last_confident_language = detected
        return None
    if _last_confident_language:
        return _last_confident_language
    odds = payload.get("language_probabilities") or {}
    return max(SUPPORTED_AUTO_LANGUAGES, key=lambda code: odds.get(code, 0))


def _post_inference(wav_path: str, base_url: str, language: str, prompt: str | None) -> dict:
    data = {"response_format": "verbose_json", "language": language}
    if prompt:
        data["prompt"] = prompt
    with open(wav_path, "rb") as f:
        resp = requests.post(
            f"{base_url}/inference",
            files={"file": f},
            data=data,
            timeout=_inference_timeout_secs(wav_path),
        )
    resp.raise_for_status()
    return resp.json()


def _transcribe_via_server(wav_path: str, base_url: str, config: dict) -> tuple[str, str]:
    requested = config.get("language", "auto")
    prompt = _glossary_prompt(wav_path, config)
    payload = _post_inference(wav_path, base_url, requested, prompt)
    # whisper-server's top-level "text" field is meant for human-readable
    # display and embeds literal newlines between (and sometimes mid-word
    # within) segments - e.g. "speech-to-\ntext transcription". Pasting that
    # verbatim splits the transcript across multiple lines/rows instead of
    # one continuous sentence. Reconstruct from "segments" instead, exactly
    # like _parse_whisper_json already does for the subprocess path.
    text = _join_segment_text(payload.get("segments", []))
    if prompt and _is_glossary_echo(text, config.get("glossary")):
        prompt = None
        payload = _post_inference(wav_path, base_url, requested, prompt)
        text = _join_segment_text(payload.get("segments", []))

    language = payload.get("language", "unknown").lower()
    language = _LANGUAGE_NAME_TO_CODE.get(language, language)
    if requested == "auto":
        rerun_language = _resolve_auto_language(payload, language)
        if rerun_language:
            payload = _post_inference(wav_path, base_url, rerun_language, prompt)
            text = _join_segment_text(payload.get("segments", []))
            language = rerun_language
    return text, language


def _transcribe_via_subprocess(wav_path: str, config: dict) -> tuple[str, str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = str(Path(tmpdir) / "out")
        cmd = [
            config["whisper_binary"],
            "-m", config["whisper_model_path"],
            "-f", wav_path,
            "-l", config.get("language", "auto"),
            "-np",
            "-oj",
            "-of", out_prefix,
        ]
        prompt = _glossary_prompt(wav_path, config)
        if prompt:
            cmd += ["--prompt", prompt]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise TranscribeError(f"whisper-cli binary not found: {exc}") from exc

        if result.returncode != 0:
            raise TranscribeError(f"whisper-cli failed: {result.stderr.strip()}")

        json_path = out_prefix + ".json"
        if not Path(json_path).exists():
            raise TranscribeError("whisper-cli did not produce output JSON")

        try:
            return _parse_whisper_json(json_path)
        except (KeyError, json.JSONDecodeError, OSError) as exc:
            raise TranscribeError(f"failed to parse whisper-cli output: {exc}") from exc
