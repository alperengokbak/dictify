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
    dictate.py already treats as "nothing to paste"."""
    return " ".join(
        stripped
        for stripped in (seg["text"].strip() for seg in segments)
        if stripped and not _ARTIFACT_SEGMENT_RE.match(stripped)
    ).strip()

# Server-side inference time scales with audio length, so a flat timeout would
# make "Transcribe File..." strictly worse for long files: it would burn the
# whole timeout and THEN pay the full whisper-cli cost on top.
INFERENCE_TIMEOUT_FLOOR_SECS = 30
INFERENCE_TIMEOUT_PER_AUDIO_SEC = 2


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


def _inference_timeout_secs(wav_path: str) -> float:
    """Scales the server request timeout with the audio's own length - live
    dictation clips keep the 30s floor, a 20-minute file gets 40 minutes."""
    try:
        with wave.open(wav_path, "rb") as wav:
            frame_rate = wav.getframerate()
            duration = wav.getnframes() / frame_rate if frame_rate else 0
    except (OSError, wave.Error, EOFError):
        # Unreadable/truncated/not-a-wav: the timeout is a safety net, not a
        # reason to fail the transcription - fall back to the floor.
        return INFERENCE_TIMEOUT_FLOOR_SECS
    return max(INFERENCE_TIMEOUT_FLOOR_SECS, duration * INFERENCE_TIMEOUT_PER_AUDIO_SEC)


def _transcribe_via_server(wav_path: str, base_url: str, config: dict) -> tuple[str, str]:
    data = {
        "response_format": "verbose_json",
        "language": config.get("language", "auto"),
    }
    glossary = config.get("glossary") or []
    if glossary:
        data["prompt"] = ", ".join(glossary)

    with open(wav_path, "rb") as f:
        resp = requests.post(
            f"{base_url}/inference",
            files={"file": f},
            data=data,
            timeout=_inference_timeout_secs(wav_path),
        )
    resp.raise_for_status()
    payload = resp.json()
    # whisper-server's top-level "text" field is meant for human-readable
    # display and embeds literal newlines between (and sometimes mid-word
    # within) segments - e.g. "speech-to-\ntext transcription". Pasting that
    # verbatim splits the transcript across multiple lines/rows instead of
    # one continuous sentence. Reconstruct from "segments" instead, exactly
    # like _parse_whisper_json already does for the subprocess path.
    segments = payload.get("segments", [])
    text = _join_segment_text(segments)
    language = payload.get("language", "unknown").lower()
    language = _LANGUAGE_NAME_TO_CODE.get(language, language)
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
        glossary = config.get("glossary") or []
        if glossary:
            cmd += ["--prompt", ", ".join(glossary)]
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
