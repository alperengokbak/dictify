import json
import subprocess
import tempfile
from pathlib import Path

import requests

import whisper_server

_LANGUAGE_NAME_TO_CODE = {"english": "en", "turkish": "tr"}


class TranscribeError(Exception):
    pass


def _parse_whisper_json(json_path: str) -> tuple[str, str]:
    with open(json_path) as f:
        data = json.load(f)
    segments = data.get("transcription", [])
    text = " ".join(seg["text"].strip() for seg in segments).strip()
    language = data.get("result", {}).get("language", "unknown")
    return text, language


def transcribe(wav_path: str, config: dict) -> tuple[str, str]:
    try:
        base_url = whisper_server.ensure_running(config)
        return _transcribe_via_server(wav_path, base_url, config)
    except (
        whisper_server.WhisperServerError,
        requests.RequestException,
        ValueError,
        KeyError,
        AttributeError,
    ):
        # Falls back on ANY server-path failure, not just connection-level
        # ones - a malformed/unexpected JSON response (ValueError/KeyError/
        # AttributeError) should degrade to the subprocess path exactly
        # like a connection failure would, same defensive stance cleanup.py
        # already takes for its own Ollama response parsing.
        return _transcribe_via_subprocess(wav_path, config)


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
            timeout=30,
        )
    resp.raise_for_status()
    payload = resp.json()
    text = payload.get("text", "").strip()
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
