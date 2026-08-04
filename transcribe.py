import json
import subprocess
import tempfile
from pathlib import Path


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
