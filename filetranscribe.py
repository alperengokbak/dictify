import subprocess
import tempfile
from pathlib import Path

from transcribe import TranscribeError, transcribe


class FileTranscribeError(Exception):
    pass


def transcribe_file(input_path: str, config: dict) -> tuple[str, str]:
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = str(Path(tmpdir) / "converted.wav")
        cmd = [
            config.get("ffmpeg_binary", "ffmpeg"),
            "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            wav_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise FileTranscribeError(f"ffmpeg not found: {exc}") from exc

        if result.returncode != 0:
            raise FileTranscribeError(f"ffmpeg conversion failed: {result.stderr.strip()}")

        try:
            return transcribe(wav_path, config)
        except TranscribeError as exc:
            raise FileTranscribeError(f"transcription failed: {exc}") from exc
