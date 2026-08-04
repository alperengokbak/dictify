import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "dictify"
CONFIG_PATH = CONFIG_DIR / "config.json"

DEFAULT_CONFIG = {
    "hotkey": "<ctrl>+<alt>",
    "whisper_binary": "/opt/homebrew/bin/whisper-cli",
    "whisper_model_path": str(CONFIG_DIR / "models" / "ggml-medium.bin"),
    "ffmpeg_binary": "/opt/homebrew/bin/ffmpeg",
    "ollama_url": "http://localhost:11434",
    "ollama_model": "qwen2.5:3b",
    "cleanup_enabled": True,
    "silence_peak_floor_dbfs": -55.0,
    "silence_rise_db": 10.0,
    "language": "auto",
    "glossary": [],
    "style": "default",
    "recording_mode": "toggle",
    "history_enabled": True,
    "history_limit": 200,
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    try:
        with open(CONFIG_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_CONFIG)
    merged = dict(DEFAULT_CONFIG)
    merged.update(data)
    return merged


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
