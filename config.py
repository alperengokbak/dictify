import json
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "dictify"
CONFIG_PATH = CONFIG_DIR / "config.json"


def _detect_homebrew_bin_dir() -> str:
    """Homebrew installs to /opt/homebrew on Apple Silicon and /usr/local on
    Intel Macs. Detect which one is actually present on this machine rather
    than hardcoding the Apple Silicon path, so whisper-cli/ffmpeg resolve
    correctly out of the box on both architectures. Falls back to the
    Apple Silicon path if neither is found (e.g. Homebrew isn't installed
    yet) - install.sh will have put a real brew prefix in place by the
    time this default actually gets used."""
    for candidate in ("/opt/homebrew/bin", "/usr/local/bin"):
        if Path(candidate).is_dir():
            return candidate
    return "/opt/homebrew/bin"


_HOMEBREW_BIN = _detect_homebrew_bin_dir()

DEFAULT_CONFIG = {
    "hotkey": "<ctrl>+<alt>+<cmd>+<d>",
    "whisper_binary": f"{_HOMEBREW_BIN}/whisper-cli",
    "whisper_model_path": str(CONFIG_DIR / "models" / "ggml-medium.bin"),
    "ffmpeg_binary": f"{_HOMEBREW_BIN}/ffmpeg",
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
    "sound_feedback_enabled": True,
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
