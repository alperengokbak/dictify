import json
from datetime import datetime, timezone

from config import CONFIG_DIR

HISTORY_PATH = CONFIG_DIR / "history.jsonl"


def append_entry(
    raw_text: str, final_text: str, language: str, style: str, limit: int = 200
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "language": language,
        "style": style,
        "raw_text": raw_text,
        "final_text": final_text,
    }
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    _trim_to_limit(limit)


def _trim_to_limit(limit: int) -> None:
    entries = load_history()
    if len(entries) > limit:
        trimmed = entries[-limit:]
        with open(HISTORY_PATH, "w") as f:
            for entry in trimmed:
                f.write(json.dumps(entry) + "\n")


def load_history() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    entries = []
    with open(HISTORY_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def clear_history() -> None:
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()
