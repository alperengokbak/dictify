from pathlib import Path
from typing import Callable, Optional

import requests

MODEL_SIZE_ORDER = ["tiny", "base", "small", "medium", "large"]

MODEL_LABELS = {
    "tiny": "Tiny (75 MB)",
    "base": "Base (142 MB)",
    "small": "Small (466 MB)",
    "medium": "Medium (1.5 GB)",
    "large": "Large (2.9 GB)",
}

_MODEL_URL_TEMPLATE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{size}.bin"


def model_path_for_size(size: str, models_dir: Path) -> Path:
    return models_dir / f"ggml-{size}.bin"


def size_from_path(path: Path) -> Optional[str]:
    """Reverse lookup for pre-selecting the Preferences dropdown from the
    currently configured whisper_model_path. Returns None for anything
    that doesn't match a known size (e.g. a hand-configured custom path) -
    callers must handle that case rather than assume a match."""
    stem = path.stem
    if not stem.startswith("ggml-"):
        return None
    candidate = stem[len("ggml-"):]
    return candidate if candidate in MODEL_LABELS else None


def is_downloaded(size: str, models_dir: Path) -> bool:
    return model_path_for_size(size, models_dir).exists()


class ModelDownloadError(Exception):
    pass


def download_model(
    size: str,
    models_dir: Path,
    on_progress: Callable[[int, int], None],
    _get=None,
) -> Path:
    """Downloads ggml-<size>.bin into models_dir, atomically. Returns the
    final path on success. On any failure, the partial download is deleted
    and ModelDownloadError is raised - models_dir is left exactly as it was
    before the call started."""
    if size not in MODEL_LABELS:
        raise ValueError(f"unknown model size: {size!r}")

    get = _get or requests.get
    dest = model_path_for_size(size, models_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_suffix(dest.suffix + ".part")

    try:
        response = get(_MODEL_URL_TEMPLATE.format(size=size), stream=True, timeout=30)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(part_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                on_progress(downloaded, total)
        part_path.rename(dest)
        return dest
    except Exception as exc:
        part_path.unlink(missing_ok=True)
        raise ModelDownloadError(f"failed to download {size} model: {exc}") from exc
