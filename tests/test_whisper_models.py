from pathlib import Path

import pytest

import whisper_models


def test_model_path_for_size(tmp_path):
    assert whisper_models.model_path_for_size("medium", tmp_path) == tmp_path / "ggml-medium.bin"


def test_size_from_path_known_size():
    assert whisper_models.size_from_path(Path("/x/ggml-small.bin")) == "small"


def test_size_from_path_unknown_size_returns_none():
    assert whisper_models.size_from_path(Path("/x/ggml-huge.bin")) is None


def test_size_from_path_non_ggml_name_returns_none():
    assert whisper_models.size_from_path(Path("/x/my-custom-model.bin")) is None


def test_is_downloaded_false_when_missing(tmp_path):
    assert whisper_models.is_downloaded("medium", tmp_path) is False


def test_is_downloaded_true_when_present(tmp_path):
    (tmp_path / "ggml-medium.bin").write_bytes(b"fake model data")
    assert whisper_models.is_downloaded("medium", tmp_path) is True


class _FakeResponse:
    def __init__(self, chunks, status=200, headers=None):
        self._chunks = chunks
        self.status = status
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status != 200:
            raise whisper_models.requests.HTTPError(f"status {self.status}")

    def iter_content(self, chunk_size):
        yield from self._chunks


def test_download_model_success_writes_file_and_reports_progress(tmp_path):
    chunks = [b"a" * 10, b"b" * 10, b"c" * 5]
    total = sum(len(c) for c in chunks)

    def fake_get(url, stream, timeout):
        return _FakeResponse(chunks, headers={"content-length": str(total)})

    progress_calls = []
    result = whisper_models.download_model(
        "tiny", tmp_path, lambda d, t: progress_calls.append((d, t)), _get=fake_get
    )

    assert result == tmp_path / "ggml-tiny.bin"
    assert result.read_bytes() == b"a" * 10 + b"b" * 10 + b"c" * 5
    assert not (tmp_path / "ggml-tiny.bin.part").exists()
    assert progress_calls == [(10, total), (20, total), (25, total)]


def test_download_model_http_error_cleans_up_and_raises(tmp_path):
    def fake_get(url, stream, timeout):
        return _FakeResponse([b"partial"], status=500)

    with pytest.raises(whisper_models.ModelDownloadError):
        whisper_models.download_model("tiny", tmp_path, lambda d, t: None, _get=fake_get)

    assert not (tmp_path / "ggml-tiny.bin").exists()
    assert not (tmp_path / "ggml-tiny.bin.part").exists()


def test_download_model_exception_mid_stream_cleans_up_and_raises(tmp_path):
    def broken_chunks():
        yield b"some data"
        raise ConnectionError("connection reset")

    class _BrokenResponse(_FakeResponse):
        def iter_content(self, chunk_size):
            yield from broken_chunks()

    def fake_get(url, stream, timeout):
        return _BrokenResponse([], headers={"content-length": "100"})

    with pytest.raises(whisper_models.ModelDownloadError):
        whisper_models.download_model("tiny", tmp_path, lambda d, t: None, _get=fake_get)

    assert not (tmp_path / "ggml-tiny.bin").exists()
    assert not (tmp_path / "ggml-tiny.bin.part").exists()


def test_download_model_unknown_size_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        whisper_models.download_model("huge", tmp_path, lambda d, t: None)
