import threading

import numpy as np

import audio
from audio import is_silent, peak_dbfs

SR = 16000


class _FakeInputStream:
    """Stands in for sounddevice.InputStream: fires the callback with one
    synthetic chunk as soon as the context manager is entered."""

    def __init__(self, **kwargs):
        self._callback = kwargs["callback"]

    def __enter__(self):
        fake_indata = np.full((10, 1), 0.5, dtype=np.float32)
        self._callback(fake_indata, 10, None, None)
        return self

    def __exit__(self, *args):
        return False


def test_pure_silence_is_silent():
    samples = np.zeros(SR * 2, dtype=np.float32)
    assert is_silent(samples, SR) is True


def test_loud_burst_with_quiet_background_is_not_silent():
    samples = np.full(SR * 2, 0.01, dtype=np.float32)
    samples[SR:SR + int(SR * 0.1)] = 0.5
    assert is_silent(samples, SR) is False


def test_signal_below_peak_floor_is_silent_regardless_of_rise():
    samples = np.full(SR * 2, 0.001, dtype=np.float32)  # ~ -60 dBFS
    assert is_silent(samples, SR) is True


def test_burst_barely_above_noise_floor_is_silent():
    # background ~ -40 dBFS, burst ~ -34 dBFS: only a 6 dB rise, below
    # the 10 dB threshold, so this should still be dropped as noise.
    samples = np.full(SR * 2, 0.01, dtype=np.float32)
    samples[SR:SR + int(SR * 0.1)] = 0.02
    assert is_silent(samples, SR) is True


def test_record_forwards_chunk_level_via_on_chunk(monkeypatch):
    monkeypatch.setattr(audio.sd, "InputStream", _FakeInputStream)
    captured_levels = []
    stop_event = threading.Event()
    stop_event.set()  # already set, so record() returns right after the fake chunk fires

    audio.record(stop_event, sample_rate=SR, on_chunk=captured_levels.append)

    assert len(captured_levels) == 1
    expected_level = peak_dbfs(np.full(10, 0.5, dtype=np.float32))
    assert captured_levels[0] == expected_level


def test_record_without_on_chunk_does_not_raise(monkeypatch):
    monkeypatch.setattr(audio.sd, "InputStream", _FakeInputStream)
    stop_event = threading.Event()
    stop_event.set()

    samples = audio.record(stop_event, sample_rate=SR)

    assert samples.size == 10
