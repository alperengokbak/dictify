import numpy as np
from audio import is_silent

SR = 16000


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
