import waveform


def test_normalize_level_at_or_below_floor_is_zero():
    assert waveform._normalize_level(-60.0) == 0.0
    assert waveform._normalize_level(-90.0) == 0.0


def test_normalize_level_at_or_above_zero_dbfs_is_one():
    assert waveform._normalize_level(0.0) == 1.0
    assert waveform._normalize_level(5.0) == 1.0


def test_normalize_level_midpoint():
    # default floor is -60.0: -30 dBFS is exactly halfway to 0
    assert waveform._normalize_level(-30.0) == 0.5


def test_normalize_level_respects_custom_floor():
    assert waveform._normalize_level(-27.5, floor_dbfs=-55.0) == 0.5
