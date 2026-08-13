import feedback


class _FakeSound:
    def __init__(self):
        self.played = False

    def play(self):
        self.played = True


class _FakeNSSound:
    """Stands in for AppKit.NSSound so tests never touch real audio playback."""

    last_requested_name = None
    _sound_to_return = None

    @classmethod
    def soundNamed_(cls, name):
        cls.last_requested_name = name
        return cls._sound_to_return


def test_play_sound_plays_the_named_sound(monkeypatch):
    fake_sound = _FakeSound()
    _FakeNSSound._sound_to_return = fake_sound
    monkeypatch.setattr(feedback, "NSSound", _FakeNSSound)

    feedback.play_sound("Tink")

    assert _FakeNSSound.last_requested_name == "Tink"
    assert fake_sound.played is True


def test_play_sound_unknown_name_is_a_noop(monkeypatch):
    _FakeNSSound._sound_to_return = None
    monkeypatch.setattr(feedback, "NSSound", _FakeNSSound)

    feedback.play_sound("NotARealSound")  # must not raise


def test_play_sound_swallows_exception_from_play(monkeypatch):
    class _RaisingSound:
        def play(self):
            raise RuntimeError("boom")

    _FakeNSSound._sound_to_return = _RaisingSound()
    monkeypatch.setattr(feedback, "NSSound", _FakeNSSound)

    feedback.play_sound("Tink")  # must not raise


def test_start_and_stop_sound_constants():
    assert feedback.START_SOUND == "Tink"
    assert feedback.STOP_SOUND == "Pop"


def test_cancel_sound_constant():
    assert feedback.CANCEL_SOUND == "Basso"
