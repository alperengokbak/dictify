from AppKit import NSSound

START_SOUND = "Tink"
STOP_SOUND = "Pop"


def play_sound(name: str) -> None:
    """Plays a named macOS system sound (e.g. "Tink", "Pop"). A silent
    no-op if the name isn't a recognized system sound - NSSound.soundNamed_
    returns None in that case rather than raising, so a typo'd/missing
    sound name degrades gracefully instead of crashing the caller."""
    sound = NSSound.soundNamed_(name)
    if sound is not None:
        sound.play()
