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
        try:
            sound.play()
        except Exception:
            # In practice NSSound.play() returns NO on failure rather than
            # raising, but a raise here must never propagate into the
            # recording state machine (_start_recording/_stop_recording
            # call this mid-transition).
            pass
