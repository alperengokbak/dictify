import subprocess
import time

from Quartz import (
    CGEventCreateKeyboardEvent,
    CGEventPost,
    CGEventSetFlags,
    kCGEventFlagMaskCommand,
    kCGHIDEventTap,
)

# Virtual keycode for 'v' - fixed across keyboard layouts, so simulating the
# paste keystroke needs no character-to-keycode lookup at all. That lookup is
# exactly what crashed here before: pynput's Controller resolves characters
# to keycodes via a Carbon TSM call that's only safe on the main thread, but
# this function runs on a background thread (_process_recording's own
# thread) - triggering the same SIGTRAP bug the hotkey listener had.
_KVK_ANSI_V = 0x09


def copy_to_clipboard(text: str) -> None:
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)


def paste_into_frontmost_app(delay_s: float = 0.05) -> None:
    time.sleep(delay_s)
    key_down = CGEventCreateKeyboardEvent(None, _KVK_ANSI_V, True)
    key_up = CGEventCreateKeyboardEvent(None, _KVK_ANSI_V, False)
    CGEventSetFlags(key_down, kCGEventFlagMaskCommand)
    CGEventSetFlags(key_up, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, key_down)
    CGEventPost(kCGHIDEventTap, key_up)
