import os
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
    # pbcopy decodes stdin using the process's locale (LC_CTYPE) to build the
    # pasteboard string. Dictify normally runs as a launchd LaunchAgent (see
    # local.dictify.plist.template), whose environment has no LANG/LC_CTYPE
    # set at all, so pbcopy falls back to misreading multi-byte UTF-8
    # sequences byte-by-byte - e.g. turning a curly apostrophe into "‚Äô"
    # mojibake. Forcing LC_CTYPE=UTF-8 here makes the decoding correct
    # regardless of what locale (if any) the parent process was started
    # with.
    env = {**os.environ, "LC_CTYPE": "UTF-8"}
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True, env=env)


def paste_into_frontmost_app(delay_s: float = 0.05) -> None:
    time.sleep(delay_s)
    key_down = CGEventCreateKeyboardEvent(None, _KVK_ANSI_V, True)
    key_up = CGEventCreateKeyboardEvent(None, _KVK_ANSI_V, False)
    CGEventSetFlags(key_down, kCGEventFlagMaskCommand)
    CGEventSetFlags(key_up, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, key_down)
    CGEventPost(kCGHIDEventTap, key_up)
