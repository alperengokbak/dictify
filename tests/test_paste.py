from unittest.mock import patch

import paste


@patch("paste.subprocess.run")
def test_copy_to_clipboard_forces_utf8_locale_for_pbcopy(mock_run):
    # Root cause: Dictify normally runs as a launchd LaunchAgent (see
    # local.dictify.plist.template), whose environment has no LANG/LC_CTYPE
    # at all. pbcopy decodes its stdin using the process locale to build the
    # pasteboard string, so with no locale set it misreads multi-byte UTF-8
    # sequences (e.g. the curly apostrophe Whisper emits) one byte at a time,
    # producing mojibake like "‚Äôt" instead of "'t". Explicitly forcing
    # LC_CTYPE=UTF-8 for the pbcopy subprocess fixes this regardless of what
    # locale the parent process was started with.
    paste.copy_to_clipboard("don’t")

    _, kwargs = mock_run.call_args
    assert kwargs["env"]["LC_CTYPE"] == "UTF-8"
    # Still needs the rest of the environment (PATH, etc.) to find pbcopy.
    assert "PATH" in kwargs["env"]
