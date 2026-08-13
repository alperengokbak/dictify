import pytest

import hotkey
from quickmachotkey.constants import controlKey, kVK_ANSI_D, kVK_F13, optionKey


def test_parse_combo_single_modifier_plus_key():
    virtual_key, modifier_mask = hotkey.parse_combo("<ctrl>+<d>")
    assert virtual_key == kVK_ANSI_D
    assert modifier_mask == controlKey


def test_parse_combo_multiple_modifiers_plus_key():
    virtual_key, modifier_mask = hotkey.parse_combo("<ctrl>+<alt>+<d>")
    assert virtual_key == kVK_ANSI_D
    assert modifier_mask == controlKey | optionKey


def test_parse_combo_key_order_does_not_matter():
    virtual_key, modifier_mask = hotkey.parse_combo("<d>+<ctrl>+<alt>")
    assert virtual_key == kVK_ANSI_D
    assert modifier_mask == controlKey | optionKey


def test_parse_combo_function_key():
    virtual_key, modifier_mask = hotkey.parse_combo("<ctrl>+<f13>")
    assert virtual_key == kVK_F13
    assert modifier_mask == controlKey


def test_parse_combo_key_only_no_modifiers():
    virtual_key, modifier_mask = hotkey.parse_combo("<d>")
    assert virtual_key == kVK_ANSI_D
    assert modifier_mask == 0


def test_parse_combo_rejects_modifier_only_combo():
    with pytest.raises(ValueError, match="non-modifier key"):
        hotkey.parse_combo("<ctrl>+<alt>")


def test_parse_combo_rejects_empty_combo():
    with pytest.raises(ValueError, match="empty"):
        hotkey.parse_combo("")


def test_parse_combo_rejects_two_non_modifier_keys():
    with pytest.raises(ValueError, match="exactly one non-modifier key"):
        hotkey.parse_combo("<d>+<f13>")


def test_parse_combo_rejects_unknown_token():
    with pytest.raises(ValueError, match="unknown hotkey token"):
        hotkey.parse_combo("<ctrl>+<nope>")


def test_key_token_by_virtual_key_is_reverse_of_key_tokens():
    for token, code in hotkey.KEY_TOKENS.items():
        assert hotkey.KEY_TOKEN_BY_VIRTUAL_KEY[code] == token


def test_hotkey_listener_toggle_mode_start_and_stop_do_not_raise():
    listener = hotkey.HotkeyListener("<ctrl>+<alt>+<d>", on_activate=lambda: None)
    listener.start()
    listener.stop()


def test_hotkey_listener_push_to_talk_start_and_stop_do_not_raise():
    listener = hotkey.HotkeyListener(
        "<ctrl>+<alt>+<d>",
        on_activate=lambda: None,
        on_deactivate=lambda: None,
        mode="push_to_talk",
    )
    listener.start()
    listener.stop()


def test_hotkey_listener_constructor_rejects_invalid_combo():
    with pytest.raises(ValueError):
        hotkey.HotkeyListener("<ctrl>+<alt>", on_activate=lambda: None)


def test_format_combo_single_modifier_plus_key():
    assert hotkey.format_combo(kVK_ANSI_D, controlKey) == "<ctrl>+<d>"


def test_format_combo_multiple_modifiers_plus_key():
    assert hotkey.format_combo(kVK_ANSI_D, controlKey | optionKey) == "<ctrl>+<alt>+<d>"


def test_format_combo_key_only_no_modifiers():
    assert hotkey.format_combo(kVK_ANSI_D, 0) == "<d>"


def test_format_combo_is_inverse_of_parse_combo():
    combo = "<ctrl>+<alt>+<cmd>+<d>"
    virtual_key, modifier_mask = hotkey.parse_combo(combo)
    assert hotkey.format_combo(virtual_key, modifier_mask) == combo


def test_format_combo_rejects_unknown_virtual_key():
    with pytest.raises(ValueError, match="unknown virtual key"):
        hotkey.format_combo(9999, 0)


def test_hotkey_listener_instances_get_unique_ids():
    first = hotkey.HotkeyListener("<ctrl>+<alt>+<d>", on_activate=lambda: None)
    second = hotkey.HotkeyListener("<ctrl>+<alt>+<e>", on_activate=lambda: None)
    assert first._hotkey_id != second._hotkey_id


def test_hotkey_listener_two_instances_register_and_teardown_concurrently():
    # Regression: RegisterEventHotKey previously used a hardcoded ID for
    # every listener. That's harmless with one listener alive, but once a
    # second one is registered on the same dispatcher target its callback
    # would fire for both combos rather than just its own. This doesn't
    # verify dispatch (see the GetEventParameter caveat in the spec) - only
    # that two listeners can be alive and torn down at once without the
    # registration itself raising.
    first = hotkey.HotkeyListener("<ctrl>+<alt>+<d>", on_activate=lambda: None)
    second = hotkey.HotkeyListener("<ctrl>+<alt>+<escape>", on_activate=lambda: None)
    first.start()
    try:
        second.start()
        second.stop()
    finally:
        first.stop()
