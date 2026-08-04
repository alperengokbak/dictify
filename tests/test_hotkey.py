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
