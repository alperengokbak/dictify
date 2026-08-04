from pynput import keyboard

import hotkey


def _parsed_keys():
    keys = keyboard.HotKey.parse("<alt>+<space>")
    return list(keys)


def test_press_hold_hotkey_activates_only_once_all_keys_pressed():
    activated = []
    deactivated = []
    alt_key, space_key = _parsed_keys()
    hk = hotkey._PressHoldHotKey(
        [alt_key, space_key], lambda: activated.append(1), lambda: deactivated.append(1)
    )

    hk.press(alt_key)
    assert activated == []

    hk.press(space_key)
    assert activated == [1]
    assert deactivated == []


def test_press_hold_hotkey_deactivates_on_first_key_released_after_activation():
    activated = []
    deactivated = []
    alt_key, space_key = _parsed_keys()
    hk = hotkey._PressHoldHotKey(
        [alt_key, space_key], lambda: activated.append(1), lambda: deactivated.append(1)
    )
    hk.press(alt_key)
    hk.press(space_key)

    hk.release(space_key)
    assert deactivated == [1]


def test_press_hold_hotkey_does_not_deactivate_from_partial_press():
    activated = []
    deactivated = []
    alt_key, space_key = _parsed_keys()
    hk = hotkey._PressHoldHotKey(
        [alt_key, space_key], lambda: activated.append(1), lambda: deactivated.append(1)
    )

    hk.press(alt_key)
    hk.release(alt_key)  # released before the combo was ever fully active

    assert activated == []
    assert deactivated == []


def test_press_hold_hotkey_only_deactivates_once_per_activation():
    deactivated = []
    alt_key, space_key = _parsed_keys()
    hk = hotkey._PressHoldHotKey(
        [alt_key, space_key], lambda: None, lambda: deactivated.append(1)
    )
    hk.press(alt_key)
    hk.press(space_key)
    hk.release(space_key)
    hk.release(alt_key)

    assert deactivated == [1]


def test_hotkey_listener_push_to_talk_start_and_stop_do_not_raise():
    listener = hotkey.HotkeyListener(
        "<alt>+<space>",
        on_activate=lambda: None,
        on_deactivate=lambda: None,
        mode="push_to_talk",
    )
    listener.start()
    listener.stop()


def test_hotkey_listener_toggle_mode_still_works():
    listener = hotkey.HotkeyListener("<alt>+<space>", on_activate=lambda: None)
    listener.start()
    listener.stop()
