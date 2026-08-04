from AppKit import (
    NSEventModifierFlagCommand,
    NSEventModifierFlagControl,
    NSEventModifierFlagOption,
    NSEventModifierFlagShift,
)

import preferences


def test_parse_glossary_text_splits_lines_and_strips_empties():
    result = preferences._parse_glossary_text("Kubernetes\n  PyQt  \n\nGrafana\n")
    assert result == ["Kubernetes", "PyQt", "Grafana"]


def test_parse_glossary_text_empty_input_returns_empty_list():
    assert preferences._parse_glossary_text("") == []
    assert preferences._parse_glossary_text("   \n  \n") == []


def test_format_glossary_text_joins_with_newlines():
    assert preferences._format_glossary_text(["Kubernetes", "PyQt"]) == "Kubernetes\nPyQt"


def test_format_glossary_text_empty_list_returns_empty_string():
    assert preferences._format_glossary_text([]) == ""


def test_parse_float_or_default_valid_input():
    assert preferences._parse_float_or_default("-55.0", 0.0) == -55.0
    assert preferences._parse_float_or_default("  10.5  ", 0.0) == 10.5


def test_parse_float_or_default_invalid_input_returns_default():
    assert preferences._parse_float_or_default("not a number", -55.0) == -55.0
    assert preferences._parse_float_or_default("", -55.0) == -55.0


def test_parse_int_or_default_valid_input():
    assert preferences._parse_int_or_default("200", 0) == 200
    assert preferences._parse_int_or_default("  42  ", 0) == 42


def test_parse_int_or_default_invalid_input_returns_default():
    assert preferences._parse_int_or_default("abc", 200) == 200
    assert preferences._parse_int_or_default("12.5", 200) == 200
    assert preferences._parse_int_or_default("", 200) == 200


def test_modifier_flags_to_combo_single_modifier():
    assert preferences._modifier_flags_to_combo(NSEventModifierFlagControl) == "<ctrl>"


def test_modifier_flags_to_combo_multiple_modifiers_use_fixed_order():
    flags = NSEventModifierFlagOption | NSEventModifierFlagControl
    assert preferences._modifier_flags_to_combo(flags) == "<ctrl>+<alt>"

    flags2 = NSEventModifierFlagCommand | NSEventModifierFlagShift
    assert preferences._modifier_flags_to_combo(flags2) == "<shift>+<cmd>"


def test_modifier_flags_to_combo_no_modifiers_is_empty_string():
    assert preferences._modifier_flags_to_combo(0) == ""


def test_combo_to_display_string_known_tokens():
    assert preferences._combo_to_display_string("<ctrl>+<alt>") == "⌃⌥"
    assert preferences._combo_to_display_string("<cmd>+<shift>") == "⌘⇧"


def test_combo_to_display_string_empty_combo():
    assert preferences._combo_to_display_string("") == ""


def test_combo_to_display_string_unknown_token_passthrough():
    assert preferences._combo_to_display_string("<ctrl>+<space>") == "⌃<space>"


def _new_controller():
    controller = preferences.PreferencesWindowController.alloc().init()
    controller._captured_flags = 0
    return controller


def test_capture_flags_finalizes_combo_on_full_release():
    controller = _new_controller()
    assert controller._process_capture_flags(NSEventModifierFlagControl) is None
    combo = controller._process_capture_flags(0)
    assert combo == "<ctrl>"


def test_capture_flags_remembers_full_combo_through_staggered_release():
    # Realistic human behavior: press ctrl, then also press alt, then release
    # alt first (flags drops back to ctrl-only) before finally releasing ctrl.
    # The finalized combo must still be "<ctrl>+<alt>", not just "<ctrl>".
    controller = _new_controller()
    assert controller._process_capture_flags(NSEventModifierFlagControl) is None
    assert (
        controller._process_capture_flags(
            NSEventModifierFlagControl | NSEventModifierFlagOption
        )
        is None
    )
    # alt released first: flags reading drops back to ctrl-only mid-release
    assert controller._process_capture_flags(NSEventModifierFlagControl) is None
    combo = controller._process_capture_flags(0)
    assert combo == "<ctrl>+<alt>"


def test_capture_flags_returns_none_while_still_holding_keys():
    controller = _new_controller()
    assert controller._process_capture_flags(NSEventModifierFlagCommand) is None
    assert (
        controller._process_capture_flags(
            NSEventModifierFlagCommand | NSEventModifierFlagShift
        )
        is None
    )


def test_start_hotkey_capture_ignores_repeat_click_instead_of_leaking_a_monitor():
    # A second click while already capturing must NOT install a second local
    # event monitor: overwriting self._capture_monitor would lose the first
    # one's reference forever, leaking a monitor that silently swallows every
    # keystroke (in every field) for the rest of the process's life.
    controller = preferences.PreferencesWindowController.alloc().init()
    controller.config = {}
    controller.on_save = None
    controller._capture_monitor = None
    controller._captured_flags = 0
    controller._build_window()

    controller.startHotkeyCapture_(None)
    first_monitor = controller._capture_monitor
    assert first_monitor is not None

    controller.startHotkeyCapture_(None)
    assert controller._capture_monitor is first_monitor

    controller._stop_hotkey_capture()
