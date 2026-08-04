from AppKit import (
    NSEventModifierFlagCommand,
    NSEventModifierFlagControl,
    NSEventModifierFlagOption,
    NSEventModifierFlagShift,
)

import hotkey
import preferences
import whisper_models


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


def test_combo_to_display_string_key_token_uppercased():
    assert preferences._combo_to_display_string("<ctrl>+<alt>+<d>") == "⌃⌥D"
    assert preferences._combo_to_display_string("<ctrl>+<f13>") == "⌃F13"
    assert preferences._combo_to_display_string("<ctrl>+<space>") == "⌃SPACE"


def test_keydown_to_combo_key_only():
    combo = preferences._keydown_to_combo(hotkey.KEY_TOKENS["<d>"], 0)
    assert combo == "<d>"


def test_keydown_to_combo_with_modifiers():
    keycode = hotkey.KEY_TOKENS["<d>"]
    flags = NSEventModifierFlagControl | NSEventModifierFlagOption
    combo = preferences._keydown_to_combo(keycode, flags)
    assert combo == "<ctrl>+<alt>+<d>"


def test_keydown_to_combo_function_key():
    keycode = hotkey.KEY_TOKENS["<f13>"]
    combo = preferences._keydown_to_combo(keycode, NSEventModifierFlagControl)
    assert combo == "<ctrl>+<f13>"


def test_keydown_to_combo_unsupported_keycode_returns_none():
    assert preferences._keydown_to_combo(-1, NSEventModifierFlagControl) is None


def test_start_hotkey_capture_ignores_repeat_click_instead_of_leaking_a_monitor():
    # A second click while already capturing must NOT install a second local
    # event monitor: overwriting self._capture_monitor would lose the first
    # one's reference forever, leaking a monitor that silently swallows every
    # keystroke (in every field) for the rest of the process's life.
    controller = preferences.PreferencesWindowController.alloc().init()
    controller.config = {}
    controller.on_save = None
    controller._capture_monitor = None
    controller._build_window()

    controller.startHotkeyCapture_(None)
    first_monitor = controller._capture_monitor
    assert first_monitor is not None

    controller.startHotkeyCapture_(None)
    assert controller._capture_monitor is first_monitor

    controller._stop_hotkey_capture()


def _controller_with_window(config=None):
    controller = preferences.PreferencesWindowController.alloc().init()
    controller.config = config if config is not None else {}
    controller.on_save = None
    controller._capture_monitor = None
    controller._is_closed = False
    controller._build_window()
    return controller


def test_transcription_section_preselects_known_model_size():
    controller = _controller_with_window({"whisper_model_path": "/x/models/ggml-small.bin"})
    assert controller.model_popup.indexOfSelectedItem() == whisper_models.MODEL_SIZE_ORDER.index("small")
    assert controller.model_status_label.stringValue() == "Ready"
    assert controller._last_confirmed_model_index == whisper_models.MODEL_SIZE_ORDER.index("small")


def test_transcription_section_handles_custom_model_path():
    controller = _controller_with_window({"whisper_model_path": "/x/my-custom-model.bin"})
    assert controller.model_popup.indexOfSelectedItem() == -1
    assert controller._last_confirmed_model_index == -1
    assert "custom" in str(controller.model_status_label.stringValue()).lower()


def test_selecting_already_downloaded_size_updates_pending_without_download(tmp_path, monkeypatch):
    monkeypatch.setattr(preferences, "CONFIG_DIR", tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "ggml-tiny.bin").write_bytes(b"fake")

    controller = _controller_with_window({"whisper_model_path": str(tmp_path / "models" / "ggml-medium.bin")})
    controller.model_popup.selectItemAtIndex_(whisper_models.MODEL_SIZE_ORDER.index("tiny"))
    controller.modelSizeChanged_(None)

    assert controller._pending_model_size == "tiny"
    assert controller.model_status_label.stringValue() == "Ready"
    assert controller.model_progress.isHidden()


def test_on_model_download_succeeded_updates_state():
    controller = _controller_with_window({"whisper_model_path": "/x/models/ggml-medium.bin"})
    controller._set_model_downloading_ui(True)

    controller._on_model_download_succeeded("large", whisper_models.MODEL_SIZE_ORDER.index("large"))

    assert controller._pending_model_size == "large"
    assert controller._last_confirmed_model_index == whisper_models.MODEL_SIZE_ORDER.index("large")
    assert controller.model_status_label.stringValue() == "Ready"
    assert controller.model_progress.isHidden()
    assert controller.model_popup.isEnabled()


def test_on_model_download_failed_reverts_selection():
    controller = _controller_with_window({"whisper_model_path": "/x/models/ggml-medium.bin"})
    controller.model_popup.selectItemAtIndex_(whisper_models.MODEL_SIZE_ORDER.index("large"))
    controller._set_model_downloading_ui(True)

    controller._on_model_download_failed("network error")

    assert controller.model_popup.indexOfSelectedItem() == whisper_models.MODEL_SIZE_ORDER.index("medium")
    assert controller.model_progress.isHidden()
    assert controller.model_popup.isEnabled()


def test_on_model_download_failed_is_noop_after_window_closed():
    controller = _controller_with_window({"whisper_model_path": "/x/models/ggml-medium.bin"})
    controller.model_popup.selectItemAtIndex_(whisper_models.MODEL_SIZE_ORDER.index("large"))
    controller._set_model_downloading_ui(True)
    controller._is_closed = True

    controller._on_model_download_failed("network error")

    # Selection must NOT have been reverted to the pre-download index (medium):
    # the callback should have returned immediately without touching torn-down UI.
    assert controller.model_popup.indexOfSelectedItem() == whisper_models.MODEL_SIZE_ORDER.index("large")


def test_save_writes_pending_model_size_into_config():
    controller = _controller_with_window({"whisper_model_path": "/x/models/ggml-medium.bin"})
    controller._pending_model_size = "small"
    controller.hotkey_field.setStringValue_("")
    controller.glossary_view.setString_("")
    controller.peak_field.setStringValue_("-55.0")
    controller.rise_field.setStringValue_("10.0")
    controller.history_limit_field.setStringValue_("200")

    controller.save_(None)

    expected = str(whisper_models.model_path_for_size("small", preferences.CONFIG_DIR / "models"))
    assert controller.config["whisper_model_path"] == expected
