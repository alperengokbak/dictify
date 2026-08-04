import objc
from AppKit import (
    NSApp,
    NSBackingStoreBuffered,
    NSButton,
    NSButtonTypeSwitch,
    NSMakeRect,
    NSObject,
    NSScrollView,
    NSTextField,
    NSTextView,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskResizable,
    NSWindowStyleMaskTitled,
)

from config import save_config


def _parse_glossary_text(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _format_glossary_text(glossary: list[str]) -> str:
    return "\n".join(glossary)


def _parse_float_or_default(text: str, default: float) -> float:
    try:
        return float(text.strip())
    except (ValueError, AttributeError):
        return default


def _parse_int_or_default(text: str, default: int) -> int:
    try:
        return int(text.strip())
    except (ValueError, AttributeError):
        return default


class PreferencesWindowController(NSObject):
    @objc.python_method
    def configure(self, config, on_save):
        self.config = config
        self.on_save = on_save
        self._build_window()
        return self

    @objc.python_method
    def _add_label(self, content, text, y):
        label = NSTextField.alloc().initWithFrame_(NSMakeRect(20, y, 440, 20))
        label.setStringValue_(text)
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        content.addSubview_(label)

    @objc.python_method
    def _build_window(self):
        rect = NSMakeRect(200, 200, 480, 480)
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable
            | NSWindowStyleMaskResizable
        )
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, NSBackingStoreBuffered, False
        )
        self.window.setTitle_("dictate-mac Preferences")
        content = self.window.contentView()

        y = 440

        self._add_label(content, "Hotkey (modifier-only combo, e.g. <ctrl>+<alt>):", y)
        y -= 24
        self.hotkey_field = NSTextField.alloc().initWithFrame_(NSMakeRect(20, y, 440, 24))
        self.hotkey_field.setStringValue_(self.config.get("hotkey", ""))
        content.addSubview_(self.hotkey_field)
        y -= 36

        self._add_label(content, "Glossary (one term per line):", y)
        y -= 110
        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(20, y, 440, 100))
        scroll.setHasVerticalScroller_(True)
        self.glossary_view = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, 440, 100))
        self.glossary_view.setString_(_format_glossary_text(self.config.get("glossary", [])))
        scroll.setDocumentView_(self.glossary_view)
        content.addSubview_(scroll)
        y -= 16

        self._add_label(content, "Silence peak floor (dBFS):", y)
        y -= 24
        self.peak_field = NSTextField.alloc().initWithFrame_(NSMakeRect(20, y, 200, 24))
        self.peak_field.setStringValue_(str(self.config.get("silence_peak_floor_dbfs", -55.0)))
        content.addSubview_(self.peak_field)
        y -= 36

        self._add_label(content, "Silence rise (dB):", y)
        y -= 24
        self.rise_field = NSTextField.alloc().initWithFrame_(NSMakeRect(20, y, 200, 24))
        self.rise_field.setStringValue_(str(self.config.get("silence_rise_db", 10.0)))
        content.addSubview_(self.rise_field)
        y -= 36

        self._add_label(content, "History limit:", y)
        y -= 24
        self.history_limit_field = NSTextField.alloc().initWithFrame_(NSMakeRect(20, y, 200, 24))
        self.history_limit_field.setStringValue_(str(self.config.get("history_limit", 200)))
        content.addSubview_(self.history_limit_field)
        y -= 40

        self.cleanup_checkbox = NSButton.alloc().initWithFrame_(NSMakeRect(20, y, 200, 24))
        self.cleanup_checkbox.setButtonType_(NSButtonTypeSwitch)
        self.cleanup_checkbox.setTitle_("Enable cleanup")
        self.cleanup_checkbox.setState_(1 if self.config.get("cleanup_enabled", True) else 0)
        content.addSubview_(self.cleanup_checkbox)

        self.history_checkbox = NSButton.alloc().initWithFrame_(NSMakeRect(240, y, 200, 24))
        self.history_checkbox.setButtonType_(NSButtonTypeSwitch)
        self.history_checkbox.setTitle_("Enable history")
        self.history_checkbox.setState_(1 if self.config.get("history_enabled", True) else 0)
        content.addSubview_(self.history_checkbox)
        y -= 44

        save_button = NSButton.alloc().initWithFrame_(NSMakeRect(340, y, 120, 32))
        save_button.setTitle_("Save")
        save_button.setTarget_(self)
        save_button.setAction_("save:")
        save_button.setKeyEquivalent_("\r")
        content.addSubview_(save_button)

        cancel_button = NSButton.alloc().initWithFrame_(NSMakeRect(220, y, 100, 32))
        cancel_button.setTitle_("Cancel")
        cancel_button.setTarget_(self)
        cancel_button.setAction_("cancel:")
        content.addSubview_(cancel_button)

    @objc.python_method
    def show(self):
        self.window.center()
        self.window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)

    def save_(self, sender):
        hotkey_value = str(self.hotkey_field.stringValue()).strip()
        if hotkey_value:
            self.config["hotkey"] = hotkey_value
        self.config["glossary"] = _parse_glossary_text(str(self.glossary_view.string()))
        self.config["silence_peak_floor_dbfs"] = _parse_float_or_default(
            str(self.peak_field.stringValue()), self.config.get("silence_peak_floor_dbfs", -55.0)
        )
        self.config["silence_rise_db"] = _parse_float_or_default(
            str(self.rise_field.stringValue()), self.config.get("silence_rise_db", 10.0)
        )
        self.config["history_limit"] = _parse_int_or_default(
            str(self.history_limit_field.stringValue()), self.config.get("history_limit", 200)
        )
        self.config["cleanup_enabled"] = bool(self.cleanup_checkbox.state())
        self.config["history_enabled"] = bool(self.history_checkbox.state())

        save_config(self.config)
        self.window.close()
        if self.on_save:
            self.on_save()

    def cancel_(self, sender):
        self.window.close()
