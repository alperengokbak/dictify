from typing import Optional

import objc
from AppKit import (
    NSApp,
    NSAtTop,
    NSBackingStoreBuffered,
    NSBox,
    NSButton,
    NSButtonTypeSwitch,
    NSEvent,
    NSEventMaskFlagsChanged,
    NSEventMaskKeyDown,
    NSEventModifierFlagCommand,
    NSEventModifierFlagControl,
    NSEventModifierFlagOption,
    NSEventModifierFlagShift,
    NSEventTypeKeyDown,
    NSFont,
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

from PyObjCTools import AppHelper

import hotkey
from config import save_config

WINDOW_WIDTH = 520
MARGIN = 20
SECTION_GAP = 14

HOTKEY_SECTION_HEIGHT = 76
CLEANUP_SECTION_HEIGHT = 190
SILENCE_SECTION_HEIGHT = 100
HISTORY_SECTION_HEIGHT = 96
BUTTON_ROW_HEIGHT = 44

_MODIFIER_ORDER = [
    (NSEventModifierFlagControl, "<ctrl>"),
    (NSEventModifierFlagOption, "<alt>"),
    (NSEventModifierFlagShift, "<shift>"),
    (NSEventModifierFlagCommand, "<cmd>"),
]

_SYMBOL_MAP = {
    "<ctrl>": "⌃",
    "<alt>": "⌥",
    "<cmd>": "⌘",
    "<shift>": "⇧",
}


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


def _modifier_flags_to_combo(flags: int) -> str:
    parts = [token for bit, token in _MODIFIER_ORDER if flags & bit]
    return "+".join(parts)


def _combo_to_display_string(combo: str) -> str:
    if not combo:
        return ""
    parts = []
    for part in combo.split("+"):
        if part in _SYMBOL_MAP:
            parts.append(_SYMBOL_MAP[part])
        elif part.startswith("<") and part.endswith(">"):
            parts.append(part[1:-1].upper())
        else:
            parts.append(part)
    return "".join(parts)


def _keydown_to_combo(keycode: int, modifier_flags: int) -> Optional[str]:
    """Turn a captured keydown's virtual keycode + modifier flags into a
    combo string, or None if the key isn't one of our supported tokens."""
    key_token = hotkey.KEY_TOKEN_BY_VIRTUAL_KEY.get(keycode)
    if key_token is None:
        return None
    modifier_tokens = [token for bit, token in _MODIFIER_ORDER if modifier_flags & bit]
    return "+".join(modifier_tokens + [key_token])


class PreferencesWindowController(NSObject):
    @objc.python_method
    def configure(self, config, on_save):
        self.config = config
        self.on_save = on_save
        self._capture_monitor = None
        self._build_window()
        return self

    @objc.python_method
    def _add_label(self, parent, text, frame, font_size=13):
        label = NSTextField.alloc().initWithFrame_(frame)
        label.setStringValue_(text)
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setFont_(NSFont.systemFontOfSize_(font_size))
        parent.addSubview_(label)
        return label

    @objc.python_method
    def _make_box(self, title, y_top, height, width):
        box = NSBox.alloc().initWithFrame_(
            NSMakeRect(MARGIN, y_top - height, width - 2 * MARGIN, height)
        )
        box.setTitle_(title)
        box.setTitlePosition_(NSAtTop)
        box.setTitleFont_(NSFont.boldSystemFontOfSize_(14))
        return box

    @objc.python_method
    def _build_hotkey_section(self, y_top, width):
        box = self._make_box("Hotkey", y_top, HOTKEY_SECTION_HEIGHT, width)
        content = box.contentView()
        cw = content.bounds().size.width

        self.hotkey_display_label = self._add_label(
            content, _combo_to_display_string(self.config.get("hotkey", "")),
            NSMakeRect(0, 4, 70, 30), font_size=20,
        )

        self.hotkey_field = NSTextField.alloc().initWithFrame_(NSMakeRect(75, 8, cw - 210, 24))
        self.hotkey_field.setStringValue_(self.config.get("hotkey", ""))
        content.addSubview_(self.hotkey_field)

        self.capture_button = NSButton.alloc().initWithFrame_(NSMakeRect(cw - 125, 6, 125, 28))
        self.capture_button.setTitle_("Set Shortcut...")
        self.capture_button.setTarget_(self)
        self.capture_button.setAction_("startHotkeyCapture:")
        content.addSubview_(self.capture_button)

        self._add_label(
            content, "Any modifier + key combo works (e.g. ⌃⌥⌘D). It's captured"
            " system-wide and never reaches other apps.",
            NSMakeRect(0, 34, cw, 16), font_size=10,
        )
        return box

    @objc.python_method
    def _build_cleanup_section(self, y_top, width):
        box = self._make_box("Cleanup", y_top, CLEANUP_SECTION_HEIGHT, width)
        content = box.contentView()
        cw = content.bounds().size.width
        ch = content.bounds().size.height

        self.cleanup_checkbox = NSButton.alloc().initWithFrame_(
            NSMakeRect(0, ch - 24, 200, 24)
        )
        self.cleanup_checkbox.setButtonType_(NSButtonTypeSwitch)
        self.cleanup_checkbox.setTitle_("Enable cleanup")
        self.cleanup_checkbox.setState_(1 if self.config.get("cleanup_enabled", True) else 0)
        content.addSubview_(self.cleanup_checkbox)

        self._add_label(
            content, "Glossary (one term per line):", NSMakeRect(0, ch - 46, cw, 18)
        )

        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, cw, ch - 52))
        scroll.setHasVerticalScroller_(True)
        self.glossary_view = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, cw, ch - 52))
        self.glossary_view.setString_(_format_glossary_text(self.config.get("glossary", [])))
        scroll.setDocumentView_(self.glossary_view)
        content.addSubview_(scroll)
        return box

    @objc.python_method
    def _build_silence_section(self, y_top, width):
        box = self._make_box("Silence Detection", y_top, SILENCE_SECTION_HEIGHT, width)
        content = box.contentView()
        cw = content.bounds().size.width
        ch = content.bounds().size.height
        half = cw / 2

        self._add_label(content, "Peak floor (dBFS):", NSMakeRect(0, ch - 20, half - 10, 18))
        self.peak_field = NSTextField.alloc().initWithFrame_(NSMakeRect(0, ch - 44, half - 10, 24))
        self.peak_field.setStringValue_(str(self.config.get("silence_peak_floor_dbfs", -55.0)))
        content.addSubview_(self.peak_field)

        self._add_label(
            content, "Rise threshold (dB):", NSMakeRect(half, ch - 20, half - 10, 18)
        )
        self.rise_field = NSTextField.alloc().initWithFrame_(
            NSMakeRect(half, ch - 44, half - 10, 24)
        )
        self.rise_field.setStringValue_(str(self.config.get("silence_rise_db", 10.0)))
        content.addSubview_(self.rise_field)

        self._add_label(
            content,
            "A recording is dropped if it never rises this many dB above its own"
            " noise floor, or if its peak stays below the floor.",
            NSMakeRect(0, 0, cw, 30), font_size=10,
        )
        return box

    @objc.python_method
    def _build_history_section(self, y_top, width):
        box = self._make_box("History", y_top, HISTORY_SECTION_HEIGHT, width)
        content = box.contentView()
        cw = content.bounds().size.width
        ch = content.bounds().size.height

        self.history_checkbox = NSButton.alloc().initWithFrame_(
            NSMakeRect(0, ch - 24, 200, 24)
        )
        self.history_checkbox.setButtonType_(NSButtonTypeSwitch)
        self.history_checkbox.setTitle_("Enable history")
        self.history_checkbox.setState_(1 if self.config.get("history_enabled", True) else 0)
        content.addSubview_(self.history_checkbox)

        self._add_label(content, "Keep at most this many entries:", NSMakeRect(0, ch - 52, cw - 90, 18))
        self.history_limit_field = NSTextField.alloc().initWithFrame_(
            NSMakeRect(cw - 80, ch - 56, 80, 24)
        )
        self.history_limit_field.setStringValue_(str(self.config.get("history_limit", 200)))
        content.addSubview_(self.history_limit_field)
        return box

    @objc.python_method
    def _build_window(self):
        total_height = (
            2 * MARGIN
            + HOTKEY_SECTION_HEIGHT
            + CLEANUP_SECTION_HEIGHT
            + SILENCE_SECTION_HEIGHT
            + HISTORY_SECTION_HEIGHT
            + 4 * SECTION_GAP
            + BUTTON_ROW_HEIGHT
        )
        rect = NSMakeRect(200, 100, WINDOW_WIDTH, total_height)
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable
            | NSWindowStyleMaskResizable
        )
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, NSBackingStoreBuffered, False
        )
        self.window.setTitle_("Dictify Preferences")
        content = self.window.contentView()

        y = total_height - MARGIN

        hotkey_box = self._build_hotkey_section(y, WINDOW_WIDTH)
        content.addSubview_(hotkey_box)
        y -= HOTKEY_SECTION_HEIGHT + SECTION_GAP

        cleanup_box = self._build_cleanup_section(y, WINDOW_WIDTH)
        content.addSubview_(cleanup_box)
        y -= CLEANUP_SECTION_HEIGHT + SECTION_GAP

        silence_box = self._build_silence_section(y, WINDOW_WIDTH)
        content.addSubview_(silence_box)
        y -= SILENCE_SECTION_HEIGHT + SECTION_GAP

        history_box = self._build_history_section(y, WINDOW_WIDTH)
        content.addSubview_(history_box)
        y -= HISTORY_SECTION_HEIGHT + SECTION_GAP

        button_y = y - BUTTON_ROW_HEIGHT + 8
        save_button = NSButton.alloc().initWithFrame_(
            NSMakeRect(WINDOW_WIDTH - MARGIN - 120, button_y, 120, 32)
        )
        save_button.setTitle_("Save")
        save_button.setTarget_(self)
        save_button.setAction_("save:")
        save_button.setKeyEquivalent_("\r")
        content.addSubview_(save_button)

        cancel_button = NSButton.alloc().initWithFrame_(
            NSMakeRect(WINDOW_WIDTH - MARGIN - 230, button_y, 100, 32)
        )
        cancel_button.setTitle_("Cancel")
        cancel_button.setTarget_(self)
        cancel_button.setAction_("cancel:")
        content.addSubview_(cancel_button)

    @objc.python_method
    def show(self):
        NSApp.activateIgnoringOtherApps_(True)
        self.window.center()

        def _raise():
            self.window.makeKeyAndOrderFront_(None)
            self.window.orderFrontRegardless()

        # Deferred by one run-loop tick: activation is not always synchronous,
        # so ordering the window front in the same call can still lose to
        # whatever app was frontmost a moment ago.
        AppHelper.callAfter(_raise)

    @objc.python_method
    def _stop_hotkey_capture(self):
        if self._capture_monitor is not None:
            NSEvent.removeMonitor_(self._capture_monitor)
            self._capture_monitor = None
        self.capture_button.setTitle_("Set Shortcut...")
        self.capture_button.setEnabled_(True)

    def startHotkeyCapture_(self, sender):
        if self._capture_monitor is not None:
            # Already capturing - ignore a repeat click instead of installing
            # a second local event monitor on top of the first. Overwriting
            # self._capture_monitor would leak the original one: its
            # reference is lost, so it can never be removed, and it keeps
            # silently swallowing every keystroke (in every field, even
            # after this window closes) for the rest of the process's life.
            return
        self.capture_button.setTitle_("Press a key combo...")
        self.capture_button.setEnabled_(False)

        relevant_mask = (
            NSEventModifierFlagControl
            | NSEventModifierFlagOption
            | NSEventModifierFlagCommand
            | NSEventModifierFlagShift
        )

        def handler(event):
            # A combo finalizes the moment a supported non-modifier key is
            # pressed (with whatever modifiers are held at that instant) -
            # FlagsChanged readings alone never finalize anything anymore.
            # Every event handled here is consumed (returns None) so nothing
            # - modifier or key - ever leaks into whatever app is focused.
            if event.type() != NSEventTypeKeyDown:
                return None

            combo = _keydown_to_combo(event.keyCode(), event.modifierFlags() & relevant_mask)
            if combo is None:
                self.capture_button.setTitle_("Unsupported key - try again")
                return None

            self.hotkey_field.setStringValue_(combo)
            self.hotkey_display_label.setStringValue_(_combo_to_display_string(combo))
            self._stop_hotkey_capture()
            return None

        self._capture_monitor = NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
            NSEventMaskFlagsChanged | NSEventMaskKeyDown, handler
        )

    def save_(self, sender):
        self._stop_hotkey_capture()
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
        self._stop_hotkey_capture()
        self.window.close()
