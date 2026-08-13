import itertools
import struct
import traceback
from typing import Callable, Optional

import objc
from quickmachotkey._MinimalHIToolbox import (
    EventTypeSpec,
    GetEventDispatcherTarget,
    GetEventKind,
    GetEventParameter,
    InstallEventHandler,
    RegisterEventHotKey,
    RemoveEventHandler,
    UnregisterEventHotKey,
    kEventClassKeyboard,
    kEventHotKeyPressed,
    kEventHotKeyReleased,
    kEventParamDirectObject,
    typeEventHotKeyID,
)
from quickmachotkey.constants import (
    cmdKey,
    controlKey,
    kVK_ANSI_0,
    kVK_ANSI_1,
    kVK_ANSI_2,
    kVK_ANSI_3,
    kVK_ANSI_4,
    kVK_ANSI_5,
    kVK_ANSI_6,
    kVK_ANSI_7,
    kVK_ANSI_8,
    kVK_ANSI_9,
    kVK_ANSI_A,
    kVK_ANSI_B,
    kVK_ANSI_C,
    kVK_ANSI_D,
    kVK_ANSI_E,
    kVK_ANSI_F,
    kVK_ANSI_G,
    kVK_ANSI_H,
    kVK_ANSI_I,
    kVK_ANSI_J,
    kVK_ANSI_K,
    kVK_ANSI_L,
    kVK_ANSI_M,
    kVK_ANSI_N,
    kVK_ANSI_O,
    kVK_ANSI_P,
    kVK_ANSI_Q,
    kVK_ANSI_R,
    kVK_ANSI_S,
    kVK_ANSI_T,
    kVK_ANSI_U,
    kVK_ANSI_V,
    kVK_ANSI_W,
    kVK_ANSI_X,
    kVK_ANSI_Y,
    kVK_ANSI_Z,
    kVK_Delete,
    kVK_DownArrow,
    kVK_Escape,
    kVK_F1,
    kVK_F2,
    kVK_F3,
    kVK_F4,
    kVK_F5,
    kVK_F6,
    kVK_F7,
    kVK_F8,
    kVK_F9,
    kVK_F10,
    kVK_F11,
    kVK_F12,
    kVK_F13,
    kVK_F14,
    kVK_F15,
    kVK_F16,
    kVK_F17,
    kVK_F18,
    kVK_F19,
    kVK_F20,
    kVK_LeftArrow,
    kVK_Return,
    kVK_RightArrow,
    kVK_Space,
    kVK_Tab,
    kVK_UpArrow,
    optionKey,
    shiftKey,
)

# A hotkey combo is exactly one non-modifier key token plus zero or more
# modifier tokens. Unlike pynput's GlobalHotKeys (which only observes
# keystrokes), Carbon's RegisterEventHotKey consumes the matching event at
# the OS level before it reaches any app - the same mechanism Spotlight and
# Alfred use - so, unlike the old pynput-based listener, there's no risk of
# a leftover character leaking into whatever app has focus. That's why a
# combo is no longer restricted to modifier-only.
MODIFIER_TOKENS = {
    "<ctrl>": controlKey,
    "<alt>": optionKey,
    "<cmd>": cmdKey,
    "<shift>": shiftKey,
}

KEY_TOKENS = {
    "<a>": kVK_ANSI_A,
    "<b>": kVK_ANSI_B,
    "<c>": kVK_ANSI_C,
    "<d>": kVK_ANSI_D,
    "<e>": kVK_ANSI_E,
    "<f>": kVK_ANSI_F,
    "<g>": kVK_ANSI_G,
    "<h>": kVK_ANSI_H,
    "<i>": kVK_ANSI_I,
    "<j>": kVK_ANSI_J,
    "<k>": kVK_ANSI_K,
    "<l>": kVK_ANSI_L,
    "<m>": kVK_ANSI_M,
    "<n>": kVK_ANSI_N,
    "<o>": kVK_ANSI_O,
    "<p>": kVK_ANSI_P,
    "<q>": kVK_ANSI_Q,
    "<r>": kVK_ANSI_R,
    "<s>": kVK_ANSI_S,
    "<t>": kVK_ANSI_T,
    "<u>": kVK_ANSI_U,
    "<v>": kVK_ANSI_V,
    "<w>": kVK_ANSI_W,
    "<x>": kVK_ANSI_X,
    "<y>": kVK_ANSI_Y,
    "<z>": kVK_ANSI_Z,
    "<0>": kVK_ANSI_0,
    "<1>": kVK_ANSI_1,
    "<2>": kVK_ANSI_2,
    "<3>": kVK_ANSI_3,
    "<4>": kVK_ANSI_4,
    "<5>": kVK_ANSI_5,
    "<6>": kVK_ANSI_6,
    "<7>": kVK_ANSI_7,
    "<8>": kVK_ANSI_8,
    "<9>": kVK_ANSI_9,
    "<f1>": kVK_F1,
    "<f2>": kVK_F2,
    "<f3>": kVK_F3,
    "<f4>": kVK_F4,
    "<f5>": kVK_F5,
    "<f6>": kVK_F6,
    "<f7>": kVK_F7,
    "<f8>": kVK_F8,
    "<f9>": kVK_F9,
    "<f10>": kVK_F10,
    "<f11>": kVK_F11,
    "<f12>": kVK_F12,
    "<f13>": kVK_F13,
    "<f14>": kVK_F14,
    "<f15>": kVK_F15,
    "<f16>": kVK_F16,
    "<f17>": kVK_F17,
    "<f18>": kVK_F18,
    "<f19>": kVK_F19,
    "<f20>": kVK_F20,
    "<space>": kVK_Space,
    "<tab>": kVK_Tab,
    "<return>": kVK_Return,
    "<escape>": kVK_Escape,
    "<delete>": kVK_Delete,
    "<left>": kVK_LeftArrow,
    "<right>": kVK_RightArrow,
    "<up>": kVK_UpArrow,
    "<down>": kVK_DownArrow,
}

# Reverse lookup used by the Preferences capture UI: NSEvent.keyCode() and
# Carbon's kVK_* constants share the same virtual-keycode numbering, so a
# captured keydown's keyCode can be mapped straight back to its token.
KEY_TOKEN_BY_VIRTUAL_KEY = {code: token for token, code in KEY_TOKENS.items()}

_SIGNATURE = struct.unpack("@I", b"DCFY")[0]

_next_hotkey_id = itertools.count(1)


def _fired_hotkey_id(event) -> tuple[int, int]:
    """Reads the (signature, id) pair Carbon attached to the hotkey event
    that fired, via the same GetEventParameter/typeEventHotKeyID mechanism
    quickmachotkey's own quickHotKey decorator uses internally to route
    events to the right registration."""
    result, _actual_type, _actual_size, relayed_param = GetEventParameter(
        event, kEventParamDirectObject, typeEventHotKeyID, None, 8, None, None,
    )
    if result != 0:
        raise RuntimeError(f"GetEventParameter failed with OSStatus {result}")
    return struct.unpack("@II", relayed_param)


def parse_combo(combo: str):
    """Parse a combo string like "<ctrl>+<alt>+<d>" into a (virtual_key,
    modifier_mask) pair. Exactly one non-modifier key token is required;
    every other token must be a modifier."""
    tokens = [token.strip() for token in combo.split("+") if token.strip()]
    if not tokens:
        raise ValueError(f"empty hotkey combo: {combo!r}")

    virtual_key = None
    modifier_mask = 0
    for token in tokens:
        if token in MODIFIER_TOKENS:
            modifier_mask |= MODIFIER_TOKENS[token]
        elif token in KEY_TOKENS:
            if virtual_key is not None:
                raise ValueError(
                    f"hotkey combo must contain exactly one non-modifier key: {combo!r}"
                )
            virtual_key = KEY_TOKENS[token]
        else:
            raise ValueError(f"unknown hotkey token {token!r} in combo {combo!r}")

    if virtual_key is None:
        raise ValueError(
            f"hotkey combo must include one non-modifier key (e.g. <d>): {combo!r}"
        )
    return virtual_key, modifier_mask


def format_combo(virtual_key: int, modifier_mask: int) -> str:
    """The inverse of parse_combo: builds a combo string like
    "<ctrl>+<alt>+<escape>" from a (virtual_key, modifier_mask) pair.
    Modifier tokens appear in MODIFIER_TOKENS' insertion order (ctrl, alt,
    cmd, shift) so the result is deterministic."""
    if virtual_key not in KEY_TOKEN_BY_VIRTUAL_KEY:
        raise ValueError(f"unknown virtual key code: {virtual_key!r}")
    tokens = [token for token, bit in MODIFIER_TOKENS.items() if modifier_mask & bit]
    tokens.append(KEY_TOKEN_BY_VIRTUAL_KEY[virtual_key])
    return "+".join(tokens)


class HotkeyListener:
    """Wraps a global hotkey combo in either toggle or push-to-talk mode,
    backed by macOS's native Carbon RegisterEventHotKey API (via
    quickmachotkey's low-level, safely PyObjC-bridged primitives) instead of
    pynput's GlobalHotKeys/Listener. pynput's macOS keyboard listener has a
    longstanding bug where its background thread calls a main-thread-only
    Carbon TSM function, crashing the process with SIGTRAP; RegisterEventHotKey
    dispatches on the main run loop and sidesteps that failure mode entirely.

    Toggle mode: on_activate fires once per full combo press; on_deactivate
    is never called.
    Push-to-talk mode: on_activate fires on combo press, on_deactivate fires
    on combo release.
    """

    def __init__(
        self,
        combo: str,
        on_activate: Callable[[], None],
        on_deactivate: Optional[Callable[[], None]] = None,
        mode: str = "toggle",
    ):
        self._virtual_key, self._modifier_mask = parse_combo(combo)
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate
        self._mode = mode
        self._hotkey_ref = None
        self._handler_ref = None
        self._callback = None
        self._hotkey_id = next(_next_hotkey_id)

    def start(self) -> None:
        target = GetEventDispatcherTarget()

        @objc.callbackFor(InstallEventHandler)
        def callback(callref, event, void):
            try:
                kind = GetEventKind(event)
                if kind not in (kEventHotKeyPressed, kEventHotKeyReleased):
                    return 0
                if _fired_hotkey_id(event) != (_SIGNATURE, self._hotkey_id):
                    return 0
                if kind == kEventHotKeyPressed:
                    self._on_activate()
                elif kind == kEventHotKeyReleased and self._on_deactivate is not None:
                    self._on_deactivate()
            except Exception:
                traceback.print_exc()
            return 0

        # Keep the trampoline alive for the listener's lifetime - nothing
        # else holds a reference to it once start() returns.
        self._callback = callback

        specs = [
            EventTypeSpec(eventClass=kEventClassKeyboard, eventKind=kEventHotKeyPressed),
            EventTypeSpec(eventClass=kEventClassKeyboard, eventKind=kEventHotKeyReleased),
        ]
        result, handler_ref = InstallEventHandler(target, callback, 2, specs, None, None)
        if result != 0:
            self._callback = None
            raise RuntimeError(f"InstallEventHandler failed with OSStatus {result}")
        self._handler_ref = handler_ref

        hotkey_id = (_SIGNATURE, self._hotkey_id)
        result, hotkey_ref = RegisterEventHotKey(
            self._virtual_key, self._modifier_mask, hotkey_id, target, 0, None
        )
        if result != 0:
            RemoveEventHandler(self._handler_ref)
            self._handler_ref = None
            self._callback = None
            raise RuntimeError(f"RegisterEventHotKey failed with OSStatus {result}")
        self._hotkey_ref = hotkey_ref

    def stop(self) -> None:
        if self._hotkey_ref is not None:
            UnregisterEventHotKey(self._hotkey_ref)
            self._hotkey_ref = None
        if self._handler_ref is not None:
            RemoveEventHandler(self._handler_ref)
            self._handler_ref = None
        self._callback = None
