from typing import Callable, Optional

from pynput import keyboard


class _PressHoldHotKey(keyboard.HotKey):
    """A HotKey that also fires on_deactivate the moment any key of an
    already-active combo is released, enabling press-and-hold behavior."""

    def __init__(self, keys, on_activate, on_deactivate):
        super().__init__(keys, on_activate)
        self._on_deactivate = on_deactivate

    def release(self, key):
        was_active = self._keys.issubset(self._state)
        super().release(key)
        if was_active and key in self._keys:
            self._on_deactivate()


class HotkeyListener:
    """Wraps a global hotkey combo in either toggle or push-to-talk mode.

    Toggle mode: on_activate fires once per full combo press; on_deactivate
    is never called.
    Push-to-talk mode: on_activate fires on full combo press, on_deactivate
    fires the moment any key of that combo is released afterward.
    """

    def __init__(
        self,
        combo: str,
        on_activate: Callable[[], None],
        on_deactivate: Optional[Callable[[], None]] = None,
        mode: str = "toggle",
    ):
        self._combo = combo
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate
        self._mode = mode
        self._global_hotkeys: Optional[keyboard.GlobalHotKeys] = None
        self._listener: Optional[keyboard.Listener] = None

    def start(self) -> None:
        if self._mode == "push_to_talk":
            keys = keyboard.HotKey.parse(self._combo)
            on_deactivate = self._on_deactivate or (lambda: None)
            hot_key = _PressHoldHotKey(keys, self._on_activate, on_deactivate)

            def on_press(key):
                hot_key.press(self._listener.canonical(key))

            def on_release(key):
                hot_key.release(self._listener.canonical(key))

            self._listener = keyboard.Listener(
                on_press=on_press, on_release=on_release
            )
            self._listener.start()
        else:
            self._global_hotkeys = keyboard.GlobalHotKeys(
                {self._combo: self._on_activate}
            )
            self._global_hotkeys.start()

    def stop(self) -> None:
        if self._global_hotkeys is not None:
            self._global_hotkeys.stop()
            self._global_hotkeys = None
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
