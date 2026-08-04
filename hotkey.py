from typing import Callable, Optional

from pynput import keyboard


class HotkeyListener:
    def __init__(self, combo: str, on_toggle: Callable[[], None]):
        self._combo = combo
        self._on_toggle = on_toggle
        self._listener: Optional[keyboard.GlobalHotKeys] = None

    def start(self) -> None:
        self._listener = keyboard.GlobalHotKeys({self._combo: self._on_toggle})
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
