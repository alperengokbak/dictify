import subprocess
import time

from pynput.keyboard import Controller, Key


def copy_to_clipboard(text: str) -> None:
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)


def paste_into_frontmost_app(delay_s: float = 0.05) -> None:
    time.sleep(delay_s)
    controller = Controller()
    with controller.pressed(Key.cmd):
        controller.press("v")
        controller.release("v")
