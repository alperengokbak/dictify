"""Builds Dictify.app. Must be run with the -A (alias) flag:

    .venv/bin/python setup.py py2app -A

Alias mode is required, not optional - it produces a bundle that
references this repo's live files and .venv by path instead of freezing a
standalone copy, preserving the project's live-edit development workflow.
Running without -A produces a full/frozen build, which is out of scope
for this project.
"""
from setuptools import setup

APP = ["dictate.py"]
DATA_FILES = []
OPTIONS = {
    "py2app": {
        "iconfile": "assets/AppIcon.icns",
        "plist": {
            "CFBundleName": "Dictify",
            "CFBundleDisplayName": "Dictify",
            "CFBundleIdentifier": "local.dictify",
            "CFBundleShortVersionString": "1.0",
            "CFBundleVersion": "1",
            "LSUIElement": True,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "NSMicrophoneUsageDescription": "Dictify records your voice to transcribe it into text.",
        },
    }
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options=OPTIONS,
    setup_requires=["py2app"],
)
