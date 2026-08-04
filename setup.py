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
        },
    }
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options=OPTIONS,
    setup_requires=["py2app"],
)
