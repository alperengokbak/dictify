# dictate-mac

A macOS menu bar app for push-to-talk dictation in Turkish and English, fully on-device (Whisper for transcription, a local Ollama model for cleanup).

## Install

```
./install.sh
```

This installs `whisper-cpp`, `ollama`, and `portaudio` via Homebrew, starts the Ollama service, pulls the `qwen2.5:3b` cleanup model, downloads the Whisper medium multilingual model, and sets up a Python virtualenv with the required packages.

## Run

```
.venv/bin/python dictate.py
```

The app lives in the menu bar. Press the hotkey once to start recording, press it again to stop; the transcript is cleaned up and pasted into the frontmost app automatically.

## Required macOS permissions

- **Microphone** — needed to record audio.
- **Accessibility** — needed for two things: the global hotkey listener, and simulating the paste keystroke into the frontmost app. Without Accessibility access, the simulated paste silently does nothing even though recording, transcription, and cleanup all still succeed — you'll get a correct transcript on the clipboard with no visible error, so if pasting never happens, check this first.

Grant both under System Settings > Privacy & Security, for whichever terminal or app you use to run `dictate.py`.

## Configuration

Default hotkey is Option+Space (`<alt>+<space>`). Settings live in `~/.config/dictate-mac/config.json` and are created with defaults on first run; edit the file to change the hotkey or other options.
