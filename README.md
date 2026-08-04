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

Settings live in `~/.config/dictate-mac/config.json` and are created with defaults on first run; edit the file to change any of these.

### Hotkey

Default is Control+Option (`<ctrl>+<alt>`), stick to modifier-only combos if you change it. The global hotkey listener observes key events but does not consume them like a native macOS-registered shortcut would, so any printable key in the combo (letters, digits, Space, etc.) still reaches whatever app has focus and types there too — e.g. `<alt>+<space>` would insert a literal space into the focused text field every time you triggered the hotkey. Modifier keys (`<ctrl>`, `<alt>`, `<cmd>`, `<shift>`) never type anything on their own, so a combo made only of those has nothing to leak.

### Recording mode

Toggle (press to start, press again to stop) or push-to-talk (hold to record, release to stop) — switchable live from the menu bar's "Recording Mode" submenu.

### Language and style

Language can be forced to Turkish or English (instead of per-utterance auto-detect) from the "Language" submenu. Cleanup tone can be set to Default, Professional, or Casual from the "Style" submenu.

### Glossary

`"glossary"` is a list of proper nouns and jargon you say often (names, project/tool names, etc.) that speech-to-text tends to mangle:

```json
"glossary": ["Kubernetes", "PyQt", "Grafana"]
```

It's used two ways: as a hint to Whisper during transcription, and as a reference list the cleanup step uses to fix a misheard word back to the correct spelling — it won't rewrite words that are already correct or unrelated. Empty by default; edit `config.json` directly to add entries (no menu UI for this yet).
