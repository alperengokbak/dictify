import fcntl
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import rumps
from AppKit import (
    NSApp,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSModalResponseOK,
    NSOpenPanel,
)

from audio import is_silent, record, save_wav
from cleanup import CleanupError, clean_transcript
from config import DEFAULT_CONFIG, load_config, save_config
from feedback import START_SOUND, STOP_SOUND, play_sound
from filetranscribe import FileTranscribeError, transcribe_file
from history import append_entry, clear_history, load_history
from hotkey import HotkeyListener
from paste import copy_to_clipboard, paste_into_frontmost_app
from preferences import PreferencesWindowController
from transcribe import TranscribeError, transcribe
from waveform import WaveformWindowController
import whisper_server

IDLE_TITLE = "🎙"
RECORDING_TITLE = "🔴"
PROCESSING_TITLE = "⏳"
SAMPLE_RATE = 16000
LOCK_PATH = Path.home() / ".config" / "dictify" / "dictify.lock"

LANGUAGE_LABELS = {
    "auto": "Auto (detect)",
    "tr": "Turkish",
    "en": "English",
}

STYLE_LABELS = {
    "default": "Default",
    "professional": "Professional",
    "casual": "Casual",
}

RECORDING_MODE_LABELS = {
    "toggle": "Toggle (press to start/stop)",
    "push_to_talk": "Push to Talk (hold to record)",
}


def _format_last_transcript_label(text, limit=50):
    single_line = " ".join(text.split())
    truncated = single_line if len(single_line) <= limit else single_line[:limit] + "…"
    return f"Last: {truncated}"


_lock_file_descriptor = None  # kept open for the process lifetime - closing it (or process exit/crash) releases the OS-level lock


def _acquire_singleton_lock(lock_path: Path) -> bool:
    """Returns True and holds an exclusive OS-level lock on lock_path for
    this process's lifetime if no other live instance holds it. Returns
    False (leaving lock_path's existing content untouched) if another
    instance already holds the lock.

    Uses fcntl.flock rather than reading/comparing a stored PID: the kernel
    releases the lock automatically on process exit for any reason (clean
    quit, crash, kill -9), so there's no stale-lock-file class of bug to
    handle, and acquisition is atomic (no check-then-write race)."""
    global _lock_file_descriptor
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return False
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    _lock_file_descriptor = fd
    return True


class DictateApp(rumps.App):
    def __init__(self):
        super().__init__("Dictify", title=IDLE_TITLE)
        rumps.events.before_quit.register(self._on_quit)
        self.config = load_config()
        self.state = "idle"
        self._stop_event = None
        self._record_thread = None
        self._samples = None
        self._option_items = {}
        self.hotkey = None
        self._preferences_controller = None
        self.waveform = WaveformWindowController()

        language_menu = self._build_option_submenu("Language", "language", LANGUAGE_LABELS)
        style_menu = self._build_option_submenu("Style", "style", STYLE_LABELS)
        recording_mode_menu = self._build_option_submenu(
            "Recording Mode",
            "recording_mode",
            RECORDING_MODE_LABELS,
            on_change=self._restart_hotkey_listener,
        )
        self._last_transcript_last_text = None
        self._last_transcript_item = rumps.MenuItem("Last: (none yet)", callback=None)
        self._seed_last_transcript_item()

        history_menu = rumps.MenuItem("History")
        history_menu.add(rumps.MenuItem("Show History", callback=self._show_history))
        history_menu.add(rumps.MenuItem("Clear History", callback=self._clear_history))

        transcribe_file_item = rumps.MenuItem(
            "Transcribe File...", callback=self._transcribe_file_menu
        )
        preferences_item = rumps.MenuItem("Preferences...", callback=self._show_preferences)

        self.menu = [
            language_menu,
            style_menu,
            recording_mode_menu,
            self._last_transcript_item,
            history_menu,
            transcribe_file_item,
            preferences_item,
        ]
        self._refresh_all_checkmarks()

        self._start_hotkey_listener()

    def _seed_last_transcript_item(self):
        try:
            entries = load_history()
        except Exception:
            # Startup must not die because history.jsonl is unreadable (bad
            # permissions, replaced by a directory, etc.) - just leave the
            # menu item showing its empty-history placeholder.
            return
        if entries:
            self._update_last_transcript_item(entries[-1].get("final_text", ""))

    def _update_last_transcript_item(self, text):
        self._last_transcript_last_text = text
        self._last_transcript_item.title = _format_last_transcript_label(text)
        self._last_transcript_item.set_callback(self._copy_last_transcript)

    def _copy_last_transcript(self, sender):
        copy_to_clipboard(self._last_transcript_last_text)

    def _show_preferences(self, sender):
        self._preferences_controller = PreferencesWindowController.alloc().init()
        self._whisper_model_path_before_edit = self.config.get("whisper_model_path")
        self._preferences_controller.configure(self.config, self._on_preferences_saved)
        self._preferences_controller.show()

    def _on_preferences_saved(self):
        self._refresh_all_checkmarks()
        self._restart_hotkey_listener()
        self._stop_whisper_server_if_model_changed()

    def _stop_whisper_server_if_model_changed(self):
        if self.config.get("whisper_model_path") != getattr(
            self, "_whisper_model_path_before_edit", None
        ):
            whisper_server.stop()

    def _on_quit(self):
        whisper_server.stop()

    def _clean_with_fallback(self, raw_text, language=None):
        final_text = raw_text
        if self.config.get("cleanup_enabled", True):
            try:
                final_text = clean_transcript(raw_text, self.config, language=language)
            except CleanupError as exc:
                rumps.notification(
                    "Dictify", "Cleanup failed, using raw transcript", str(exc)
                )
                final_text = raw_text
        return final_text

    def _record_history(self, raw_text, final_text, language):
        if self.config.get("history_enabled", True):
            append_entry(
                raw_text,
                final_text,
                language,
                self.config.get("style", "default"),
                limit=self.config.get("history_limit", 200),
            )

    def _transcribe_file_menu(self, sender):
        panel = NSOpenPanel.openPanel()
        panel.setCanChooseFiles_(True)
        panel.setCanChooseDirectories_(False)
        panel.setAllowsMultipleSelection_(False)
        panel.setTitle_("Choose an audio or video file to transcribe")
        if panel.runModal() != NSModalResponseOK:
            return
        urls = panel.URLs()
        if not urls:
            return
        input_path = str(urls[0].path())
        threading.Thread(target=self._run_file_transcription, args=(input_path,)).start()

    def _run_file_transcription(self, input_path):
        try:
            raw_text, language = transcribe_file(input_path, self.config)
            final_text = self._clean_with_fallback(raw_text, language=language)
            self._update_last_transcript_item(final_text)

            output_path = Path(input_path).with_suffix(".txt")
            output_path.write_text(final_text)
            subprocess.run(["open", str(output_path)])
            rumps.notification("Dictify", "File transcribed", f"Saved to {output_path.name}")

            self._record_history(raw_text, final_text, language)
        except FileTranscribeError as exc:
            rumps.notification("Dictify", "File transcription failed", str(exc))
        except Exception as exc:
            rumps.notification("Dictify", "File transcription failed", str(exc))

    def _show_history(self, sender):
        entries = load_history()
        if not entries:
            rumps.notification("Dictify", "History", "No dictations recorded yet.")
            return

        lines = []
        for entry in reversed(entries):
            timestamp = entry.get("timestamp", "?")
            language = entry.get("language", "?")
            style = entry.get("style", "?")
            lines.append(f"[{timestamp}] ({language}, {style})")
            lines.append(entry.get("final_text", ""))
            lines.append("")

        history_view_path = Path(tempfile.gettempdir()) / "dictify-history.txt"
        history_view_path.write_text("\n".join(lines))
        subprocess.run(["open", str(history_view_path)])

    def _clear_history(self, sender):
        clear_history()
        self._last_transcript_last_text = None
        self._last_transcript_item.title = "Last: (none yet)"
        self._last_transcript_item.set_callback(None)
        rumps.notification("Dictify", "History", "Dictation history cleared.")

    def _build_option_submenu(self, title, config_key, labels, on_change=None):
        items = {
            value: rumps.MenuItem(
                label, callback=self._make_option_callback(config_key, value, on_change)
            )
            for value, label in labels.items()
        }
        self._option_items[config_key] = items
        submenu = rumps.MenuItem(title)
        for item in items.values():
            submenu.add(item)
        return submenu

    def _make_option_callback(self, config_key, value, on_change=None):
        def callback(sender):
            self.config[config_key] = value
            save_config(self.config)
            self._refresh_all_checkmarks()
            if on_change:
                on_change()

        return callback

    def _refresh_all_checkmarks(self):
        for config_key, items in self._option_items.items():
            current = self.config.get(config_key)
            for value, item in items.items():
                item.state = value == current

    def _build_hotkey_listener(self, combo):
        if self.config.get("recording_mode", "toggle") == "push_to_talk":
            return HotkeyListener(
                combo,
                on_activate=self._on_hotkey_press,
                on_deactivate=self._on_hotkey_release,
                mode="push_to_talk",
            )
        return HotkeyListener(combo, on_activate=self.on_hotkey)

    def _start_hotkey_listener(self):
        combo = self.config.get("hotkey", DEFAULT_CONFIG["hotkey"])
        try:
            self.hotkey = self._build_hotkey_listener(combo)
        except ValueError:
            # Old-format modifier-only combos (e.g. "<ctrl>+<alt>") saved by
            # a previous version are no longer valid: RegisterEventHotKey
            # requires a real, non-modifier key. Fall back to the current
            # default instead of crashing on startup.
            combo = DEFAULT_CONFIG["hotkey"]
            self.config["hotkey"] = combo
            save_config(self.config)
            try:
                rumps.notification(
                    "Dictify",
                    "Hotkey reset",
                    f"Your saved hotkey was in an old format; reset to {combo}.",
                )
            except Exception:
                # Startup must not die because the notification center isn't
                # available yet - the config fix above already happened.
                pass
            self.hotkey = self._build_hotkey_listener(combo)
        self.hotkey.start()

    def _restart_hotkey_listener(self):
        if self.hotkey is not None:
            self.hotkey.stop()
        self._start_hotkey_listener()

    def on_hotkey(self):
        if self.state == "idle":
            self._start_recording()
        elif self.state == "recording":
            self._stop_recording()
        # "processing": ignored, recording/processing never queues

    def _on_hotkey_press(self):
        if self.state == "idle":
            self._start_recording()
        # "recording"/"processing": ignore a repeat press while held down

    def _on_hotkey_release(self):
        if self.state == "recording":
            self._stop_recording()
        # ignore releases that don't correspond to an active recording

    def _play_start_sound(self):
        if self.config.get("sound_feedback_enabled", True):
            play_sound(START_SOUND)

    def _play_stop_sound(self):
        if self.config.get("sound_feedback_enabled", True):
            play_sound(STOP_SOUND)

    def _start_recording(self):
        self.state = "recording"
        self._play_start_sound()
        self.title = RECORDING_TITLE
        self._stop_event = threading.Event()
        self._samples = None
        self.waveform.show()

        def run():
            try:
                self._samples = record(
                    self._stop_event, SAMPLE_RATE, on_chunk=self.waveform.push_level
                )
            except Exception as exc:
                self._samples = None
                rumps.notification("Dictify", "Recording failed", str(exc))
                self.waveform.hide()
                self.state = "idle"
                self.title = IDLE_TITLE

        self._record_thread = threading.Thread(target=run)
        self._record_thread.start()

    def _stop_recording(self):
        self.state = "processing"
        self._play_stop_sound()
        self.title = PROCESSING_TITLE
        self.waveform.hide()
        self._stop_event.set()
        self._record_thread.join()
        threading.Thread(target=self._process_recording, args=(self._samples,)).start()

    def _process_recording(self, samples):
        try:
            if samples is None or samples.size == 0 or is_silent(
                samples,
                SAMPLE_RATE,
                peak_floor_dbfs=self.config["silence_peak_floor_dbfs"],
                rise_db=self.config["silence_rise_db"],
            ):
                return

            with tempfile.TemporaryDirectory() as tmpdir:
                wav_path = str(Path(tmpdir) / "recording.wav")
                save_wav(samples, SAMPLE_RATE, wav_path)

                try:
                    raw_text, language = transcribe(wav_path, self.config)
                except TranscribeError as exc:
                    rumps.notification("Dictify", "Transcription failed", str(exc))
                    return

            if not raw_text:
                return

            final_text = self._clean_with_fallback(raw_text, language=language)

            self._update_last_transcript_item(final_text)

            copy_to_clipboard(final_text)
            paste_into_frontmost_app()

            self._record_history(raw_text, final_text, language)
        except Exception as exc:
            rumps.notification("Dictify", "Dictation failed", str(exc))
        finally:
            self.state = "idle"
            self.title = IDLE_TITLE


def main():
    if not _acquire_singleton_lock(LOCK_PATH):
        print(f"Another Dictify instance holds {LOCK_PATH}; exiting.", file=sys.stderr)
        return
    # rumps never sets this itself. Harmless no-op when launched from
    # Dictify.app (which has a real Info.plist with LSUIElement), but still
    # required when run directly as `.venv/bin/python dictate.py` - without
    # it the app defaults to NSApplicationActivationPolicyProhibited, which
    # structurally blocks it from ever gaining real keyboard focus no
    # matter how hard `activate()` is called elsewhere.
    NSApplication.sharedApplication()  # guarantee it exists before we configure it
    NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    DictateApp().run()


if __name__ == "__main__":
    main()
