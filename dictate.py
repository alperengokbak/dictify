import subprocess
import tempfile
import threading
from pathlib import Path

import rumps
from AppKit import NSModalResponseOK, NSOpenPanel

from audio import is_silent, record, save_wav
from cleanup import CleanupError, clean_transcript
from config import load_config, save_config
from filetranscribe import FileTranscribeError, transcribe_file
from history import append_entry, clear_history, load_history
from hotkey import HotkeyListener
from paste import copy_to_clipboard, paste_into_frontmost_app
from preferences import PreferencesWindowController
from transcribe import TranscribeError, transcribe

IDLE_TITLE = "🎙"
RECORDING_TITLE = "🔴"
PROCESSING_TITLE = "⏳"
SAMPLE_RATE = 16000

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


class DictateApp(rumps.App):
    def __init__(self):
        super().__init__("dictate-mac", title=IDLE_TITLE)
        self.config = load_config()
        self.state = "idle"
        self._stop_event = None
        self._record_thread = None
        self._samples = None
        self._option_items = {}
        self.hotkey = None
        self._preferences_controller = None

        language_menu = self._build_option_submenu("Language", "language", LANGUAGE_LABELS)
        style_menu = self._build_option_submenu("Style", "style", STYLE_LABELS)
        recording_mode_menu = self._build_option_submenu(
            "Recording Mode",
            "recording_mode",
            RECORDING_MODE_LABELS,
            on_change=self._restart_hotkey_listener,
        )
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
            history_menu,
            transcribe_file_item,
            preferences_item,
        ]
        self._refresh_all_checkmarks()

        self._start_hotkey_listener()

    def _show_preferences(self, sender):
        self._preferences_controller = PreferencesWindowController.alloc().init()
        self._preferences_controller.configure(self.config, self._on_preferences_saved)
        self._preferences_controller.show()

    def _on_preferences_saved(self):
        self._refresh_all_checkmarks()
        self._restart_hotkey_listener()

    def _clean_with_fallback(self, raw_text):
        final_text = raw_text
        if self.config.get("cleanup_enabled", True):
            try:
                final_text = clean_transcript(raw_text, self.config)
            except CleanupError as exc:
                rumps.notification(
                    "dictate-mac", "Cleanup failed, using raw transcript", str(exc)
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
            final_text = self._clean_with_fallback(raw_text)

            output_path = Path(input_path).with_suffix(".txt")
            output_path.write_text(final_text)
            subprocess.run(["open", str(output_path)])
            rumps.notification("dictate-mac", "File transcribed", f"Saved to {output_path.name}")

            self._record_history(raw_text, final_text, language)
        except FileTranscribeError as exc:
            rumps.notification("dictate-mac", "File transcription failed", str(exc))
        except Exception as exc:
            rumps.notification("dictate-mac", "File transcription failed", str(exc))

    def _show_history(self, sender):
        entries = load_history()
        if not entries:
            rumps.notification("dictate-mac", "History", "No dictations recorded yet.")
            return

        lines = []
        for entry in reversed(entries):
            timestamp = entry.get("timestamp", "?")
            language = entry.get("language", "?")
            style = entry.get("style", "?")
            lines.append(f"[{timestamp}] ({language}, {style})")
            lines.append(entry.get("final_text", ""))
            lines.append("")

        history_view_path = Path(tempfile.gettempdir()) / "dictate-mac-history.txt"
        history_view_path.write_text("\n".join(lines))
        subprocess.run(["open", str(history_view_path)])

    def _clear_history(self, sender):
        clear_history()
        rumps.notification("dictate-mac", "History", "Dictation history cleared.")

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

    def _start_hotkey_listener(self):
        if self.config.get("recording_mode", "toggle") == "push_to_talk":
            self.hotkey = HotkeyListener(
                self.config["hotkey"],
                on_activate=self._on_hotkey_press,
                on_deactivate=self._on_hotkey_release,
                mode="push_to_talk",
            )
        else:
            self.hotkey = HotkeyListener(self.config["hotkey"], on_activate=self.on_hotkey)
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

    def _start_recording(self):
        self.state = "recording"
        self.title = RECORDING_TITLE
        self._stop_event = threading.Event()
        self._samples = None

        def run():
            try:
                self._samples = record(self._stop_event, SAMPLE_RATE)
            except Exception as exc:
                self._samples = None
                rumps.notification("dictate-mac", "Recording failed", str(exc))
                self.state = "idle"
                self.title = IDLE_TITLE

        self._record_thread = threading.Thread(target=run)
        self._record_thread.start()

    def _stop_recording(self):
        self.state = "processing"
        self.title = PROCESSING_TITLE
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
                    rumps.notification("dictate-mac", "Transcription failed", str(exc))
                    return

            if not raw_text:
                return

            final_text = self._clean_with_fallback(raw_text)

            copy_to_clipboard(final_text)
            paste_into_frontmost_app()

            self._record_history(raw_text, final_text, language)
        except Exception as exc:
            rumps.notification("dictate-mac", "Dictation failed", str(exc))
        finally:
            self.state = "idle"
            self.title = IDLE_TITLE


def main():
    DictateApp().run()


if __name__ == "__main__":
    main()
