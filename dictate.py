import tempfile
import threading
from pathlib import Path

import rumps

from audio import is_silent, record, save_wav
from cleanup import CleanupError, clean_transcript
from config import load_config
from hotkey import HotkeyListener
from paste import copy_to_clipboard, paste_into_frontmost_app
from transcribe import TranscribeError, transcribe

IDLE_TITLE = "🎙"
RECORDING_TITLE = "🔴"
PROCESSING_TITLE = "⏳"
SAMPLE_RATE = 16000


class DictateApp(rumps.App):
    def __init__(self):
        super().__init__("dictate-mac", title=IDLE_TITLE)
        self.config = load_config()
        self.state = "idle"
        self._stop_event = None
        self._record_thread = None
        self._samples = None
        self.hotkey = HotkeyListener(self.config["hotkey"], self.on_hotkey)
        self.hotkey.start()

    def on_hotkey(self):
        if self.state == "idle":
            self._start_recording()
        elif self.state == "recording":
            self._stop_recording()
        # "processing": ignored, recording/processing never queues

    def _start_recording(self):
        self.state = "recording"
        self.title = RECORDING_TITLE
        self._stop_event = threading.Event()
        self._samples = None

        def run():
            self._samples = record(self._stop_event, SAMPLE_RATE)

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
                    raw_text, _language = transcribe(wav_path, self.config)
                except TranscribeError as exc:
                    rumps.notification("dictate-mac", "Transcription failed", str(exc))
                    return

            if not raw_text:
                return

            final_text = raw_text
            if self.config.get("cleanup_enabled", True):
                try:
                    final_text = clean_transcript(raw_text, self.config)
                except CleanupError as exc:
                    rumps.notification(
                        "dictate-mac", "Cleanup failed, pasted raw transcript", str(exc)
                    )
                    final_text = raw_text

            copy_to_clipboard(final_text)
            paste_into_frontmost_app()
        finally:
            self.state = "idle"
            self.title = IDLE_TITLE


def main():
    DictateApp().run()


if __name__ == "__main__":
    main()
