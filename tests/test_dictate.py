import fcntl
import os
from unittest.mock import MagicMock

import rumps

import appcontext
import dictate


def test_cancel_combos_normal_record_hotkey_yields_two_distinct_combos():
    combos = dictate._cancel_combos("<ctrl>+<alt>+<cmd>+<d>")
    assert set(combos) == {"<escape>", "<ctrl>+<alt>+<cmd>+<escape>"}


def test_cancel_combos_record_hotkey_with_no_modifiers_yields_one_combo():
    assert dictate._cancel_combos("<d>") == ["<escape>"]


def test_cancel_combos_record_hotkey_is_escape_skips_colliding_combo():
    # The record hotkey's own combo (<ctrl>+<alt>+<escape>) would be
    # registered twice if not excluded - once by the record listener,
    # once again as a cancel candidate.
    assert dictate._cancel_combos("<ctrl>+<alt>+<escape>") == ["<escape>"]


def test_cancel_combos_record_hotkey_is_bare_escape_yields_no_combos():
    # Degenerate case: the record hotkey IS bare Escape, so every cancel
    # candidate collides with it. No cancel hotkey can be offered.
    assert dictate._cancel_combos("<escape>") == []


def test_acquire_singleton_lock_succeeds_when_no_lock_file(tmp_path):
    lock_path = tmp_path / "dictify.lock"
    assert dictate._acquire_singleton_lock(lock_path) is True
    assert lock_path.read_text().strip() == str(os.getpid())


def test_acquire_singleton_lock_fails_when_another_process_holds_it(tmp_path):
    lock_path = tmp_path / "dictify.lock"
    # Simulate another live instance by holding a real flock via a
    # separate file descriptor - real OS-level locking behavior, no mocks.
    holder_fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY)
    fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert dictate._acquire_singleton_lock(lock_path) is False
    finally:
        os.close(holder_fd)


def test_acquire_singleton_lock_leaves_existing_content_untouched_on_failure(tmp_path):
    lock_path = tmp_path / "dictify.lock"
    holder_fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY)
    os.write(holder_fd, b"12345")
    fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert dictate._acquire_singleton_lock(lock_path) is False
        assert lock_path.read_text() == "12345"
    finally:
        os.close(holder_fd)


def test_acquire_singleton_lock_succeeds_after_holder_releases(tmp_path):
    lock_path = tmp_path / "dictify.lock"
    holder_fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY)
    fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(holder_fd)  # releases the lock, simulating the holder process exiting
    assert dictate._acquire_singleton_lock(lock_path) is True


def _bare_app(config=None):
    """Builds a DictateApp instance without running its __init__ (which
    starts a real global hotkey listener and other live side effects) -
    same bypass-construction technique tests/test_preferences.py uses for
    PreferencesWindowController, adapted for a plain-Python App subclass."""
    app = dictate.DictateApp.__new__(dictate.DictateApp)
    app.config = config if config is not None else {}
    return app


def test_play_start_sound_plays_when_enabled(monkeypatch):
    played = []
    monkeypatch.setattr(dictate, "play_sound", lambda name: played.append(name))
    app = _bare_app({"sound_feedback_enabled": True})

    app._play_start_sound()

    assert played == [dictate.START_SOUND]


def test_play_start_sound_silent_when_disabled(monkeypatch):
    played = []
    monkeypatch.setattr(dictate, "play_sound", lambda name: played.append(name))
    app = _bare_app({"sound_feedback_enabled": False})

    app._play_start_sound()

    assert played == []


def test_play_stop_sound_plays_when_enabled(monkeypatch):
    played = []
    monkeypatch.setattr(dictate, "play_sound", lambda name: played.append(name))
    app = _bare_app({"sound_feedback_enabled": True})

    app._play_stop_sound()

    assert played == [dictate.STOP_SOUND]


def test_play_stop_sound_defaults_to_enabled_when_key_missing(monkeypatch):
    played = []
    monkeypatch.setattr(dictate, "play_sound", lambda name: played.append(name))
    app = _bare_app({})  # no "sound_feedback_enabled" key at all

    app._play_stop_sound()

    assert played == [dictate.STOP_SOUND]


def _bare_app_with_last_transcript_item():
    app = _bare_app()
    app._last_transcript_last_text = None
    app._last_transcript_item = rumps.MenuItem("Last: (none yet)", callback=None)
    return app


def test_format_last_transcript_label_short_text_passes_through():
    assert dictate._format_last_transcript_label("hello world") == "Last: hello world"


def test_format_last_transcript_label_truncates_long_text():
    text = "x" * 60
    label = dictate._format_last_transcript_label(text, limit=50)
    assert label == "Last: " + "x" * 50 + "…"


def test_format_last_transcript_label_collapses_whitespace():
    text = "line one\nline two\n\nline three"
    assert dictate._format_last_transcript_label(text) == "Last: line one line two line three"


def test_update_last_transcript_item_sets_title_text_and_enables():
    app = _bare_app_with_last_transcript_item()

    app._update_last_transcript_item("hello world")

    assert app._last_transcript_last_text == "hello world"
    assert str(app._last_transcript_item.title) == "Last: hello world"
    assert app._last_transcript_item.callback == app._copy_last_transcript


def test_seed_last_transcript_item_from_existing_history(monkeypatch):
    monkeypatch.setattr(
        dictate, "load_history",
        lambda: [
            {"final_text": "first entry"},
            {"final_text": "most recent entry"},
        ],
    )
    app = _bare_app_with_last_transcript_item()

    app._seed_last_transcript_item()

    assert app._last_transcript_last_text == "most recent entry"
    assert str(app._last_transcript_item.title) == "Last: most recent entry"


def test_seed_last_transcript_item_empty_history_leaves_placeholder(monkeypatch):
    monkeypatch.setattr(dictate, "load_history", lambda: [])
    app = _bare_app_with_last_transcript_item()

    app._seed_last_transcript_item()

    assert app._last_transcript_last_text is None
    assert str(app._last_transcript_item.title) == "Last: (none yet)"
    assert app._last_transcript_item.callback is None


def test_seed_last_transcript_item_history_load_failure_leaves_placeholder(monkeypatch):
    def _raise():
        raise OSError("permission denied")

    monkeypatch.setattr(dictate, "load_history", _raise)
    app = _bare_app_with_last_transcript_item()

    app._seed_last_transcript_item()  # must not raise

    assert app._last_transcript_last_text is None
    assert str(app._last_transcript_item.title) == "Last: (none yet)"


def test_clear_history_resets_the_last_transcript_item(monkeypatch):
    # Regression test: clicking "Clear History" deleted history.jsonl but
    # left the "Last: ..." menu item (and its clipboard-copy-on-click
    # behavior) showing the old text, since that's separate in-memory
    # state clear_history() alone never touched.
    monkeypatch.setattr(dictate, "clear_history", lambda: None)
    monkeypatch.setattr(dictate.rumps, "notification", lambda *a, **kw: None)
    app = _bare_app_with_last_transcript_item()
    app._update_last_transcript_item("some old dictation")

    app._clear_history(None)

    assert app._last_transcript_last_text is None
    assert str(app._last_transcript_item.title) == "Last: (none yet)"
    assert app._last_transcript_item.callback is None


def test_copy_last_transcript_copies_full_text(monkeypatch):
    copied = []
    monkeypatch.setattr(dictate, "copy_to_clipboard", lambda text: copied.append(text))
    app = _bare_app_with_last_transcript_item()
    app._last_transcript_last_text = "the full dictated text"

    app._copy_last_transcript(None)

    assert copied == ["the full dictated text"]


def test_stop_whisper_server_if_model_changed_calls_stop_when_path_differs(monkeypatch):
    stopped = []
    monkeypatch.setattr(dictate.whisper_server, "stop", lambda: stopped.append(True))
    app = _bare_app({"whisper_model_path": "/new/path.bin"})
    app._whisper_model_path_before_edit = "/old/path.bin"

    app._stop_whisper_server_if_model_changed()

    assert stopped == [True]


def test_stop_whisper_server_if_model_changed_no_op_when_path_unchanged(monkeypatch):
    stopped = []
    monkeypatch.setattr(dictate.whisper_server, "stop", lambda: stopped.append(True))
    app = _bare_app({"whisper_model_path": "/same/path.bin"})
    app._whisper_model_path_before_edit = "/same/path.bin"

    app._stop_whisper_server_if_model_changed()

    assert stopped == []


def test_on_quit_stops_whisper_server(monkeypatch):
    stopped = []
    monkeypatch.setattr(dictate.whisper_server, "stop", lambda: stopped.append(True))
    app = _bare_app()

    app._on_quit()

    assert stopped == [True]


def _real_app(monkeypatch):
    """Constructs a genuine DictateApp - __init__ and all - with only the
    live OS side effects (config file I/O, the global hotkey listener, the
    waveform window, history reads) stubbed out. Needed for the wiring that
    only happens inside __init__, which the _bare_app bypass skips entirely."""
    monkeypatch.setattr(dictate, "load_config", lambda: dict(dictate.DEFAULT_CONFIG))
    monkeypatch.setattr(dictate, "load_history", lambda: [])
    monkeypatch.setattr(dictate, "WaveformWindowController", lambda: None)
    monkeypatch.setattr(dictate.DictateApp, "_start_hotkey_listener", lambda self: None)
    return dictate.DictateApp()


def test_init_registers_on_quit_as_a_before_quit_callback(monkeypatch):
    """The quit hook is what stops the whisper-server child on app exit -
    without the registration in __init__, _on_quit is never called and a
    1.5GB+ process outlives the app. _bare_app skips __init__, so nothing
    else in this file would notice the registration disappearing."""
    app = _real_app(monkeypatch)
    try:
        assert app._on_quit in rumps.events.before_quit.callbacks
    finally:
        # Module-level registry: leaving it registered would leak into other tests.
        rumps.events.before_quit.unregister(app._on_quit)


def test_show_preferences_snapshots_the_model_path_so_an_unchanged_save_is_a_no_op(
    monkeypatch,
):
    """Regression test for the snapshot line in _show_preferences: without
    it, _whisper_model_path_before_edit is never set, so the getattr default
    of None never equals the configured path and EVERY Preferences save
    would needlessly stop (and cold-restart) the server."""
    stopped = []
    monkeypatch.setattr(dictate.whisper_server, "stop", lambda: stopped.append(True))
    monkeypatch.setattr(dictate, "PreferencesWindowController", MagicMock())
    app = _bare_app({"whisper_model_path": "/models/ggml-medium.bin"})

    app._show_preferences(None)
    app._stop_whisper_server_if_model_changed()  # user saved without touching the model

    assert stopped == []


def test_stop_whisper_server_if_model_changed_stops_when_no_snapshot_was_taken(monkeypatch):
    # Documents the behavior the test above guards against: with no snapshot
    # attribute at all, the comparison is against None and the server is
    # always stopped. Correct as a fail-safe, wrong as an everyday path.
    stopped = []
    monkeypatch.setattr(dictate.whisper_server, "stop", lambda: stopped.append(True))
    app = _bare_app({"whisper_model_path": "/models/ggml-medium.bin"})
    assert not hasattr(app, "_whisper_model_path_before_edit")

    app._stop_whisper_server_if_model_changed()

    assert stopped == [True]


def test_clean_with_fallback_forwards_detected_language_to_clean_transcript(monkeypatch):
    # Regression: Whisper detects the spoken language correctly (confirmed
    # live via history.jsonl, both entries tagged "en"), but that fact was
    # never passed to the cleanup step, leaving the small local cleanup
    # model to infer language from text alone - it drifted into Turkish
    # on long/rambling English input despite the "DO NOT TRANSLATE"
    # instruction. The detected language must reach clean_transcript.
    received = {}

    def _fake_clean_transcript(raw_text, config, language=None):
        received["language"] = language
        return raw_text

    monkeypatch.setattr(dictate, "clean_transcript", _fake_clean_transcript)
    app = _bare_app({"cleanup_enabled": True})

    app._clean_with_fallback("some text", app.config, language="en")

    assert received["language"] == "en"


def test_clean_with_fallback_honors_the_config_passed_to_it(monkeypatch):
    # cleanup_enabled differs between self.config and the passed config, so
    # which one is consulted is observable.
    monkeypatch.setattr(dictate, "clean_transcript", lambda *a, **kw: "CLEANED")
    app = _bare_app({"cleanup_enabled": True})

    assert app._clean_with_fallback("raw", {"cleanup_enabled": False}) == "raw"


def test_clean_with_fallback_cleans_when_passed_config_enables_it(monkeypatch):
    monkeypatch.setattr(dictate, "clean_transcript", lambda *a, **kw: "CLEANED")
    app = _bare_app({"cleanup_enabled": False})

    assert app._clean_with_fallback("raw", {"cleanup_enabled": True}) == "CLEANED"


def test_record_history_logs_the_effective_style(monkeypatch):
    # A history entry claiming "professional" for text a Terminal rule passed
    # through verbatim would misreport what actually happened.
    logged = []
    monkeypatch.setattr(
        dictate,
        "append_entry",
        lambda raw, final, language, style, limit=None: logged.append(style),
    )
    app = _bare_app({"style": "professional", "history_enabled": True})

    app._record_history(
        "raw", "final", "en",
        {"style": "casual", "history_enabled": True, "history_limit": 200},
    )

    assert logged == ["casual"]


def test_resolve_effective_config_applies_the_frontmost_apps_profile(monkeypatch):
    monkeypatch.setattr(appcontext, "frontmost_bundle_id", lambda: "com.apple.Terminal")
    app = _bare_app({
        "cleanup_enabled": True,
        "app_profiles": [
            {"bundle_ids": ["com.apple.Terminal"], "overrides": {"cleanup_enabled": False}}
        ],
    })

    resolved = app._resolve_effective_config()

    assert resolved["cleanup_enabled"] is False
    assert app.config["cleanup_enabled"] is True  # stored config left untouched


def test_resolve_effective_config_returns_stored_config_when_no_profiles(monkeypatch):
    # The regression guard: with app_profiles empty, the pipeline must receive
    # the stored config object itself, not a copy that merely equals it.
    monkeypatch.setattr(appcontext, "frontmost_bundle_id", lambda: "com.apple.Terminal")
    app = _bare_app({"cleanup_enabled": True, "app_profiles": []})

    assert app._resolve_effective_config() is app.config


def test_resolve_effective_config_falls_back_when_lookup_fails(monkeypatch):
    monkeypatch.setattr(appcontext, "frontmost_bundle_id", lambda: None)
    app = _bare_app({
        "cleanup_enabled": True,
        "app_profiles": [
            {"bundle_ids": ["com.apple.Terminal"], "overrides": {"cleanup_enabled": False}}
        ],
    })

    assert app._resolve_effective_config() is app.config
