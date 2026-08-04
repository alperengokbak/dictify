import history as history_module


def test_append_and_load_roundtrip(tmp_path, monkeypatch):
    hist_path = tmp_path / "history.jsonl"
    monkeypatch.setattr(history_module, "HISTORY_PATH", hist_path)

    history_module.append_entry("raw", "cleaned", "en", "default")

    entries = history_module.load_history()
    assert len(entries) == 1
    assert entries[0]["raw_text"] == "raw"
    assert entries[0]["final_text"] == "cleaned"
    assert entries[0]["language"] == "en"
    assert entries[0]["style"] == "default"
    assert "timestamp" in entries[0]


def test_load_history_returns_empty_list_when_file_missing(tmp_path, monkeypatch):
    hist_path = tmp_path / "history.jsonl"
    monkeypatch.setattr(history_module, "HISTORY_PATH", hist_path)

    assert history_module.load_history() == []


def test_load_history_skips_malformed_lines(tmp_path, monkeypatch):
    hist_path = tmp_path / "history.jsonl"
    hist_path.write_text('{"raw_text": "ok"}\nnot json\n{"raw_text": "ok2"}\n')
    monkeypatch.setattr(history_module, "HISTORY_PATH", hist_path)

    entries = history_module.load_history()
    assert len(entries) == 2


def test_append_entry_trims_to_limit(tmp_path, monkeypatch):
    hist_path = tmp_path / "history.jsonl"
    monkeypatch.setattr(history_module, "HISTORY_PATH", hist_path)

    for i in range(5):
        history_module.append_entry(f"raw{i}", f"cleaned{i}", "en", "default", limit=3)

    entries = history_module.load_history()
    assert len(entries) == 3
    assert [e["raw_text"] for e in entries] == ["raw2", "raw3", "raw4"]


def test_clear_history_removes_all_entries(tmp_path, monkeypatch):
    hist_path = tmp_path / "history.jsonl"
    monkeypatch.setattr(history_module, "HISTORY_PATH", hist_path)
    history_module.append_entry("raw", "cleaned", "en", "default")

    history_module.clear_history()

    assert history_module.load_history() == []
