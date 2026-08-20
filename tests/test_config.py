import config as config_module


def test_load_defaults_when_missing(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "dictify"
    monkeypatch.setattr(config_module, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_dir / "config.json")

    cfg = config_module.load_config()

    assert cfg == config_module.DEFAULT_CONFIG
    assert (cfg_dir / "config.json").exists()


def test_save_and_reload_roundtrip(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "dictify"
    monkeypatch.setattr(config_module, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_dir / "config.json")

    custom = dict(config_module.DEFAULT_CONFIG)
    custom["hotkey"] = "<cmd>+<space>"
    config_module.save_config(custom)

    reloaded = config_module.load_config()

    assert reloaded["hotkey"] == "<cmd>+<space>"


def test_malformed_json_falls_back_to_defaults(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "dictify"
    cfg_dir.mkdir(parents=True)
    cfg_path = cfg_dir / "config.json"
    cfg_path.write_text("{not valid json")
    monkeypatch.setattr(config_module, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_path)

    cfg = config_module.load_config()

    assert cfg == config_module.DEFAULT_CONFIG


def test_default_config_has_sound_feedback_enabled():
    assert config_module.DEFAULT_CONFIG["sound_feedback_enabled"] is True


def test_default_config_includes_whisper_server_binary():
    assert "whisper_server_binary" in config_module.DEFAULT_CONFIG
    assert config_module.DEFAULT_CONFIG["whisper_server_binary"].endswith("whisper-server")


def test_undecodable_config_falls_back_to_defaults(tmp_path, monkeypatch):
    """load_config()'s contract is that an unreadable config never stops the
    app from starting - it falls back to defaults. A config file that isn't
    valid UTF-8 raises UnicodeDecodeError, which is a ValueError and so slips
    past an `except (json.JSONDecodeError, OSError)`. Left uncaught it kills
    the app during __init__, which surfaces as a bare py2app "Launch error"
    dialog with no clue what went wrong."""
    cfg_dir = tmp_path / "dictify"
    cfg_dir.mkdir(parents=True)
    cfg_path = cfg_dir / "config.json"
    # "Gokbak" with the umlaut encoded as latin-1 - not decodable as UTF-8.
    cfg_path.write_bytes(b'{"glossary": ["Alperen G\xf6kbak"]}')
    monkeypatch.setattr(config_module, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_path)

    cfg = config_module.load_config()

    assert cfg == config_module.DEFAULT_CONFIG
