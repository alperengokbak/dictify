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
