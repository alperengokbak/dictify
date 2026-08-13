from unittest.mock import MagicMock

import appcontext


def _config(profiles):
    return {"app_profiles": profiles, "style": "default", "language": "auto", "cleanup_enabled": True}


def test_resolve_profile_returns_empty_when_no_profiles_configured():
    assert appcontext.resolve_profile("com.apple.Terminal", _config([])) == {}


def test_resolve_profile_returns_overrides_for_matching_bundle_id():
    cfg = _config([{"bundle_ids": ["com.apple.Terminal"], "overrides": {"cleanup_enabled": False}}])
    assert appcontext.resolve_profile("com.apple.Terminal", cfg) == {"cleanup_enabled": False}


def test_resolve_profile_returns_empty_for_unmatched_bundle_id():
    cfg = _config([{"bundle_ids": ["com.apple.Terminal"], "overrides": {"cleanup_enabled": False}}])
    assert appcontext.resolve_profile("com.apple.mail", cfg) == {}


def test_resolve_profile_returns_empty_for_none_bundle_id():
    cfg = _config([{"bundle_ids": ["com.apple.Terminal"], "overrides": {"cleanup_enabled": False}}])
    assert appcontext.resolve_profile(None, cfg) == {}


def test_resolve_profile_matches_any_id_in_a_multi_app_rule():
    cfg = _config([{
        "bundle_ids": ["com.apple.Terminal", "com.googlecode.iterm2"],
        "overrides": {"cleanup_enabled": False},
    }])
    assert appcontext.resolve_profile("com.googlecode.iterm2", cfg) == {"cleanup_enabled": False}


def test_resolve_profile_first_matching_rule_wins():
    cfg = _config([
        {"bundle_ids": ["com.apple.mail"], "overrides": {"style": "professional"}},
        {"bundle_ids": ["com.apple.mail"], "overrides": {"style": "casual"}},
    ])
    assert appcontext.resolve_profile("com.apple.mail", cfg) == {"style": "professional"}


def test_resolve_profile_drops_keys_outside_the_allowlist():
    # A hand-edited config.json must not be able to repoint infrastructure
    # settings per-app. The allowlisted sibling in the same rule still applies.
    cfg = _config([{
        "bundle_ids": ["com.apple.Terminal"],
        "overrides": {"style": "casual", "whisper_model_path": "/tmp/evil"},
    }])
    assert appcontext.resolve_profile("com.apple.Terminal", cfg) == {"style": "casual"}


def test_resolve_profile_tolerates_rule_missing_bundle_ids():
    cfg = _config([{"overrides": {"style": "casual"}}])
    assert appcontext.resolve_profile("com.apple.Terminal", cfg) == {}


def test_resolve_profile_tolerates_rule_missing_overrides():
    cfg = _config([{"bundle_ids": ["com.apple.Terminal"]}])
    assert appcontext.resolve_profile("com.apple.Terminal", cfg) == {}


def test_resolve_profile_tolerates_missing_app_profiles_key():
    # Upgrading from a version before this feature: config.json has no such key.
    assert appcontext.resolve_profile("com.apple.Terminal", {"style": "default"}) == {}


def test_effective_config_returns_the_same_object_when_no_overrides():
    cfg = _config([])
    assert appcontext.effective_config(cfg, {}) is cfg


def test_effective_config_applies_overrides_in_a_copy():
    cfg = _config([])
    result = appcontext.effective_config(cfg, {"style": "casual"})
    assert result["style"] == "casual"
    assert result is not cfg


def test_effective_config_never_mutates_the_input():
    cfg = _config([])
    appcontext.effective_config(cfg, {"style": "casual"})
    assert cfg["style"] == "default"


def _workspace_returning(app):
    """Builds a stand-in for NSWorkspace whose sharedWorkspace().frontmostApplication()
    yields `app`. Mirrors the injectable-transport style used in whisper_models.py."""
    workspace = MagicMock()
    workspace.sharedWorkspace.return_value.frontmostApplication.return_value = app
    return workspace


def test_frontmost_bundle_id_returns_the_identifier(monkeypatch):
    app = MagicMock()
    app.bundleIdentifier.return_value = "com.apple.Terminal"
    monkeypatch.setattr(appcontext, "NSWorkspace", _workspace_returning(app))

    assert appcontext.frontmost_bundle_id() == "com.apple.Terminal"


def test_frontmost_bundle_id_returns_none_when_no_frontmost_app(monkeypatch):
    # Happens during login/logout transitions and some Spaces switches.
    monkeypatch.setattr(appcontext, "NSWorkspace", _workspace_returning(None))

    assert appcontext.frontmost_bundle_id() is None


def test_frontmost_bundle_id_returns_none_when_identifier_is_none(monkeypatch):
    app = MagicMock()
    app.bundleIdentifier.return_value = None
    monkeypatch.setattr(appcontext, "NSWorkspace", _workspace_returning(app))

    assert appcontext.frontmost_bundle_id() is None


def test_frontmost_bundle_id_returns_none_when_identifier_is_empty(monkeypatch):
    app = MagicMock()
    app.bundleIdentifier.return_value = ""
    monkeypatch.setattr(appcontext, "NSWorkspace", _workspace_returning(app))

    assert appcontext.frontmost_bundle_id() is None


def test_frontmost_bundle_id_swallows_pyobjc_errors(monkeypatch):
    # A cosmetic lookup must never be able to prevent a dictation.
    workspace = MagicMock()
    workspace.sharedWorkspace.side_effect = RuntimeError("PyObjC exploded")
    monkeypatch.setattr(appcontext, "NSWorkspace", workspace)

    assert appcontext.frontmost_bundle_id() is None
