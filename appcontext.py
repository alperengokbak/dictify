from AppKit import NSWorkspace

# Only these may be overridden by a profile. Everything else in the config
# (binaries, model paths, hotkey, silence thresholds, history settings) is
# infrastructure - a profile that could repoint whisper_model_path would make
# per-dictation behavior genuinely unpredictable, and nothing about which app
# is frontmost should influence where the model lives. Filtered at resolve
# time rather than at save time so a hand-edited config.json is covered too.
OVERRIDABLE_KEYS = frozenset({"style", "language", "cleanup_enabled"})


def resolve_profile(bundle_id, config: dict) -> dict:
    """Returns the override dict for this app, or {} if no rule matches.
    Pure - no macOS involvement, no I/O."""
    if not bundle_id:
        return {}
    for rule in config.get("app_profiles", []):
        if bundle_id in rule.get("bundle_ids", []):
            return {
                k: v
                for k, v in rule.get("overrides", {}).items()
                if k in OVERRIDABLE_KEYS
            }
    return {}


def effective_config(config: dict, overrides: dict) -> dict:
    """A copy of config with overrides applied. Returns the input object
    itself when there are no overrides, so the no-profiles path is provably
    identical rather than merely equivalent. Never mutates the input - the
    user's stored settings must survive a dictation untouched."""
    if not overrides:
        return config
    merged = dict(config)
    merged.update(overrides)
    return merged
