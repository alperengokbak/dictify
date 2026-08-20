"""Guards against locale-dependent text IO.

Dictify ships as a py2app bundle whose stub embeds libpython directly. An
embedding host does not run the PEP 538 C-locale coercion that the standalone
`python` binary does in its main(), so inside the .app the LC_CTYPE locale
stays "C" and Python's default text encoding is ASCII - regardless of LANG.
Every open()/read_text()/write_text() that relies on the ambient locale
therefore works when launched from a terminal and blows up when launched from
Finder. A single "Gokbak" with an umlaut in the glossary was enough to make
the app die at startup with a py2app "Launch error" dialog.

The fix is to name the encoding explicitly at every text IO site. These tests
re-create the ASCII locale in a child interpreter so a regression fails here
instead of only in a release build.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

# "Alperen Gokbak" as it actually appears in the user's glossary - the exact
# string that triggered the launch failure.
NON_ASCII = "Alperen Gökbak"
TURKISH_SENTENCE = "Bugün hava çok güzel."


def _ascii_locale_env(**extra):
    """Environment that forces the child interpreter to ASCII text IO,
    mirroring what the py2app stub gives the app at runtime."""
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": os.environ.get("HOME", ""),
        "LC_ALL": "C",
        "PYTHONUTF8": "0",
        "PYTHONCOERCECLOCALE": "0",
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.update(extra)
    return env


def _run_under_ascii_locale(snippet, **extra_env):
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        env=_ascii_locale_env(**extra_env),
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert "PREFERRED=US-ASCII" in proc.stdout, (
        "child interpreter was not actually running under an ASCII locale, so "
        "this test would pass vacuously.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.returncode == 0, f"child failed:\n{proc.stderr}"
    return proc.stdout


_PREAMBLE = (
    "import locale, os, sys, json\n"
    "print('PREFERRED=' + locale.getpreferredencoding(False))\n"
    "from pathlib import Path\n"
)


def _codepoints(text):
    return ",".join(str(ord(ch)) for ch in text)


def test_ascii_locale_harness_actually_forces_ascii(tmp_path):
    """Meta-guard: if this stops reproducing, every test below is vacuous."""
    target = tmp_path / "probe.json"
    target.write_bytes(json.dumps({"g": NON_ASCII}).encode("utf-8"))

    out = _run_under_ascii_locale(
        _PREAMBLE
        + "p = Path(os.environ['PROBE'])\n"
        + "try:\n"
        + "    open(p).read()\n"
        + "    print('RESULT=no-error')\n"
        + "except UnicodeDecodeError:\n"
        + "    print('RESULT=unicode-decode-error')\n",
        PROBE=str(target),
    )
    # json.dumps escapes non-ASCII by default, so write real UTF-8 bytes.
    assert "PREFERRED=US-ASCII" in out


def test_load_config_reads_utf8_glossary_under_ascii_locale(tmp_path):
    """The actual launch bug: a non-ASCII glossary entry killed startup."""
    cfg_dir = tmp_path / "dictify"
    cfg_dir.mkdir()
    payload = json.dumps({"glossary": [NON_ASCII]}, ensure_ascii=False)
    (cfg_dir / "config.json").write_bytes(payload.encode("utf-8"))

    out = _run_under_ascii_locale(
        _PREAMBLE
        + "import config\n"
        + "cfg_dir = Path(os.environ['CFG_DIR'])\n"
        + "config.CONFIG_DIR = cfg_dir\n"
        + "config.CONFIG_PATH = cfg_dir / 'config.json'\n"
        + "cfg = config.load_config()\n"
        + "print('CODEPOINTS=' + ','.join(str(ord(c)) for c in cfg['glossary'][0]))\n",
        CFG_DIR=str(cfg_dir),
    )

    assert f"CODEPOINTS={_codepoints(NON_ASCII)}" in out


def test_save_config_roundtrips_non_ascii_under_ascii_locale(tmp_path):
    cfg_dir = tmp_path / "dictify"

    out = _run_under_ascii_locale(
        _PREAMBLE
        + "import config\n"
        + "cfg_dir = Path(os.environ['CFG_DIR'])\n"
        + "config.CONFIG_DIR = cfg_dir\n"
        + "config.CONFIG_PATH = cfg_dir / 'config.json'\n"
        + "cfg = dict(config.DEFAULT_CONFIG)\n"
        + "cfg['glossary'] = [os.environ['WORD']]\n"
        + "config.save_config(cfg)\n"
        + "back = config.load_config()\n"
        + "print('CODEPOINTS=' + ','.join(str(ord(c)) for c in back['glossary'][0]))\n",
        CFG_DIR=str(cfg_dir),
        WORD=NON_ASCII,
    )

    assert f"CODEPOINTS={_codepoints(NON_ASCII)}" in out


def test_parse_whisper_json_reads_turkish_under_ascii_locale(tmp_path):
    """Turkish is a primary dictation language here, so whisper's JSON output
    is routinely non-ASCII - this path breaks on every Turkish transcript."""
    json_path = tmp_path / "out.json"
    payload = json.dumps(
        {
            "result": {"language": "turkish"},
            "transcription": [{"text": TURKISH_SENTENCE}],
        },
        ensure_ascii=False,
    )
    json_path.write_bytes(payload.encode("utf-8"))

    out = _run_under_ascii_locale(
        _PREAMBLE
        + "import transcribe\n"
        + "text, lang = transcribe._parse_whisper_json(os.environ['JSON_PATH'])\n"
        + "print('CODEPOINTS=' + ','.join(str(ord(c)) for c in text))\n",
        JSON_PATH=str(json_path),
    )

    assert f"CODEPOINTS={_codepoints(TURKISH_SENTENCE)}" in out


def test_history_roundtrips_turkish_under_ascii_locale(tmp_path):
    cfg_dir = tmp_path / "dictify"
    cfg_dir.mkdir()

    out = _run_under_ascii_locale(
        _PREAMBLE
        + "import history\n"
        + "history.HISTORY_PATH = Path(os.environ['CFG_DIR']) / 'history.jsonl'\n"
        + "history.append_entry(os.environ['WORD'], os.environ['WORD'], 'tr', 'default')\n"
        + "entries = history.load_history()\n"
        + "print('CODEPOINTS=' + ','.join(str(ord(c)) for c in entries[0]['final_text']))\n",
        CFG_DIR=str(cfg_dir),
        WORD=TURKISH_SENTENCE,
    )

    assert f"CODEPOINTS={_codepoints(TURKISH_SENTENCE)}" in out


# --- Static guard -----------------------------------------------------------
# Covers the sites that cannot be exercised in isolation (the write_text calls
# buried in DictateApp) and stops new locale-dependent IO from creeping in.

PRODUCTION_MODULES = sorted(
    p for p in REPO_ROOT.glob("*.py") if p.name != "setup.py"
)

_BINARY_MODES = {"rb", "wb", "ab", "r+b", "w+b", "rb+", "wb+", "ab+", "xb"}


def _has_encoding_kwarg(call):
    return any(kw.arg == "encoding" for kw in call.keywords)


def _locale_dependent_calls(tree):
    """Yields (lineno, description) for text IO that inherits the locale."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        if isinstance(func, ast.Name) and func.id == "open":
            mode = ""
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = node.args[1].value or ""
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = kw.value.value or ""
            if mode in _BINARY_MODES:
                continue
            if not _has_encoding_kwarg(node):
                yield node.lineno, f"open(..., mode={mode!r}) without encoding="

        elif isinstance(func, ast.Attribute) and func.attr in {
            "read_text",
            "write_text",
        }:
            if not _has_encoding_kwarg(node):
                yield node.lineno, f".{func.attr}() without encoding="


def test_no_locale_dependent_text_io_in_production_modules():
    offenders = []
    for module in PRODUCTION_MODULES:
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for lineno, description in _locale_dependent_calls(tree):
            offenders.append(f"{module.name}:{lineno}: {description}")

    assert not offenders, (
        "Text IO here inherits the ambient locale, which is ASCII inside the "
        "py2app bundle. Pass encoding='utf-8' explicitly:\n  "
        + "\n  ".join(offenders)
    )
