"""Owns the whisper-server child process lifecycle: starting it lazily,
health-checking it, resetting/firing an idle timer, and stopping it.
Nothing outside this module manages the child process directly - callers
(transcribe.py) only ever call ensure_running()/stop()."""

import subprocess
import threading
import time

import requests

HOST = "127.0.0.1"
PORT = 8090
BASE_URL = f"http://{HOST}:{PORT}"
IDLE_TIMEOUT_SECS = 600  # 10 minutes
STARTUP_TIMEOUT_SECS = 10
FAILURE_COOLDOWN_SECS = 300  # 5 minutes


class WhisperServerError(Exception):
    pass


_process = None            # subprocess.Popen | None
_idle_timer = None         # threading.Timer | None
_lock = threading.RLock()  # guards _process/_idle_timer; reentrant because
                           # ensure_running()'s failure path calls stop() while
                           # already holding the lock - a plain Lock would deadlock
_last_failure_time = None  # float | None - set on a start failure, cleared on success


def _is_healthy() -> bool:
    try:
        resp = requests.get(f"{BASE_URL}/", timeout=1)
        return resp.status_code < 500
    except requests.RequestException:
        return False


def _reset_idle_timer():
    global _idle_timer
    if _idle_timer is not None:
        _idle_timer.cancel()
    _idle_timer = threading.Timer(IDLE_TIMEOUT_SECS, stop)
    _idle_timer.daemon = True
    _idle_timer.start()


def ensure_running(config: dict) -> str:
    """Returns BASE_URL once whisper-server is confirmed healthy. Raises
    WhisperServerError if it can't be started/reached - callers are
    expected to fall back to the whisper-cli subprocess path on this."""
    global _process, _last_failure_time
    with _lock:
        if _last_failure_time is not None:
            if time.monotonic() - _last_failure_time < FAILURE_COOLDOWN_SECS:
                raise WhisperServerError("whisper-server in failure cooldown")
            # cooldown elapsed - fall through and retry once

        if _process is not None and _process.poll() is None and _is_healthy():
            _reset_idle_timer()
            return BASE_URL

        # Not running under our management - check whether something (e.g. an
        # orphaned instance from a crashed previous run) is already listening
        # on the port and healthy; adopt it instead of double-spawning, which
        # would just fail to bind.
        if _process is None and _is_healthy():
            _reset_idle_timer()
            return BASE_URL

        binary = config["whisper_server_binary"]
        try:
            _process = subprocess.Popen(
                [binary, "-m", config["whisper_model_path"], "--host", HOST, "--port", str(PORT)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            _last_failure_time = time.monotonic()
            raise WhisperServerError(f"failed to launch whisper-server: {exc}") from exc

        deadline = time.monotonic() + STARTUP_TIMEOUT_SECS
        while time.monotonic() < deadline:
            if _is_healthy():
                _last_failure_time = None
                _reset_idle_timer()
                return BASE_URL
            if _process.poll() is not None:
                break  # died during startup - stop polling, fall through to failure
            time.sleep(0.2)

        _last_failure_time = time.monotonic()
        stop()
        raise WhisperServerError("whisper-server did not become healthy in time")


def stop():
    """Terminates the managed child, if any. Safe to call when nothing is
    running (idempotent) - used by the idle timer, the quit hook, and a
    model-change in Preferences."""
    global _process, _idle_timer
    with _lock:
        if _idle_timer is not None:
            _idle_timer.cancel()
            _idle_timer = None
        if _process is not None:
            _process.terminate()
            try:
                _process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _process.kill()
                _process.wait(timeout=5)
            _process = None
