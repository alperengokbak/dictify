"""Owns the whisper-server child process lifecycle: starting it lazily,
health-checking it, resetting/firing an idle timer, and stopping it.
Nothing outside this module manages the child process directly - callers
(transcribe.py) only ever call ensure_running()/stop()."""

import os
import signal
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


class _AdoptedProcess:
    """Duck-types subprocess.Popen's poll()/terminate()/kill()/wait() just
    well enough that stop() can treat an adopted orphan (a whisper-server we
    found already listening on the port, but never spawned ourselves - see
    the orphan-adoption branch in ensure_running()) the same as a child we
    did spawn. Without this, stop() would have no handle to terminate an
    adopted process by, since subprocess.Popen() was never called for it."""

    def __init__(self, pid: int):
        self.pid = pid

    def poll(self):
        try:
            os.kill(self.pid, 0)
        except OSError:
            return 0  # no such process - it's gone
        return None  # still alive

    def terminate(self):
        try:
            os.kill(self.pid, signal.SIGTERM)
        except OSError:
            pass

    def kill(self):
        try:
            os.kill(self.pid, signal.SIGKILL)
        except OSError:
            pass

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(cmd=f"pid {self.pid}", timeout=timeout)
            time.sleep(0.05)
        return 0


_process = None            # subprocess.Popen | _AdoptedProcess | None
_idle_timer = None         # threading.Timer | None
_lock = threading.RLock()  # guards _process/_idle_timer/_generation; reentrant
                           # because ensure_running()'s failure path calls stop()
                           # while already holding the lock - a plain Lock would
                           # deadlock
_last_failure_time = None  # float | None - set on a start failure, cleared on success
_generation = 0            # int - bumped every time _reset_idle_timer() issues a new
                           # Timer. Timer.cancel() is a documented no-op once the
                           # timer's thread has already started running its target,
                           # so a stale timer that fired just before a renewal can
                           # still reach _on_idle_timeout() after the renewal
                           # happened; comparing generations there lets it detect
                           # it's stale and skip stopping the (renewed) process
                           # instead of killing it out from under a caller that was
                           # just handed BASE_URL.


def _is_healthy() -> bool:
    try:
        resp = requests.get(f"{BASE_URL}/", timeout=1)
        return resp.status_code < 500
    except requests.RequestException:
        return False


def _find_listening_pid() -> "int | None":
    """Best-effort PID discovery for a whisper-server we didn't spawn
    ourselves (the orphan-adoption path in ensure_running()). Uses lsof
    (standard on macOS) rather than a third-party dependency. If lsof is
    unavailable or nothing is found, adoption still succeeds - we just
    won't have a way to stop that particular process later."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{PORT}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    pids = result.stdout.split()
    if not pids:
        return None
    try:
        return int(pids[0])
    except ValueError:
        return None


def _on_idle_timeout(gen: int):
    """Timer callback - runs on the Timer's own thread. Only stop() if we're
    still the most recent generation once we actually hold the lock; see the
    _generation comment above for why this check exists."""
    with _lock:
        if gen == _generation:
            stop()


def _reset_idle_timer():
    global _idle_timer, _generation
    if _idle_timer is not None:
        _idle_timer.cancel()
    _generation += 1
    gen = _generation
    _idle_timer = threading.Timer(IDLE_TIMEOUT_SECS, _on_idle_timeout, args=(gen,))
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
        # would just fail to bind. Try to discover its PID so stop() (idle
        # timer, quit hook, model change) can actually end it later - if that
        # lookup fails, adoption still proceeds, we just can't stop it.
        if _process is None and _is_healthy():
            pid = _find_listening_pid()
            if pid is not None:
                _process = _AdoptedProcess(pid)
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
