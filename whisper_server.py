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
HEALTH_CHECK_TIMEOUT_SECS = 3  # generous on purpose: a false negative here
                               # costs a whole process restart (and a doomed
                               # double model load), a slow check costs nothing
SERVER_PROCESS_NAME = "whisper-server"


class WhisperServerError(Exception):
    pass


def _is_whisper_server_pid(pid: int) -> bool:
    """Confirms a PID really belongs to a whisper-server. Nothing is ever
    adopted or signalled without this: the port alone proves nothing, so
    without an identity check any unrelated process that happened to bind
    8090 would get adopted and then SIGTERM'd by our idle timer."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    comm = result.stdout.strip()
    return bool(comm) and os.path.basename(comm) == SERVER_PROCESS_NAME


def _can_signal(pid: int) -> bool:
    """A process we're not allowed to signal (another user's) can't be
    stopped by us either. Adopting one would hand stop() a handle it can
    never honour, so it would spin out both of its wait() timeouts - a
    10-second hang on the quit hook - and still leave the process running."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


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
        except ProcessLookupError:
            return 0  # no such process - it's gone
        except OSError:
            # It exists but we can't signal it (e.g. PermissionError - it
            # belongs to another user). "Can't tell" is not "it's gone";
            # reporting it as exited would let a caller assume a clean slate.
            return None
        return None  # still alive

    def _signal(self, sig):
        # Re-verify identity on every signal, not just at adoption: PIDs get
        # recycled, so a pid adopted ten minutes ago may belong to something
        # unrelated by now. Never signal what we can't positively confirm.
        if not _is_whisper_server_pid(self.pid):
            return
        try:
            os.kill(self.pid, sig)
        except OSError:
            pass

    def terminate(self):
        self._signal(signal.SIGTERM)

    def kill(self):
        self._signal(signal.SIGKILL)

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
        resp = requests.get(f"{BASE_URL}/", timeout=HEALTH_CHECK_TIMEOUT_SECS)
        return resp.status_code < 500
    except requests.RequestException:
        return False


def _find_listening_pid() -> "int | None":
    """Best-effort PID discovery for a whisper-server we didn't spawn
    ourselves (the orphan-adoption path in ensure_running()). Uses lsof
    (standard on macOS) rather than a third-party dependency. Returning a
    PID does NOT mean it's a whisper-server - callers must run it through
    _is_whisper_server_pid() before adopting or signalling it."""
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
        # would just fail to bind. Adoption hands stop() (idle timer, quit
        # hook, model change) a licence to SIGTERM that PID, so it happens
        # ONLY when the PID is positively confirmed to be a whisper-server.
        # If we can't confirm, we deliberately fall through to the spawn
        # below, which fails to bind and raises WhisperServerError - the
        # already-handled path that degrades to the whisper-cli fallback.
        if _process is None and _is_healthy():
            pid = _find_listening_pid()
            if pid is not None and _is_whisper_server_pid(pid) and _can_signal(pid):
                _process = _AdoptedProcess(pid)
                _reset_idle_timer()
                return BASE_URL

        # Guarantee a clean slate before spawning. A managed process that is
        # alive but failed its health check would otherwise just be
        # overwritten below: it would keep the model resident forever AND
        # keep holding the port, dooming the replacement - which loads its
        # whole model into memory before it even tries to bind.
        if _process is not None:
            stop()

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
            # Our child's liveness is checked FIRST: this loop exists to
            # confirm the process we just spawned came up. If it died (e.g.
            # it couldn't bind because something else holds the port), a
            # healthy-looking port is that something else - treating it as
            # our success would skip the failure cooldown and re-spawn a
            # doomed, model-loading process on every single dictation.
            if _process.poll() is not None:
                break  # died during startup - stop polling, fall through to failure
            if _is_healthy():
                _last_failure_time = None
                _reset_idle_timer()
                return BASE_URL
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
            try:
                _process.terminate()
                try:
                    _process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _process.kill()
                    try:
                        _process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass  # unreapable (e.g. another user's process) - we
                              # did all we can; don't break the no-raise contract
            finally:
                # Always drop the handle: keeping a stale one is what makes
                # ensure_running() think it still manages something.
                _process = None
