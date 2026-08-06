import os
import signal
import subprocess
import sys
import time as real_time
from unittest.mock import MagicMock, patch

import pytest
import requests

import whisper_server

CONFIG = {
    "whisper_server_binary": "/opt/homebrew/bin/whisper-server",
    "whisper_model_path": "/config/models/ggml-medium.bin",
}


@pytest.fixture(autouse=True)
def _reset_module_state():
    """whisper_server.py holds process/timer state in module globals -
    reset before and after every test so tests can't leak state into
    each other."""
    def _reset():
        whisper_server._process = None
        if whisper_server._idle_timer is not None:
            whisper_server._idle_timer.cancel()
        whisper_server._idle_timer = None
        whisper_server._last_failure_time = None
        whisper_server._generation = 0
    _reset()
    yield
    _reset()


class _FakeProcess:
    def __init__(self, exit_code_after_start=None):
        self._exit_code = exit_code_after_start
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._exit_code

    def terminate(self):
        self.terminated = True
        self._exit_code = 0

    def kill(self):
        self.killed = True
        self._exit_code = -9

    def wait(self, timeout=None):
        return self._exit_code


def _fake_response(status_code):
    resp = MagicMock()
    resp.status_code = status_code
    return resp


def _two_phase_get():
    """Returns a side_effect function for requests.get: the first call is
    unhealthy (simulating 'nothing listening yet'), every call after that
    is healthy (simulating the just-spawned server coming up). Needed
    because ensure_running() checks health BEFORE spawning (to decide
    whether to adopt an already-running server) and AGAIN in the
    post-spawn polling loop - a test that wants to exercise the actual
    spawn path must be unhealthy on that first pre-spawn check."""
    responses = [requests.exceptions.ConnectionError("refused"), _fake_response(200)]

    def fake_get(url, timeout):
        result = responses.pop(0) if len(responses) > 1 else responses[0]
        if isinstance(result, Exception):
            raise result
        return result

    return fake_get


def test_ensure_running_short_circuits_when_already_healthy():
    fake_process = _FakeProcess()
    whisper_server._process = fake_process
    with patch("whisper_server.requests.get", return_value=_fake_response(200)), \
         patch("whisper_server.subprocess.Popen") as mock_popen:
        url = whisper_server.ensure_running(CONFIG)
    assert url == whisper_server.BASE_URL
    mock_popen.assert_not_called()


def test_ensure_running_cold_starts_when_nothing_running():
    fake_process = _FakeProcess()
    with patch("whisper_server.subprocess.Popen", return_value=fake_process) as mock_popen, \
         patch("whisper_server.requests.get", side_effect=_two_phase_get()):
        url = whisper_server.ensure_running(CONFIG)
    assert url == whisper_server.BASE_URL
    mock_popen.assert_called_once()
    args = mock_popen.call_args[0][0]
    assert args[0] == CONFIG["whisper_server_binary"]
    assert "-m" in args and CONFIG["whisper_model_path"] in args


def test_ensure_running_raises_when_server_never_becomes_healthy(monkeypatch):
    monkeypatch.setattr(whisper_server, "STARTUP_TIMEOUT_SECS", 0)
    fake_process = _FakeProcess()  # poll() keeps returning None ("still starting")
    with patch("whisper_server.subprocess.Popen", return_value=fake_process), \
         patch("whisper_server.requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
        with pytest.raises(whisper_server.WhisperServerError):
            whisper_server.ensure_running(CONFIG)
    assert whisper_server._last_failure_time is not None


def test_ensure_running_raises_immediately_when_launch_itself_fails():
    with patch("whisper_server.requests.get", side_effect=requests.exceptions.ConnectionError("refused")), \
         patch("whisper_server.subprocess.Popen", side_effect=OSError("no such file")):
        with pytest.raises(whisper_server.WhisperServerError):
            whisper_server.ensure_running(CONFIG)
    assert whisper_server._last_failure_time is not None


def test_ensure_running_respects_failure_cooldown():
    whisper_server._last_failure_time = real_time.monotonic()
    with patch("whisper_server.subprocess.Popen") as mock_popen:
        with pytest.raises(whisper_server.WhisperServerError):
            whisper_server.ensure_running(CONFIG)
    mock_popen.assert_not_called()


def test_ensure_running_retries_after_cooldown_elapses():
    whisper_server._last_failure_time = (
        real_time.monotonic() - whisper_server.FAILURE_COOLDOWN_SECS - 1
    )
    fake_process = _FakeProcess()
    with patch("whisper_server.subprocess.Popen", return_value=fake_process) as mock_popen, \
         patch("whisper_server.requests.get", side_effect=_two_phase_get()):
        url = whisper_server.ensure_running(CONFIG)
    assert url == whisper_server.BASE_URL
    mock_popen.assert_called_once()


def test_ensure_running_adopts_already_healthy_unmanaged_server():
    # _process is None (nothing we started ourselves) but something is
    # already listening healthily on the port AND is confirmed to be a
    # whisper-server - e.g. an orphan from a crashed previous run. Must adopt
    # it, not double-spawn, and must record a handle on it so stop() can
    # later end it.
    with patch("whisper_server.requests.get", return_value=_fake_response(200)), \
         patch("whisper_server.subprocess.Popen") as mock_popen, \
         patch("whisper_server._find_listening_pid", return_value=4242), \
         patch("whisper_server._is_whisper_server_pid", return_value=True), \
         patch("whisper_server.os.kill"):  # signalable
        url = whisper_server.ensure_running(CONFIG)
    assert url == whisper_server.BASE_URL
    mock_popen.assert_not_called()
    assert isinstance(whisper_server._process, whisper_server._AdoptedProcess)
    assert whisper_server._process.pid == 4242


def test_ensure_running_does_not_adopt_when_pid_discovery_fails():
    # PID discovery (lsof) can fail (missing binary, no match, etc.). Without
    # a PID we can't confirm what's on the port, so we must NOT adopt it -
    # falling through to the spawn attempt (which fails to bind and raises)
    # is the safe outcome.
    with patch("whisper_server.requests.get", return_value=_fake_response(200)), \
         patch("whisper_server.subprocess.Popen", return_value=_FakeProcess()) as mock_popen, \
         patch("whisper_server._find_listening_pid", return_value=None):
        whisper_server.ensure_running(CONFIG)
    mock_popen.assert_called_once()  # fell through to the normal spawn path
    assert not isinstance(whisper_server._process, whisper_server._AdoptedProcess)


def _foreign_process_on_the_port():
    """Simulates the reproduced bug's setup: an unrelated process (a
    `python -m http.server`, say) holds port 8090 and answers HTTP 200, so
    `ps -p <its pid> -o comm=` reports something that is not whisper-server,
    and any whisper-server we spawn dies immediately because it can't bind."""
    def fake_ps(cmd, **kwargs):
        assert cmd[:2] == ["ps", "-p"]  # the identity check, not lsof
        return MagicMock(returncode=0, stdout="/usr/bin/python3\n")

    return [
        patch("whisper_server.requests.get", return_value=_fake_response(200)),
        patch("whisper_server.subprocess.run", side_effect=fake_ps),
        patch("whisper_server.subprocess.Popen",
              return_value=_FakeProcess(exit_code_after_start=1)),
        patch("whisper_server._find_listening_pid", return_value=4242),
    ]


def test_unrelated_process_on_the_port_is_never_adopted_or_signalled():
    """Regression test for the reproduced bug: Dictify adopted whatever held
    port 8090 as long as it answered <500, then SIGTERM'd it on the idle
    timeout. Identity must be confirmed first, so an unrelated process is
    neither adopted nor ever signalled."""
    patches = _foreign_process_on_the_port()
    with patches[0], patches[1], patches[2], patches[3], \
         patch("whisper_server.os.kill") as mock_kill:
        try:
            whisper_server.ensure_running(CONFIG)
        except whisper_server.WhisperServerError:
            pass  # a failed start here is fine; killing the neighbour is not
        whisper_server.stop()  # the idle timer / quit hook's kill path

    mock_kill.assert_not_called()
    assert not isinstance(whisper_server._process, whisper_server._AdoptedProcess)


def test_doomed_spawn_against_an_occupied_port_arms_the_failure_cooldown():
    # Our child dies because it can't bind. The foreign server still answers
    # healthily, but that must not be mistaken for our own success - otherwise
    # no cooldown is armed and every dictation re-loads a 1.5GB model into a
    # process that's guaranteed to die.
    patches = _foreign_process_on_the_port()
    with patches[0], patches[1], patches[2], patches[3]:
        with pytest.raises(whisper_server.WhisperServerError):
            whisper_server.ensure_running(CONFIG)

    assert whisper_server._last_failure_time is not None


def test_ensure_running_does_not_adopt_a_process_it_is_not_allowed_to_signal():
    # A whisper-server belonging to another user: we could confirm what it
    # is, but we can't stop it. Adopting it would make every stop() spin out
    # both wait() timeouts (10s on the quit hook) and still leave it running.
    with patch("whisper_server.requests.get", return_value=_fake_response(200)), \
         patch("whisper_server.subprocess.Popen", return_value=_FakeProcess()), \
         patch("whisper_server._find_listening_pid", return_value=4242), \
         patch("whisper_server._is_whisper_server_pid", return_value=True), \
         patch("whisper_server.os.kill", side_effect=PermissionError("not permitted")):
        whisper_server.ensure_running(CONFIG)
    assert not isinstance(whisper_server._process, whisper_server._AdoptedProcess)


def test_adopted_process_is_not_signalled_once_its_pid_no_longer_matches():
    # PIDs get recycled: identity is re-checked on every signal, not only at
    # adoption time, so a pid that has since become something else is left
    # alone.
    adopted = whisper_server._AdoptedProcess(4242)
    with patch("whisper_server._is_whisper_server_pid", return_value=False), \
         patch("whisper_server.os.kill") as mock_kill:
        adopted.terminate()
        adopted.kill()
    mock_kill.assert_not_called()


def test_adopted_poll_reports_alive_when_the_process_cannot_be_signalled():
    # PermissionError means "it exists, I just can't touch it" - reporting it
    # as exited would let stop()/ensure_running() assume a clean slate.
    adopted = whisper_server._AdoptedProcess(4242)
    with patch("whisper_server.os.kill", side_effect=PermissionError("not permitted")):
        assert adopted.poll() is None


def test_adopted_poll_reports_gone_only_on_process_lookup_error():
    adopted = whisper_server._AdoptedProcess(4242)
    with patch("whisper_server.os.kill", side_effect=ProcessLookupError()):
        assert adopted.poll() == 0


def test_is_whisper_server_pid_rejects_this_python_process():
    # No mocks: a real `ps` call against a real PID that is definitely not a
    # whisper-server.
    assert whisper_server._is_whisper_server_pid(os.getpid()) is False


def test_is_whisper_server_pid_accepts_a_matching_process_name(monkeypatch):
    # Same real `ps` call, but with the expected name pointed at this test
    # process's own executable - proves the check accepts a genuine match and
    # isn't just always returning False.
    monkeypatch.setattr(
        whisper_server, "SERVER_PROCESS_NAME", os.path.basename(sys.executable)
    )
    assert whisper_server._is_whisper_server_pid(os.getpid()) is True


def test_stop_terminates_adopted_unmanaged_process():
    # Simulates a real OS process: os.kill(pid, 0) raises ProcessLookupError
    # once the process is actually gone, and turns "gone" only after the
    # SIGTERM lands - this lets _AdoptedProcess.wait() return immediately
    # instead of spinning for real time.
    state = {"alive": True}

    def fake_kill(pid, sig):
        assert pid == 4242
        if sig == signal.SIGTERM:
            state["alive"] = False
        elif sig == 0 and not state["alive"]:
            raise ProcessLookupError()

    with patch("whisper_server.requests.get", return_value=_fake_response(200)), \
         patch("whisper_server.subprocess.Popen") as mock_popen, \
         patch("whisper_server._find_listening_pid", return_value=4242), \
         patch("whisper_server._is_whisper_server_pid", return_value=True), \
         patch("whisper_server.os.kill", side_effect=fake_kill) as mock_kill:
        whisper_server.ensure_running(CONFIG)
        mock_popen.assert_not_called()
        whisper_server.stop()

    assert (4242, signal.SIGTERM) in [call.args for call in mock_kill.call_args_list]
    assert whisper_server._process is None


def test_ensure_running_stops_a_live_but_unhealthy_process_before_respawning():
    """Regression test: a managed process that's alive but fails one health
    check used to be silently overwritten by the new Popen handle. It kept
    running, kept the model resident, and kept holding port 8090 - dooming
    the replacement it was replaced by."""
    stale_process = _FakeProcess()  # alive (poll() -> None) but unhealthy
    whisper_server._process = stale_process
    fresh_process = _FakeProcess()
    with patch("whisper_server.subprocess.Popen", return_value=fresh_process) as mock_popen, \
         patch("whisper_server.requests.get", side_effect=_two_phase_get()):
        url = whisper_server.ensure_running(CONFIG)

    assert url == whisper_server.BASE_URL
    mock_popen.assert_called_once()
    assert stale_process.terminated is True  # not just dropped on the floor
    assert whisper_server._process is fresh_process


def test_stop_does_not_raise_when_the_process_refuses_to_die():
    """stop()'s contract is must-not-raise and idempotent. An unreapable
    process (both waits time out) must not break that, and must not leave a
    stale handle behind either - that stale handle is what feeds the
    live-but-unhealthy leak above."""
    class _UnreapableProcess(_FakeProcess):
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired(cmd="whisper-server", timeout=timeout)

    stubborn = _UnreapableProcess()
    whisper_server._process = stubborn

    whisper_server.stop()  # must not raise

    assert stubborn.terminated is True
    assert stubborn.killed is True
    assert whisper_server._process is None


def test_health_check_timeout_is_forgiving_of_a_transient_hiccup():
    # A false negative here costs a full process restart, so the timeout is a
    # named constant well above one second - and _is_healthy() actually uses it.
    assert whisper_server.HEALTH_CHECK_TIMEOUT_SECS >= 2
    with patch("whisper_server.requests.get", return_value=_fake_response(200)) as mock_get:
        whisper_server._is_healthy()
    assert mock_get.call_args.kwargs["timeout"] == whisper_server.HEALTH_CHECK_TIMEOUT_SECS


def test_stop_is_idempotent_with_nothing_running():
    whisper_server.stop()  # must not raise


def test_stop_terminates_running_process():
    fake_process = _FakeProcess()
    whisper_server._process = fake_process
    whisper_server.stop()
    assert fake_process.terminated is True
    assert whisper_server._process is None


def test_idle_timer_fires_stop_after_timeout(monkeypatch):
    monkeypatch.setattr(whisper_server, "IDLE_TIMEOUT_SECS", 0.05)
    fake_process = _FakeProcess()
    with patch("whisper_server.subprocess.Popen", return_value=fake_process), \
         patch("whisper_server.requests.get", side_effect=_two_phase_get()):
        whisper_server.ensure_running(CONFIG)
    assert whisper_server._process is fake_process
    real_time.sleep(0.2)
    assert whisper_server._process is None
    assert fake_process.terminated is True


def test_stale_idle_timeout_does_not_kill_a_renewed_process():
    """Regression test for the race where a timer thread that already fired
    (Timer.cancel() is a documented no-op past that point) can still reach
    the stop() call after ensure_running() has since re-validated the
    process and issued a newer generation via _reset_idle_timer(). The fired
    thread must recognize it's stale and back off instead of stopping the
    process out from under the caller that was just handed BASE_URL."""
    fake_process = _FakeProcess()
    with patch("whisper_server.requests.get", return_value=_fake_response(200)), \
         patch("whisper_server.subprocess.Popen") as mock_popen:
        whisper_server._process = fake_process
        whisper_server.ensure_running(CONFIG)  # issues generation 1
        stale_gen = whisper_server._generation
        whisper_server.ensure_running(CONFIG)  # renews the timer -> generation 2
        mock_popen.assert_not_called()

    assert stale_gen != whisper_server._generation

    # Simulate the stale generation-1 timer thread finally acquiring the
    # lock and running its callback, after the renewal above.
    whisper_server._on_idle_timeout(stale_gen)

    assert whisper_server._process is fake_process
    assert fake_process.terminated is False

    # A callback carrying the current generation must still be able to stop it.
    whisper_server._on_idle_timeout(whisper_server._generation)
    assert whisper_server._process is None
    assert fake_process.terminated is True
