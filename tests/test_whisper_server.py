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
    # already listening healthily on the port - e.g. an orphan from a
    # crashed previous run. Must adopt it, not double-spawn.
    with patch("whisper_server.requests.get", return_value=_fake_response(200)), \
         patch("whisper_server.subprocess.Popen") as mock_popen:
        url = whisper_server.ensure_running(CONFIG)
    assert url == whisper_server.BASE_URL
    mock_popen.assert_not_called()


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
