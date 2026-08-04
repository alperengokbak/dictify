import os

import dictate


def test_acquire_singleton_lock_succeeds_when_no_lock_file(tmp_path):
    lock_path = tmp_path / "dictify.lock"
    assert dictate._acquire_singleton_lock(lock_path) is True
    assert lock_path.read_text().strip() == str(os.getpid())


def test_acquire_singleton_lock_fails_when_another_instance_is_alive(tmp_path):
    lock_path = tmp_path / "dictify.lock"
    # Our own PID is always alive from our own perspective - a reliable
    # stand-in for "another live process" without spawning a real one.
    lock_path.write_text(str(os.getpid()))
    assert dictate._acquire_singleton_lock(lock_path) is False


def test_acquire_singleton_lock_succeeds_when_lock_file_is_stale(tmp_path):
    lock_path = tmp_path / "dictify.lock"
    dead_pid = 999999  # far past any realistic live PID on macOS
    lock_path.write_text(str(dead_pid))
    assert dictate._acquire_singleton_lock(lock_path) is True
    assert lock_path.read_text().strip() == str(os.getpid())


def test_acquire_singleton_lock_succeeds_when_lock_file_is_corrupt(tmp_path):
    lock_path = tmp_path / "dictify.lock"
    lock_path.write_text("not-a-pid")
    assert dictate._acquire_singleton_lock(lock_path) is True
