import fcntl
import os

import dictate


def test_acquire_singleton_lock_succeeds_when_no_lock_file(tmp_path):
    lock_path = tmp_path / "dictify.lock"
    assert dictate._acquire_singleton_lock(lock_path) is True
    assert lock_path.read_text().strip() == str(os.getpid())


def test_acquire_singleton_lock_fails_when_another_process_holds_it(tmp_path):
    lock_path = tmp_path / "dictify.lock"
    # Simulate another live instance by holding a real flock via a
    # separate file descriptor - real OS-level locking behavior, no mocks.
    holder_fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY)
    fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert dictate._acquire_singleton_lock(lock_path) is False
    finally:
        os.close(holder_fd)


def test_acquire_singleton_lock_leaves_existing_content_untouched_on_failure(tmp_path):
    lock_path = tmp_path / "dictify.lock"
    holder_fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY)
    os.write(holder_fd, b"12345")
    fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert dictate._acquire_singleton_lock(lock_path) is False
        assert lock_path.read_text() == "12345"
    finally:
        os.close(holder_fd)


def test_acquire_singleton_lock_succeeds_after_holder_releases(tmp_path):
    lock_path = tmp_path / "dictify.lock"
    holder_fd = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY)
    fcntl.flock(holder_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.close(holder_fd)  # releases the lock, simulating the holder process exiting
    assert dictate._acquire_singleton_lock(lock_path) is True
