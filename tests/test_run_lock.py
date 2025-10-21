import os
import threading
import time
from pathlib import Path

from common.run_lock import RunLock


def test_acquire_and_release(tmp_path: Path) -> None:
    root = tmp_path
    lock_name = "testlock"

    rl = RunLock(lock_name, timeout=3, root=root)
    rl.acquire()
    assert (root / "locks" / f"{lock_name}.lock").exists()

    acquired_event = threading.Event()
    errors: list[Exception] = []

    def worker(ev: threading.Event) -> None:
        try:
            rl2 = RunLock(lock_name, timeout=5, root=root)
            rl2.acquire()
            ev.set()
            time.sleep(0.05)
            rl2.release()
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=worker, args=(acquired_event,))
    t.start()

    # ensure worker is blocked while main holds the lock
    time.sleep(0.1)
    assert not acquired_event.is_set()

    # release and let worker acquire
    rl.release()
    t.join(timeout=3)
    assert t.is_alive() is False
    assert acquired_event.is_set()


def test_stale_lock_removal(tmp_path: Path) -> None:
    root = tmp_path
    lock_name = "stalelock"
    locks_dir = root / "locks"
    locks_dir.mkdir(parents=True)
    lock_path = locks_dir / f"{lock_name}.lock"
    lock_path.mkdir()
    # set mtime in the past so it's considered stale
    old_time = time.time() - 3600 * 24
    Path(lock_path).stat()
    try:
        lock_path.joinpath("owner.txt").write_text("stale")
    except Exception:
        pass
    try:
        os.utime(lock_path, (old_time, old_time))
    except Exception:
        # best-effort; some filesystems in CI may not support utime on empty dirs
        pass

    rl = RunLock(lock_name, timeout=3, stale_seconds=1, root=root)
    # should acquire by removing stale lock
    rl.acquire()
    assert (root / "locks" / f"{lock_name}.lock").exists()
    rl.release()
