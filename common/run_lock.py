"""Small cross-platform run lock utility used to serialize heavy pipeline runs.

This implementation uses an atomic directory creation as the lock primitive.
Directory creation is atomic on both POSIX and Windows, making it a simple
cross-platform locking mechanism without extra dependencies.

Usage:
    from common.run_lock import RunLock

    lock = RunLock('today_signals', timeout=300)
    lock.acquire()
    try:
        # critical section
    finally:
        lock.release()

You can also use it as a context manager:
    with RunLock('today_signals'):
        # critical section

The lock directory lives under PROJECT_ROOT/locks by default. A stale-lock
cleanup is attempted when the existing lock looks older than
PIPELINE_LOCK_STALE_SECONDS (default 3600s).
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import time
from typing import Any

DEFAULT_STALE_SECONDS = int(os.getenv("PIPELINE_LOCK_STALE_SECONDS", "3600"))


class RunLock:
    """Context manager / helper that acquires a lock by creating a directory.

    The lock is represented as a directory at <repo_root>/locks/<name>.lock.
    Attempting to create the same directory concurrently fails with
    FileExistsError which we use to detect contention.
    """

    def __init__(
        self,
        name: str = "pipeline",
        timeout: float | None = 600.0,
        stale_seconds: int | None = None,
        root: Path | None = None,
    ) -> None:
        self.name = str(name)
        self.timeout = None if timeout is None else float(timeout)
        self.stale_seconds = (
            DEFAULT_STALE_SECONDS if stale_seconds is None else int(stale_seconds)
        )
        if root is None:
            # repo root is two levels up from this file (common/ -> repo)
            root = Path(__file__).resolve().parents[1]
        self.root = Path(root)
        self.locks_dir = self.root / "locks"
        self.lock_path = self.locks_dir / f"{self.name}.lock"
        self._acquired = False

    def acquire(self, poll_interval: float = 0.25) -> None:
        """Try to acquire the lock, waiting up to timeout seconds.

        Raises TimeoutError on timeout.
        """
        start = time.monotonic()
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                # atomic directory creation
                os.mkdir(self.lock_path)
                # create metadata for diagnostics
                try:
                    with open(self.lock_path / "owner.txt", "w", encoding="utf-8") as f:
                        f.write(str({"pid": os.getpid(), "time": time.time()}))
                except Exception:
                    # non-fatal - metadata is best-effort
                    pass
                self._acquired = True
                return
            except FileExistsError:
                # existing lock - check staleness and try cleanup if old
                try:
                    mtime = self.lock_path.stat().st_mtime
                    age = time.time() - float(mtime)
                except Exception:
                    age = 0
                if self.stale_seconds and age > float(self.stale_seconds):
                    # attempt to remove stale lock (best-effort)
                    try:
                        shutil.rmtree(self.lock_path)
                    except Exception:
                        # failed to clear stale lock - keep waiting
                        pass
                # retry or time out
                if (
                    self.timeout is not None
                    and (time.monotonic() - start) >= self.timeout
                ):
                    raise TimeoutError(
                        f"Timeout acquiring lock '{self.lock_path}' after "
                        f"{self.timeout} seconds"
                    )
                time.sleep(poll_interval)

    def release(self) -> None:
        """Release the lock if held. Non-fatal on failures."""
        if not self._acquired:
            return
        try:
            # remove directory and metadata
            if self.lock_path.exists():
                shutil.rmtree(self.lock_path)
        except Exception:
            # best-effort release
            pass
        self._acquired = False

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.release()
