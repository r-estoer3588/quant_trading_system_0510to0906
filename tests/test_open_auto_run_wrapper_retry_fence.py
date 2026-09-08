"""Source contracts for the scheduled open-run wrapper safety fence."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "open_auto_run.ps1"


def _source() -> str:
    return WRAPPER.read_text(encoding="utf-8-sig")


def test_wrapper_scrubs_clock_unknown_env_before_python_launch():
    """Nightly cannot inherit the emergency clock-unknown bypass from .env."""
    source = _source()
    scrub = "Remove-Item Env:OPEN_RUN_ALLOW_CLOCK_UNKNOWN"
    launch = "& python @pyArgs"
    assert scrub in source
    assert launch in source
    assert source.index(scrub) < source.index(launch)
    assert 'if ($AllowClockUnknown) { $pyArgs += "--allow-clock-unknown" }' in source


def test_wrapper_uses_atomic_inflight_create_new_before_python():
    """Concurrent/retried wrappers cannot both enter the order runner."""
    source = _source()
    acquire = "[System.IO.FileMode]::CreateNew"
    launch = "& python @pyArgs"
    assert '"INFLIGHT.lock"' in source
    assert acquire in source
    assert source.index(acquire) < source.index(launch)
    assert "exit 5" in source


def test_wrapper_releases_inflight_only_for_known_safe_exit_codes():
    """Unexpected failure preserves the fence for manual order-state inspection."""
    source = _source()
    assert "($code -eq 0) -or ($code -eq 3) -or ($code -eq 4)" in source
    assert "preserving INFLIGHT.lock after unexpected exit=$code" in source
    assert "manual order-state check required" in source


def test_done_lock_can_clear_stale_inflight_after_completed_run():
    """Wrapper death after Python DONE must not freeze the next scheduled trigger."""
    source = _source()
    assert "(Test-Path $DoneLock) -or $Force" in source
    assert "cleared stale INFLIGHT.lock" in source
