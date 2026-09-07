"""Regression guards for the 2026-09-07 whole-share Paper execution policy.

The live correctness issue was not order sizing itself: fractional long entries could
not carry Alpaca-native stop/trailing protection, so S1/S4 silently changed exit
semantics. Production Paper entry planning must therefore never re-enable notional
fractional execution without an explicit design change and replacement protection.
"""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts" / "paper_trading_submit.py"
DRYRUN = ROOT / "scripts" / "paper_trading_dryrun.py"


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def test_submit_json_path_is_whole_share_only() -> None:
    code = _read(SUBMIT)
    assert "prefer_fractional=False" in code
    assert '"prefer_fractional": False' in code
    assert '"whole_share_only": PAPER_WHOLE_SHARE_ONLY' in code
    # Old default silently re-enabled fractional entries unless a CLI flag was passed.
    assert "prefer_fractional=(not args.no_fractional)" not in code


def test_dryrun_matches_submit_whole_share_policy() -> None:
    code = _read(DRYRUN)
    assert "PAPER_WHOLE_SHARE_ONLY = True" in code
    assert "prefer_fractional=False" in code
    assert '"prefer_fractional": False' in code
    assert '"whole_share_only": PAPER_WHOLE_SHARE_ONLY' in code
    assert "prefer_fractional=(not args.no_fractional)" not in code


def test_legacy_no_fractional_flag_cannot_disable_policy() -> None:
    """Keep the old CLI accepted but make it a compatibility no-op.

    This avoids breaking Task Scheduler/wrappers that still pass --no-fractional while
    preventing the inverse (flag absent => fractional) behavior from returning.
    """
    submit = _read(SUBMIT)
    dryrun = _read(DRYRUN)
    assert "後方互換 no-op。Paper 実発注は常に整数株のみ" in submit
    assert "後方互換 no-op。Paper は常に整数株のみで計画する" in dryrun
