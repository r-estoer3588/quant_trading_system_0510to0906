# System1 fractional trailing-stop gap (2026-09-07)

## Finding

System1's canonical exit contract is 5ATR20 initial stop plus a 25% trailing stop, with no profit target and no time exit. Integer-share positions can use Alpaca's native `trailing_stop` order, but fractional positions cannot. The current fractional path therefore falls back to a once-per-day synthetic ATR stop and does not preserve the System1 high-water-mark (HWM) trailing behavior.

This changes the strategy semantics for fractional System1 positions: winners are not exited after a 25% drawdown from post-entry highs, and protection is evaluated only when `paper_exit_check.py` runs.

## Required remediation

The remediation must satisfy one of these contracts before being considered closed:

1. Keep System1/S4 positions integer-sized so the existing native Alpaca trailing-stop path is always available; or
2. Persist per-position HWM and implement a synthetic 25% trailing threshold with an intraday watcher whose latency is explicitly bounded and surfaced.

Daily-only HWM evaluation is not equivalent to the native contract and should not be described as such.

## Acceptance criteria

- New System1 entries cannot silently enter the `synthetic_daily` path solely because sizing produced fractional shares.
- Existing fractional System1 positions are explicitly surfaced as `trailing_gap` (or an equivalent fail-visible state) until migrated/closed.
- Tests cover fractional sizing, native trailing proposal generation, and the dashboard/artifact representation of the gap.
- No change weakens the existing 5ATR initial-stop behavior for integer positions.
