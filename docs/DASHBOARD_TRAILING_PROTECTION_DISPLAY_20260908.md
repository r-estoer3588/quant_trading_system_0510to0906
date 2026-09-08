# Dashboard trailing-protection display remediation (2026-09-08)

## Finding

The Alpaca dashboard currently renders both `exit_type="trailing"` and `exit_type="stop"` through the same branch:

```tsx
if ((p.exit_type === 'trailing' || p.exit_type === 'stop') && p.stop_price_est != null) {
  return { text: `stop ${fmtPrice(p.stop_price_est)}`, ... };
}
```

That is misleading for System1/System4.

For whole-share positions, the canonical protection path generates a broker-resident `trailing_stop` order (System1: 25%, System4: 20%) and gives trailing priority over ATR stop. The displayed `stop_price_est` is only the ATR stop estimate, not the broker trailing order's current stop price.

This caused whole-share System1 positions such as XPON/AMIX to be shown as `stop $0.0100`, even though the strategy exit type is trailing. `$0.0100` comes from the old ATR-estimation clamp and must not be presented as the active trailing protection.

Fractional legacy positions are different: Alpaca cannot host native stop/limit/trailing protection for fractional shares, so the exit checker falls back to once-daily synthetic ATR protection. Those positions must not be presented as native trailing either.

## Proposed display contract

The dashboard should fail visible and distinguish **strategy intent** from **broker-observed state**.

### 1. Whole-share trailing strategy position

When `exit_type === "trailing"` and quantity is whole-share:

- primary badge: `trail 25%` for System1 / `trail 20%` for System4, sourced from snapshot data rather than hard-coded in the component;
- secondary text: `broker常駐 期待` (or equivalent wording that explicitly means expected, not verified);
- do **not** display `stop_price_est` as though it were the current trailing-stop trigger.

Until the snapshot explicitly observes and exports the broker resting order, the dashboard must not say `armed`, `active`, or otherwise imply verified broker residency.

### 2. Legacy fractional trailing-strategy position

When `exit_type === "trailing"` and quantity is fractional:

- primary badge: `synthetic stop $X ⚠`;
- secondary text: `trail gap / 日次ATR`;
- use the ATR stop estimate only as a synthetic fallback threshold, never label it as native trailing.

### 3. Ordinary stop strategy position

When `exit_type === "stop"`:

- keep the existing `stop $X` representation.

### 4. Broker verification (follow-up / preferred final state)

Extend the read-only snapshot exporter to inspect open Alpaca protection orders and export per-position facts such as:

```json
{
  "protection_expected": "trailing",
  "protection_mode": "native_trailing_expected | synthetic_daily | native_stop_expected",
  "protection_observed": "trailing | stop | oco | none | unmeasured",
  "protection_verified": true,
  "trail_percent": 25.0,
  "resting_client_order_id": "..."
}
```

Then the UI can distinguish:

- `trail 25% ✓ broker` — verified resting native trailing;
- `trail 25% ?` — expected but broker observation unavailable;
- `synthetic stop $X ⚠` — legacy fractional gap;
- `UNPROTECTED` — expected protection absent in a measured snapshot.

## Backend consistency issue discovered at the same time

`scripts/export_alpaca_snapshot.py::_estimate_stop_target()` still uses the historical long-stop clamp:

```python
max(0.01, avg_entry - dist)
```

while the authoritative exit path in `common/alpaca_trading.py` now avoids a silent `$0.01` pseudo-stop and uses the configured disaster-stop floor when the raw ATR stop is non-positive.

The snapshot estimator should use the same canonical stop-price helper (or a shared public helper) instead of maintaining a second formula. The dashboard must not reintroduce a value that execution itself no longer considers the correct protective stop.

## Acceptance criteria

1. Whole-share System1 positions never render an ATR estimate as `stop $...` when `exit_type` is trailing.
2. Whole-share System4 follows the same rule.
3. Fractional legacy System1/System4 positions are explicitly marked synthetic / trailing-gap.
4. No UI wording claims broker protection is armed unless the exporter actually observed a resting broker order.
5. Snapshot trailing percentage comes from strategy rules/exported data, not duplicated constants in React.
6. Snapshot stop estimates share the same floor logic as the execution path; `$0.0100` is not silently emitted when execution would use the disaster-stop floor.
7. Add regression coverage for whole-share trailing, fractional trailing-gap, ordinary stop, and non-positive raw ATR stop.

## Safety boundary

This remediation is read-only / presentation + snapshot consistency work. It must not place, cancel, replace, resize, or migrate any broker order or position. Existing fractional positions remain untouched.
