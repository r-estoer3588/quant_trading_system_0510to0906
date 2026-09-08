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
- secondary text: `broker常駐 期待` only when broker observation has not yet been performed;
- do **not** display `stop_price_est` as though it were the current trailing-stop trigger.

Until the snapshot explicitly observes and exports the broker resting order, the dashboard must not say `armed`, `active`, or otherwise imply verified broker residency.

Whole-share detection must be robust to float representation. Do not rely on a raw `qty % 1 == 0` comparison; normalize broker quantity using decimal/string semantics or an explicit epsilon.

### 2. Legacy fractional trailing-strategy position

When `exit_type === "trailing"` and quantity is fractional:

- primary badge: `synthetic stop $X ⚠` only when the ATR threshold is actually measured;
- secondary text: `trail gap / 日次ATR`;
- if the ATR threshold is unavailable, show `synthetic daily ⚠` + `threshold未計測` rather than inventing a price;
- use the ATR stop estimate only as a synthetic fallback threshold, never label it as native trailing.

### 3. Ordinary stop strategy position

When `exit_type === "stop"`:

- keep the existing `stop $X` representation when the threshold is measured;
- show an explicit unmeasured state when it is not.

### 4. Broker verification — required for the final state

The preferred final fix is not merely a cosmetic badge change. Extend the read-only snapshot exporter to inspect **open protection orders** and export per-position facts such as:

```json
{
  "protection_expected": "trailing",
  "protection_mode": "native_trailing_expected | synthetic_daily | native_stop_expected",
  "protection_observed": "trailing | stop | oco | none | unmeasured",
  "protection_verified": true,
  "protection_observed_at": "...Z",
  "trail_percent": 25.0,
  "resting_client_order_id": "...",
  "broker_stop_price": 4.32,
  "broker_hwm": 5.76
}
```

Use an **OPEN-orders query** for this verification rather than reusing the historical `ALL, limit=500` entry-order index. A bounded historical list can omit the currently relevant protection order on a mature account; open-order observation needs its own read path and failure state.

Then the UI can distinguish:

- `trail 25% ✓ broker` — verified resting native trailing;
- `trail 25% ?` — expected, but broker observation unavailable/unmeasured;
- `synthetic stop $X ⚠` — legacy fractional gap;
- `UNPROTECTED` — expected protection absent **only when the open-order observation succeeded and the lifecycle says the order should already be armed**.

Do not label `none` as `UNPROTECTED` before the protection run is expected to have executed. Otherwise the dashboard creates a false alarm during the normal interval between entry fill and the protection-order run. Export enough timing/lifecycle context to distinguish `pending_arm` from `missing_after_arm`.

When Alpaca exposes the current trailing `stop_price` / HWM on the resting order, prefer that broker-observed value. Do not synthesize a current trailing trigger from entry price or ATR.

## Backend consistency issue discovered at the same time

`scripts/export_alpaca_snapshot.py::_estimate_stop_target()` still uses the historical long-stop clamp:

```python
max(0.01, avg_entry - dist)
```

while the authoritative exit path in `common/alpaca_trading.py` now avoids a silent `$0.01` pseudo-stop and uses the configured disaster-stop floor when the raw ATR stop is non-positive.

The snapshot estimator should not copy that formula again. Extract/share a public canonical stop-price helper used by both execution and snapshot generation so environment-driven floor settings cannot drift. Importing a private underscore helper from the execution module is acceptable only as a temporary bridge, not the final design.

## Self-review findings / implementation blockers

1. **Documentation-only is not a fix.** This draft must remain draft until runtime snapshot/UI code and tests are present.
2. **Intent is not broker state.** `exit_type="trailing"` proves strategy intent only; it does not prove that an Alpaca trailing order is resting.
3. **Historical order lookup is the wrong verifier.** Protection verification needs a dedicated OPEN-order query with explicit success/failure measurement semantics.
4. **Missing protection needs lifecycle context.** `none` is only an error after the system had a reasonable opportunity to arm protection; before then it is pending, not broken.
5. **Fractional fallback can itself be unmeasured.** If ATR/current-price inputs are missing, show that gap instead of a fabricated threshold.
6. **Stop estimation has duplicate business logic.** The dashboard exporter must consume the same stop-floor function/configuration as execution.
7. **Quantity classification needs numeric tolerance.** Exact float integer tests are too brittle for broker quantities.
8. **Do not overclaim from XPON/AMIX.** The current screenshots prove a dashboard representation bug; they do not by themselves prove the broker order is present or absent.

## Acceptance criteria

1. Whole-share System1 positions never render an ATR estimate as `stop $...` when `exit_type` is trailing.
2. Whole-share System4 follows the same rule.
3. Fractional legacy System1/System4 positions are explicitly marked synthetic / trailing-gap.
4. No UI wording claims broker protection is armed unless the exporter actually observed a resting broker order.
5. Snapshot trailing percentage comes from strategy rules/exported data, not duplicated constants in React.
6. Snapshot stop estimates share the same canonical floor logic as the execution path; `$0.0100` is not silently emitted when execution would use the disaster-stop floor.
7. Open-order observation is separately measured; API failure maps to `unmeasured`, never `none`.
8. `UNPROTECTED` is emitted only after successful observation and after the expected arming lifecycle point; pre-arm positions are `pending_arm`.
9. Fractional detection is robust to numeric representation.
10. Add regression coverage for: verified whole-share trailing, whole-share trailing unmeasured, pending-arm, missing-after-arm, fractional trailing-gap with/without ATR, ordinary stop, and non-positive raw ATR stop.
11. Read-only invariant remains intact: no order placement/cancel/replace, no resizing, no forced migration.

## Safety boundary

This remediation is read-only / presentation + snapshot consistency work. It must not place, cancel, replace, resize, or migrate any broker order or position. Existing fractional positions remain untouched.
