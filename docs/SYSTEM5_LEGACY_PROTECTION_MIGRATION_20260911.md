# System5 legacy protection migration (2026-09-11)

## Purpose

This change does **not** modify the Bensdorp System5 strategy. It repairs broker-resident
Paper protection left by the pre-#178 execution path.

Canonical System5 remains:

- entry: previous close -3% LIMIT
- stop: entry - 3 x ATR10, using ATR10 frozen from the completed bar before entry
- target trigger: entry + 1 x the same frozen ATR10
- target execution: next trading-session open, market
- timeout: observe six post-entry sessions, then exit at the seventh session open

## Observed legacy state

The 2026-09-10 read-only Paper audit found legacy S5 OCO orders. WIX and YEXT also
proved ATR drift because the broker-resident stop differed from the frozen-entry-ATR
canonical stop. A new 2026-09-10 ARCT position had no native stop, demonstrating the
separate post-entry protection timing gap.

## Migration safety contract

A legacy OCO/stop is eligible for cancel+replace only when the runner has already read
and preserved its broker-resident downside stop price. Missing or ambiguous rollback
evidence is fail-closed: the existing order remains untouched.

For an eligible migration:

1. propose the canonical frozen-ATR standalone GTC stop;
2. scope cancellation to the exact old protection client_order_id;
3. preserve the observed old stop price in the proposal as rollback evidence;
4. if canonical replacement submission fails, re-arm the observed old downside stop;
5. perform at most **one System5 destructive migration per paper_exit_check run**;
6. mark later candidates `s5_migration_deferred:one_per_run` so they are visible and
   advance on a later run rather than being batch-canceled;
7. never use rollback evidence to alter target/stop strategy math.

A same-day rollback stop is not churned again. A later run may retry migration on a
prior-day rollback stop using a fresh client_order_id.

## Non-goals

- no System5 parameter/filter/ranking/sizing changes
- no change to S1/S2/S3/S4/S6/S7 strategy semantics
- no live-account enablement
- no automatic liquidation/resize of existing positions
- no attempt to solve the separate post-entry protection-arm timing gap in this patch
