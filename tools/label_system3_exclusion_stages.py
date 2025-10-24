"""Label System3 symbols with the stage they were excluded at.

Reads a diagnostics snapshot to find system3 exclude_symbols and inspects the
latest row of each symbol's cached DataFrame. Writes a CSV with the
exclusion stage and numeric values. Thresholds are not modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# ensure repo root is on sys.path so imports work when running from tools/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.system_setup_predicates import system3_setup_predicate
from common.utils import get_cached_data


def _parse_setlike(s: Any) -> list[str]:
    if s is None:
        return []
    if isinstance(s, (list, tuple, set)):
        return list(map(str, s))
    if isinstance(s, str):
        s2 = s.strip()
        if s2.startswith("{") and s2.endswith("}"):
            inner = s2[1:-1].strip()
            if not inner:
                return []
            parts = [p.strip().strip("'\"") for p in inner.split(",")]
            return [p for p in parts if p]
    return []


def _get_thresholds() -> tuple[float, float]:
    # Follow core/system3 behaviour: default drop3d=0.125, atr_ratio=0.05
    drop_thr = 0.125
    atr_thr = 0.05
    try:
        from config.environment import get_env_config as _get_env

        if _get_env is not None:
            env = _get_env()
            try:
                v = getattr(env, "min_atr_ratio_for_test", None)
                if v is not None:
                    atr_thr = float(v)
            except Exception:
                pass
            try:
                v2 = getattr(env, "min_drop3d_for_test", None)
                if (
                    v2 is not None
                    and hasattr(env, "is_test_mode")
                    and bool(env.is_test_mode())
                ):
                    drop_thr = float(v2)
            except Exception:
                pass
    except Exception:
        pass
    return drop_thr, atr_thr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True)
    p.add_argument(
        "--out-csv", default="results_csv_test/system3_stage_labels_20251022.csv"
    )
    p.add_argument("--symbols", nargs="*", help="optional list of symbols to process")
    args = p.parse_args()

    snap_p = Path(args.snapshot)
    if not snap_p.exists():
        print("snapshot not found", snap_p)
        raise SystemExit(2)
    with snap_p.open("r", encoding="utf-8") as fh:
        snap = json.load(fh)

    systems = snap.get("systems") or []
    sys3 = next((s for s in systems if s.get("system_id") == "system3"), None)
    if not sys3:
        print("no system3 block in snapshot")
        raise SystemExit(3)
    excl = (sys3.get("diagnostics_extra") or {}).get("exclude_symbols") or {}
    drop_set = _parse_setlike(excl.get("drop3d"))
    close_set = _parse_setlike(excl.get("close_vs_sma150"))
    default_symbols = sorted(set(drop_set + close_set))

    symbols = args.symbols if args.symbols else default_symbols
    if not symbols:
        print(
            "no symbols selected (either pass --symbols or ensure snapshot has exclude_symbols)"
        )
        raise SystemExit(4)

    drop_thr, atr_thr = _get_thresholds()

    out_rows: list[dict[str, Any]] = []
    rc_missing = []
    for sym in symbols:
        df = get_cached_data(sym)
        if df is None or getattr(df, "empty", True):
            rc_missing.append(sym)
            continue
        try:
            last = df.iloc[-1]
        except Exception:
            rc_missing.append(sym)
            continue

        def _f(k: str) -> float | None:
            try:
                v = last.get(k)
                if v is None:
                    return None
                if pd.isna(v):
                    return None
                return float(v)
            except Exception:
                return None

        close_v = _f("Close")
        dvol_v = _f("dollarvolume20")
        atr_v = _f("atr_ratio")
        drop_v = _f("drop3d")

        close_ok = close_v is not None and close_v >= 5.0
        vol_ok = dvol_v is not None and dvol_v > 25_000_000
        atr_ok = atr_v is not None and atr_v >= atr_thr

        filter_pass = close_ok and vol_ok and atr_ok

        # predicate check using shared predicate function
        try:
            pred_res = system3_setup_predicate(last, return_reason=True)
            if isinstance(pred_res, tuple):
                pred_flag, pred_reason = bool(pred_res[0]), pred_res[1]
            else:
                pred_flag, pred_reason = bool(pred_res), None
        except Exception:
            pred_flag, pred_reason = False, None

        final_flag = pred_flag
        # test-mode override (do not change thresholds here)
        try:
            from config.environment import get_env_config as _get_env

            if _get_env is not None:
                env = _get_env()
                if (
                    not final_flag
                    and hasattr(env, "is_test_mode")
                    and bool(env.is_test_mode())
                ):
                    v = getattr(env, "min_drop3d_for_test", None)
                    if v is not None and drop_v is not None and drop_v >= float(v):
                        final_flag = True
        except Exception:
            pass

        stage = "passed"
        fail_reasons = []
        if not filter_pass:
            stage = "filter_fail"
            if not close_ok:
                fail_reasons.append("close_lt_5")
            if not vol_ok:
                fail_reasons.append("dvol_le_25m")
            if not atr_ok:
                fail_reasons.append("atr_ratio_lt_thr")
        elif not pred_flag:
            stage = "predicate_fail"
        elif not final_flag:
            stage = "drop3d_fail"

        out_rows.append(
            {
                "symbol": sym,
                "stage": stage,
                "fail_reasons": ",".join(fail_reasons) if fail_reasons else None,
                "pred_reason": pred_reason,
                "close": close_v,
                "dollarvolume20": dvol_v,
                "atr_ratio": atr_v,
                "drop3d": drop_v,
            }
        )

    out_p = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(out_p, index=False)
    print(f"Wrote CSV: {out_p} (rows={len(df_out)})")
    if rc_missing:
        print(f"Missing cached frames for symbols: {rc_missing}")


if __name__ == "__main__":
    main()
