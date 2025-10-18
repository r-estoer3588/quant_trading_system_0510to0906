from __future__ import annotations

"""
rolling最終日が直近営業日でない銘柄の『理由』を自動分類します。

機能:
- logs/rolling_latest_day_check_*.csv（直近）を読み込み、ズレのある銘柄を分類
- 分類結果を表示し、logs/rolling_latest_day_classified_YYYYMMDD.csv に保存
- オプションでサンプル数件を個別APIで取得してキャッシュへ反映（検証用）

想定する代表的な理由:
- derivative_suffix: 記号末尾が W/R/U などで、ワラント/権利/ユニット等の可能性
- no_expected_in_full: full（バックアップ）に expected 日が存在しない（上流未着/遅配）
- latest_row_all_nan: rolling の最終行の価格がすべて NaN（品質チェックで除外）
- full_newer_than_rolling: full の最新日が rolling より新しい（rolling再構築ラグ）
- unknown: 上記いずれにも該当しない
"""

import glob
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd

from common.cache_manager import CacheManager
from config.settings import get_settings

PRICE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "adjclose",
    "volume",
    "Open",
    "High",
    "Low",
    "Close",
    "AdjClose",
    "Volume",
]


def _detect_date_col(df: pd.DataFrame) -> str | None:
    for c in ("date", "Date"):
        if c in df.columns:
            return c
    return None


DERIV_SUFFIX_PATTERNS = (
    r"W$",  # Warrant
    r"WW$",  # Double W
    r"WS$|WT$",  # Warrant variants
    r"R$|RT$",  # Rights
    r"U$|UN$",  # Units
)


def _is_derivative_like(symbol: str) -> bool:
    sym = symbol.upper()
    return any(re.search(pat, sym) for pat in DERIV_SUFFIX_PATTERNS)


def _classify_reason(
    cm: CacheManager,
    symbol: str,
    expected: pd.Timestamp,
    last_rolling_date: pd.Timestamp,
) -> str:
    # 1) 末尾サフィックスによるデリバティブ/非普通株の可能性
    if _is_derivative_like(symbol):
        return "derivative_suffix"

    # rolling の最終行価格が全NaNか
    try:
        roll = cm.read(symbol, "rolling")
    except Exception:
        roll = None
    if roll is not None and not getattr(roll, "empty", True):
        dcol = _detect_date_col(roll)
        if dcol:
            try:
                idx = pd.to_datetime(roll[dcol], errors="coerce").idxmax()
                last_row = roll.loc[idx]
                prices = [last_row[c] for c in PRICE_COLS if c in roll.columns]
                if prices and pd.isna(pd.Series(prices)).all():
                    return "latest_row_all_nan"
            except Exception:
                pass

    # full（バックアップ）側の日付状況
    have_expected_in_full = False
    full_latest = None
    try:
        full = cm.read(symbol, "full")
        if full is not None and not getattr(full, "empty", True):
            dcolf = _detect_date_col(full)
            if dcolf:
                dts = pd.to_datetime(full[dcolf], errors="coerce").dt.normalize()
                have_expected_in_full = bool((dts == expected).any())
                full_latest = dts.max()
    except Exception:
        pass

    if not have_expected_in_full:
        return "no_expected_in_full"

    if full_latest is not None and pd.notna(full_latest) and full_latest > last_rolling_date:
        return "full_newer_than_rolling"

    return "unknown"


def classify_and_report(patch_samples: int = 0, include_derivatives: bool = False) -> int:
    settings = get_settings(create_dirs=True)
    cm = CacheManager(settings)

    # 最新のチェックCSVを取得
    csvs = sorted(glob.glob("logs/rolling_latest_day_check_*.csv"))
    if not csvs:
        print("❌ 先に scripts/diag_check_rolling_latest_day.py を実行してください。")
        return 2
    src_path = csvs[-1]
    dfc = pd.read_csv(src_path)
    if dfc.empty:
        print("❌ 入力CSVが空です:", src_path)
        return 2

    expected_str = str(dfc["expected_date"].iloc[0])
    expected = pd.Timestamp(expected_str)

    target = dfc[dfc["last_date"] != expected_str].copy()
    if target.empty:
        print("✅ すべての銘柄が直近営業日です。")
        return 0

    # 分類
    reasons: list[str] = []
    out_rows: list[dict[str, Any]] = []
    for _, row in target.iterrows():
        sym = str(row["symbol"]).strip()
        last_date = pd.Timestamp(row["last_date"]).normalize()
        reason = _classify_reason(cm, sym, expected, last_date)
        reasons.append(reason)
        out_rows.append(
            {
                "symbol": sym,
                "last_date": str(last_date.date()),
                "expected_date": expected_str,
                "diff_days": int((expected - last_date).days),
                "reason": reason,
            }
        )

    # 集計
    s = pd.Series(reasons).value_counts()
    print("============================================================")
    print(f"[分類] 対象銘柄: {len(target)} 件  基準日: {expected_str}")
    print("[内訳]")
    for k, v in s.items():
        print(f"  - {k}: {v}")

    # 代表例を表示
    ex_df = pd.DataFrame(out_rows)
    for k in s.index.tolist():
        samp = ex_df[ex_df["reason"] == k].head(5)["symbol"].tolist()
        if samp:
            print(f"  例({k}): {', '.join(samp)}")

    # 保存
    out_dir = Path("logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (f"rolling_latest_day_classified_{expected.strftime('%Y%m%d')}.csv")
    pd.DataFrame(out_rows).sort_values(["reason", "diff_days", "symbol"]).to_csv(
        out_path, index=False, encoding="utf-8"
    )
    print("------------------------------------------------------------")
    print(f"[保存] {out_path} に分類結果を書き出しました。")

    # 追加: サンプル個別APIで最新行を取得して反映（任意）
    if patch_samples > 0:
        try:
            from scripts.verify_bulk_accuracy import BulkDataVerifier
        except Exception as e:  # 安全フォールバック
            print(f"⚠️ 個別APIモジュール読み込み失敗: {e}")
            return 1

        verifier = BulkDataVerifier()
        patched = 0

        # 対象候補: デリバティブを除外（include_derivatives=Falseのとき）
        cand_df = ex_df.copy()
        if not include_derivatives:
            cand_df = cand_df[cand_df["reason"] != "derivative_suffix"]

        # diff_days が小さい順で試す
        cand_df = cand_df.sort_values(["diff_days", "symbol"]).reset_index(drop=True)

        for _, r in cand_df.iterrows():
            sym = r["symbol"]
            # 取得（期待日は expected_str）
            data = verifier.fetch_individual_eod(sym, expected_str)
            if not data:
                print(f"  ⚠️ 個別API: データ無し {sym}")
                continue

            # 1行DataFrameへ成形
            try:
                row = {
                    "date": pd.to_datetime(data.get("date") or data.get("Date") or expected_str),
                    "open": data.get("open") or data.get("Open"),
                    "high": data.get("high") or data.get("High"),
                    "low": data.get("low") or data.get("Low"),
                    "close": data.get("close") or data.get("Close"),
                    "adjusted_close": data.get("adjusted_close") or data.get("AdjClose"),
                    "volume": data.get("volume") or data.get("Volume"),
                }
                df_one = pd.DataFrame([row])
                # upsert
                cm.upsert_both(sym, df_one)
                patched += 1
                print(f"  ✅ 反映完了: {sym}")
            except Exception as e:
                print(f"  ⚠️ 反映失敗: {sym} err={e}")

            if patched >= patch_samples:
                break

        # 確認: 反映済みシンボルの最終日
        if patched > 0:
            print("\n[確認] 反映後の最終日チェック")
            checked = 0
            for _, r in cand_df.iterrows():
                sym = r["symbol"]
                try:
                    roll = cm.read(sym, "rolling")
                    if roll is None or getattr(roll, "empty", True):
                        continue
                    dcol = _detect_date_col(roll)
                    if not dcol:
                        continue
                    last = pd.to_datetime(roll[dcol], errors="coerce").max()
                    print(f"  - {sym}: last={pd.Timestamp(last).date()}")
                    checked += 1
                except Exception:
                    continue
                if checked >= patched:
                    break

    return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="rolling最終日ズレの理由を自動分類")
    parser.add_argument(
        "--patch-samples",
        type=int,
        default=0,
        help="サンプル件数だけ個別APIから取得してキャッシュへ反映（0で無効）",
    )
    parser.add_argument(
        "--include-derivatives",
        action="store_true",
        help="パッチ対象にデリバティブ疑い銘柄を含める",
    )

    args = parser.parse_args()
    return classify_and_report(args.patch_samples, args.include_derivatives)


if __name__ == "__main__":
    sys.exit(main())
