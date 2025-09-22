"""CLI ツール: 共有指標を前計算し CSV として書き出す。

当日シグナル実行の前に ATR/SMA/RSI などの共有指標が必要になる場合、
このスクリプトを使って事前計算と CSV への書き出しを行える。
既存の ``precompute_shared_indicators`` を利用し、計算結果は
``data_cache/signals/shared_indicators``（既定）に保存される。
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from common.cache_manager import round_dataframe
from common.data_loader import load_price
from common.indicators_precompute import precompute_shared_indicators
from common.universe import load_universe_file
from config.settings import get_settings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Precompute shared indicators (ATR/SMA/RSI 等) and export them to CSV")
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="処理対象の銘柄コード。指定が無ければユニバースファイルを参照。",
    )
    parser.add_argument(
        "--universe-file",
        type=Path,
        help="ユニバースファイルのパス。symbols が未指定のときに使用。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "CSV を保存するディレクトリ。未指定の場合は <signals_dir>/shared_indicators を利用。"
        ),
    )
    parser.add_argument(
        "--cache-profile",
        choices=("full", "rolling"),
        default="full",
        help="OHLCV データの読み込み元キャッシュ種別 (default: full)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="並列実行を強制的に無効化する。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="並列実行時の最大ワーカー数。未指定時は設定値を使用。",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        help="全銘柄を結合した単一 CSV の出力先 (任意)。",
    )
    return parser.parse_args()


def _unique_upper_symbols(symbols: Iterable[str]) -> list[str]:
    ordered = []
    for sym in symbols:
        key = sym.strip().upper()
        if not key:
            continue
        if key not in ordered:
            ordered.append(key)
    return ordered


def _prepare_output_frame(df: pd.DataFrame) -> pd.DataFrame:
    """CSV 書き出し向けに列を整形する。"""

    frame = df.copy()

    if "Date" in frame.columns:
        date_series = pd.to_datetime(frame["Date"], errors="coerce")
    elif "date" in frame.columns:
        date_series = pd.to_datetime(frame["date"], errors="coerce")
    else:
        date_series = pd.to_datetime(frame.index, errors="coerce")

    frame["Date"] = date_series.dt.normalize()
    frame = frame.dropna(subset=["Date"])
    frame = frame.sort_values("Date")

    if "date" in frame.columns:
        frame = frame.drop(columns=["date"])

    preferred_order = [
        "Date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
    ordered_cols: list[str] = []
    for col in preferred_order:
        if col in frame.columns and col not in ordered_cols:
            ordered_cols.append(col)
    for col in frame.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)
    frame = frame.loc[:, ordered_cols]

    return frame


def main() -> None:
    args = _parse_args()
    settings = get_settings(create_dirs=True)

    if args.symbols:
        symbols = _unique_upper_symbols(args.symbols)
    else:
        universe_path = args.universe_file
        if universe_path is not None:
            symbols = load_universe_file(str(universe_path))
        else:
            symbols = load_universe_file()

    if not symbols:
        print("🛑 銘柄リストが取得できませんでした。symbols/universe の設定を確認してください。")
        return

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(settings.outputs.signals_dir) / "shared_indicators"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    basic_data: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for sym in symbols:
        try:
            df = load_price(sym, cache_profile=args.cache_profile)
        except Exception:
            df = pd.DataFrame()
        if df is None or getattr(df, "empty", True):
            skipped.append(sym)
            continue
        work = df.copy()
        if "date" in work.columns and "Date" not in work.columns:
            work["Date"] = pd.to_datetime(work["date"], errors="coerce")
        basic_data[sym] = work

    if not basic_data:
        print("🛑 指標計算用のデータが1件も取得できませんでした。")
        if skipped:
            print("  スキップ銘柄:", ", ".join(skipped))
        return

    use_parallel = not args.no_parallel
    if use_parallel:
        if args.max_workers is not None:
            max_workers = max(1, int(args.max_workers))
        else:
            max_workers = int(getattr(settings, "THREADS_DEFAULT", 12))
    else:
        max_workers = None

    enriched = precompute_shared_indicators(
        basic_data,
        log=lambda msg: print(msg),
        parallel=use_parallel,
        max_workers=max_workers if use_parallel else None,
    )

    combined_frames: list[pd.DataFrame] = []
    written = 0
    for sym, df in enriched.items():
        if df is None or getattr(df, "empty", True):
            continue
        prepared = _prepare_output_frame(df)
        if prepared.empty:
            continue
        prepared.insert(0, "Symbol", sym)
        try:
            round_dec = getattr(settings.cache, "round_decimals", None)
        except Exception:
            round_dec = None
        try:
            prepared_to_write = round_dataframe(prepared, round_dec)
        except Exception:
            prepared_to_write = prepared
        prepared_to_write.to_csv(output_dir / f"{sym}.csv", index=False)
        combined_frames.append(prepared)
        written += 1

    if args.combined_output and combined_frames:
        combined_path = Path(args.combined_output)
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            round_dec = getattr(settings.cache, "round_decimals", None)
        except Exception:
            round_dec = None
        try:
            combined = pd.concat(combined_frames, ignore_index=True)
            combined = round_dataframe(combined, round_dec)
        except Exception:
            combined = pd.concat(combined_frames, ignore_index=True)
        combined.to_csv(combined_path, index=False)
        print(f"📦 結合 CSV を出力しました: {combined_path}")

    if skipped:
        print("⚠️ データ不足によりスキップした銘柄:", ", ".join(skipped))

    print(f"✅ 共有指標のCSV出力が完了しました: {written} 銘柄 (保存先: {output_dir.resolve()})")


if __name__ == "__main__":  # pragma: no cover - CLI エントリポイント
    main()
