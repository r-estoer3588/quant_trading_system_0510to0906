#!/usr/bin/env python3
"""
scripts/round_cache.py
既存の data_cache/full_backup と data_cache/rolling、および data_cache/base の
ファイルを設定 `cache.round_decimals` に従って丸めて上書きするユーティリティ。
デフォルトはドライラン。`--apply` を付けると上書きします。

使用例:
  python scripts/round_cache.py --apply
  python scripts/round_cache.py --decimals 2 --apply

注意: 実行前に back up を取ることを強く推奨します。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from common.cache_manager import (  # noqa: E402
    BASE_SUBDIR,
    _get_default_cache_manager,
    save_base_cache,
)
from config.settings import get_settings  # noqa: E402

try:
    from common.cache_format import round_dataframe  # type: ignore # noqa: E402
except ImportError:  # pragma: no cover - tests may stub cache_manager

    def round_dataframe(df: pd.DataFrame, decimals: int | None) -> pd.DataFrame:
        if decimals is None:
            return df
        try:
            decimals_int = int(decimals)
        except Exception:
            return df
        try:
            return df.copy().round(decimals_int)
        except Exception:
            try:
                return df.round(decimals_int)
            except Exception:
                return df


def find_cache_files(base_dir: Path) -> list[Path]:
    out = []
    for p in base_dir.glob("*.*"):
        if p.name.startswith("_"):
            continue
        if p.suffix.lower() not in {".csv", ".parquet", ".feather"}:
            continue
        out.append(p)
    return out


def read_any(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".feather":
            return pd.read_feather(path)
        # CSV の場合、先にヘッダだけ読み 'date' 列があるかを確認してから読み込む
        cols = pd.read_csv(path, nrows=0).columns
        if "date" in cols:
            return pd.read_csv(path, parse_dates=["date"])
        return pd.read_csv(path)
    except Exception as e:
        print(f"skip read {path}: {e}")
        return None


def write_any(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        df.to_parquet(tmp, index=False)
    elif path.suffix == ".feather":
        df.reset_index(drop=True).to_feather(tmp)
    else:
        df.to_csv(tmp, index=False)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decimals",
        type=int,
        default=None,
        help=("丸め桁数 (未指定で設定値を使う)"),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="上書きして適用する",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="上書き前に .bak を作成する",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=("full", "rolling", "base"),
        help="特定サブフォルダ: full|rolling|base を指定",
    )
    args = parser.parse_args(argv)

    settings = get_settings(create_dirs=False)
    if args.decimals is not None:
        cfg_round = args.decimals
    else:
        cfg_round = getattr(settings.cache, "round_decimals", None)
    if cfg_round is None:
        print(
            "round_decimals が設定されていません。--decimals で指定するか config/config.yaml を編集してください。",
        )
        return 1

    targets: list[Path] = []
    if args.only in (None, "full"):
        targets += find_cache_files(Path(settings.cache.full_dir))
    if args.only in (None, "rolling"):
        targets += find_cache_files(Path(settings.cache.rolling_dir))
    if args.only in (None, "base"):
        base_dir = Path(settings.DATA_CACHE_DIR) / BASE_SUBDIR
        if base_dir.exists():
            targets += find_cache_files(base_dir)
        else:
            print(f"base ディレクトリが存在しません: {base_dir}")

    print(f"対象ファイル数: {len(targets)} (decimals={cfg_round})")

    # default cache manager used for atomic rewrite (applies rounding + formatters)
    cm = _get_default_cache_manager()

    for p in targets:
        df = read_any(p)
        if df is None:
            continue
        # 丸め
        try:
            df2 = round_dataframe(df, cfg_round)
        except Exception as e:
            print(f"丸め失敗: {p} ({e})")
            continue

        # ドライランなら差分サイズを報告
        if not args.apply:
            try:
                orig_size = p.stat().st_size
                # 簡易見積: CSV に変換したときのバイト長を使う
                s = df2.to_csv(index=False)
                new_size = len(s.encode("utf-8"))
                print(f"{p.name}: {orig_size} -> approx {new_size} bytes")
            except Exception:
                print(f"{p.name}: processed (dry-run)")
        else:
            # Determine which cache this file belongs to and use CacheManager when possible
            try:
                full_dir = Path(settings.cache.full_dir).resolve()
            except Exception:
                full_dir = None
            try:
                rolling_dir = Path(settings.cache.rolling_dir).resolve()
            except Exception:
                rolling_dir = None
            base_dir = (Path(settings.DATA_CACHE_DIR) / BASE_SUBDIR).resolve()

            parent = p.parent.resolve()
            # optional backup
            if args.backup:
                try:
                    bak = p.with_suffix(p.suffix + ".bak")
                    bak.write_bytes(p.read_bytes())
                except Exception:
                    print(f"backup failed for {p}")

            try:
                if full_dir is not None and parent == full_dir:
                    symbol = p.stem
                    cm.write_atomic(df2, symbol, "full")
                    print(f"{p.name}: overwritten via CacheManager (full)")
                elif rolling_dir is not None and parent == rolling_dir:
                    symbol = p.stem
                    cm.write_atomic(df2, symbol, "rolling")
                    print(f"{p.name}: overwritten via CacheManager (rolling)")
                elif parent == base_dir:
                    # base cache: use save_base_cache which writes CSV
                    symbol = p.stem
                    save_base_cache(symbol, df2)
                    print(f"{p.name}: overwritten via save_base_cache (base)")
                else:
                    # fallback: direct write
                    write_any(p, df2)
                    print(f"{p.name}: overwritten (fallback)")
            except Exception:
                print(f"failed overwrite {p}")

    print("完了")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
