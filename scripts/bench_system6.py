"""System6 パフォーマンス比較ベンチマーク

目的:
  - original / optimized / ultra / fixed の4モードで
    1) データ前処理(prepare_data) 所要時間
    2) 候補生成(generate_candidates) 所要時間
  を同一シンボル集合・同一キャッシュ条件で比較し、結果を CSV 出力する。

設計方針:
  - 既存 CacheManager / rolling/base データをそのまま利用し外部 I/O 増やさない
  - ネットワーク呼び出しなし (既存キャッシュ前提)
  - 汚染防止のため乱数・時刻をログ以外へ影響させない
  - 再現用に: 実行時設定をメタ情報として JSON 併出力

出力:
  logs/perf/bench_system6_<YYYYMMDD_HHMMSS>.csv
  logs/perf/bench_system6_<YYYYMMDD_HHMMSS>.json (メタ: シンボル数/行数など)

CLI:
  python scripts/bench_system6.py --symbols-limit 120 --repeat 2 --ultra --fixed

制約:
  - ultra モードは data_dict を全モードで再利用不可 (前処理差異) のため個別計測
  - fixed はインジケータ再計算を基本スキップ
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from dataclasses import dataclass
import json
from pathlib import Path
from pathlib import Path as _Path
from statistics import mean
import sys
import time

import pandas as pd

# Ensure project root on sys.path when executed directly via `python scripts/bench_system6.py`
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.cache_manager import CacheManager, load_base_cache
from config.settings import get_settings
from strategies.system6_strategy import System6Strategy


@dataclass(slots=True)
class BenchResult:
    mode: str
    prepare_sec: float
    gen_sec: float
    symbols: int
    candidates: int
    phase_detail: dict[str, float]

    def to_row(self) -> list:
        return [
            self.mode,
            f"{self.prepare_sec:.3f}",
            f"{self.gen_sec:.3f}",
            self.symbols,
            self.candidates,
        ]


MODES = ["original", "optimized", "ultra", "fixed"]


_CACHE_MANAGER: CacheManager | None = None


def _get_cache_manager() -> CacheManager:
    global _CACHE_MANAGER
    if _CACHE_MANAGER is None:
        _CACHE_MANAGER = CacheManager(get_settings(create_dirs=False))
    return _CACHE_MANAGER


def _select_symbols(limit: int | None) -> list[str]:
    settings = get_settings(create_dirs=False)
    rolling_dir = Path(settings.cache.rolling_dir)
    full_dir = Path(settings.cache.full_dir)
    symbols: set[str] = set()
    exts = ("*.parquet", "*.csv", "*.feather")
    if rolling_dir.exists():
        for pattern in exts:
            for p in rolling_dir.glob(pattern):
                symbols.add(p.stem.upper())
    if not symbols and full_dir.exists():
        for pattern in exts:
            for p in full_dir.glob(pattern):
                symbols.add(p.stem.upper())
    out = sorted(symbols)
    if limit:
        out = out[:limit]
    return out


def _prepare_raw_dict(symbols: list[str]) -> dict[str, pd.DataFrame]:
    cache = _get_cache_manager()
    raw: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = load_base_cache(sym, rebuild_if_missing=False, cache_manager=cache)
        except Exception:
            df = None
        if df is None or getattr(df, "empty", True):
            continue
        raw[sym] = df
    return raw


def run_one(
    mode: str,
    raw_dict: dict[str, pd.DataFrame],
    ultra_flag: bool,
    timeout: float | None,
) -> BenchResult | None:
    start_all = time.perf_counter()
    phase_times: dict[str, float] = {}
    strat = System6Strategy(fixed_mode=(mode == "fixed"))

    def timed(label: str, fn):
        t0 = time.perf_counter()
        out = fn()
        phase_times[label] = time.perf_counter() - t0
        return out

    def expired() -> bool:
        return timeout is not None and (time.perf_counter() - start_all) > timeout

    if expired():
        return None

    def prep_fn():
        if mode == "original":
            return strat.prepare_data(
                raw_dict, enable_optimization=False, ultra_mode=False, fixed_mode=False
            )
        if mode == "optimized":
            return strat.prepare_data(
                raw_dict, enable_optimization=True, ultra_mode=False, fixed_mode=False
            )
        if mode == "ultra":
            return strat.prepare_data(
                raw_dict, enable_optimization=True, ultra_mode=True, fixed_mode=False
            )
        if mode == "fixed":
            return strat.prepare_data(
                raw_dict, enable_optimization=True, ultra_mode=False, fixed_mode=True
            )
        return None

    prepped = timed("prepare", prep_fn)
    if prepped is None:
        return None
    if expired():
        return None

    def gen_fn():
        return strat.generate_candidates(
            prepped, fixed_mode=(mode == "fixed"), ultra_mode=(mode == "ultra")
        )

    cands_dict, _ = timed("generate", gen_fn)
    if expired():
        return None

    candidates_total = sum(len(v) for v in cands_dict.values()) if cands_dict else 0
    return BenchResult(
        mode,
        prepare_sec=phase_times.get("prepare", 0.0),
        gen_sec=phase_times.get("generate", 0.0),
        symbols=len(raw_dict),
        candidates=candidates_total,
        phase_detail=phase_times,
    )


def main():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Benchmark System6 modes")
    parser.add_argument("--symbols-limit", type=int, default=120)
    parser.add_argument(
        "--symbols", nargs="*", help="明示的に利用するシンボル (指定時は探索をスキップ)"
    )
    parser.add_argument("--repeat", type=int, default=1, help="各モード繰り返し回数 (平均測定)")
    parser.add_argument(
        "--include",
        nargs="*",
        default=["original", "optimized", "ultra", "fixed"],
        help="対象モード制限",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="1モードあたり許容最大秒数 (超過でスキップ)",
    )
    args = parser.parse_args()

    settings = get_settings(create_dirs=False)
    # settings.outputs.logs_dir (新構成) / 旧互換 LOGS_DIR のどちらか
    logs_root = getattr(settings, "LOGS_DIR", None) or settings.outputs.logs_dir
    perf_dir = Path(logs_root) / "perf"
    perf_dir.mkdir(parents=True, exist_ok=True)

    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = _select_symbols(args.symbols_limit)
    if not symbols:
        print("No symbols found in cache. Exiting.")
        return
    print(f"[bench] Using {len(symbols)} symbols (limit={args.symbols_limit})")
    raw_dict = _prepare_raw_dict(symbols)
    print(f"[bench] Loaded {len(raw_dict)} symbols with data")

    modes = [m for m in MODES if m in args.include]
    results: list[BenchResult] = []
    meta_runs: dict[str, list[BenchResult]] = {}

    for mode in modes:
        run_list: list[BenchResult] = []
        for i in range(args.repeat):
            t_iter0 = time.perf_counter()
            res = run_one(mode, raw_dict, ultra_flag=(mode == "ultra"), timeout=args.timeout)
            print(f"[bench] mode={mode} iter={i} total={(time.perf_counter()-t_iter0):.2f}s")
            if res:
                run_list.append(res)
        if run_list:
            # 平均フェーズ詳細（現状 prepare / generate のみ）
            phase_detail_avg = {
                k: mean(r.phase_detail.get(k, 0.0) for r in run_list)
                for k in {ph for r in run_list for ph in r.phase_detail.keys()}
            }
            avg = BenchResult(
                mode,
                prepare_sec=mean(r.prepare_sec for r in run_list),
                gen_sec=mean(r.gen_sec for r in run_list),
                symbols=run_list[0].symbols,
                candidates=int(mean(r.candidates for r in run_list)),
                phase_detail=phase_detail_avg,
            )
            results.append(avg)
            meta_runs[mode] = run_list

    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = perf_dir / f"bench_system6_{ts}.csv"
    json_path = perf_dir / f"bench_system6_{ts}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "prepare_sec", "generate_sec", "symbols", "candidates"])
        for r in results:
            writer.writerow(r.to_row())

    runs_detail = {m: [dataclasses.asdict(r) for r in rr] for m, rr in meta_runs.items()}
    meta = {
        "timestamp": ts,
        "symbols_limit": args.symbols_limit,
        "repeat": args.repeat,
        "modes": modes,
        "raw_symbols": len(raw_dict),
        "raw_symbols_list_sample": symbols[:20],
        "runs_detail": runs_detail,
    }
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(meta, jf, ensure_ascii=False, indent=2)

    print("Saved:", csv_path)
    print("Saved:", json_path)
    for r in results:
        print(
            f"{r.mode:<9} prep={r.prepare_sec:6.2f}s gen={r.gen_sec:6.2f}s symbols={r.symbols:4d} cands={r.candidates}"  # noqa: E501
        )


if __name__ == "__main__":  # pragma: no cover
    main()
