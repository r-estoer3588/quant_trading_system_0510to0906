from __future__ import annotations

import argparse

from common.universe import build_universe_from_cache, save_universe_file


def main():
    p = argparse.ArgumentParser(description="data_cache からユニバースを自動生成し保存")
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--min-dollar-vol", type=float, default=25_000_000.0)
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--out", type=str, default=None, help="出力パス（既定: data/universe_auto.txt）")
    args = p.parse_args()

    syms = build_universe_from_cache(
        min_price=args.min_price,
        min_dollar_vol=args.min_dollar_vol,
        limit=args.limit,
    )
    path = save_universe_file(syms, path=args.out)
    print(f"✅ {len(syms)}銘柄を保存: {path}")


if __name__ == "__main__":
    main()

