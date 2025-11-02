from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import yaml  # type: ignore

from common import broker_alpaca as ba

"""Alpacaのポジションに対しトレーリングストップを設定/更新するスクリプト."""


def update_trailing_stops(
    *,
    trail_percent: float | None = 25.0,
    symbol_trail_pct: Mapping[str, float] | None = None,
    paper: bool = True,
) -> None:
    """ポジションごとにトレーリングストップ注文を発行する.

    ``symbol_trail_pct`` が与えられた場合はシンボルごとの割合を優先し、
    指定がないシンボルには ``trail_percent`` を適用する。
    いずれの値も指定されない場合は何もしない。
    """

    client = ba.get_client(paper=paper)
    ba.cancel_all_orders(client)
    positions = client.get_all_positions()
    for pos in positions:
        symbol = getattr(pos, "symbol", None)
        if symbol is None:
            continue
        pct = None
        if symbol_trail_pct and symbol in symbol_trail_pct:
            pct = symbol_trail_pct[symbol]
        elif trail_percent is not None:
            pct = trail_percent
        if pct is None:
            continue

        qty = abs(int(float(getattr(pos, "qty", 0))))
        if qty <= 0:
            continue
        side = "sell" if getattr(pos, "side", "long") == "long" else "buy"
        ba.submit_order(
            client,
            symbol,
            qty,
            side=side,
            order_type="trailing_stop",
            trail_percent=pct,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpacaのストップ注文を更新")
    parser.add_argument(
        "--trail-percent",
        type=float,
        default=25.0,
        help="全シンボルに一律で適用するトレーリングストップ割合(%)",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        help="シンボルごとのトレーリングストップ割合を定義した JSON/YAML ファイル",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="ライブ口座を利用 (デフォルトはPaper)",
    )
    args = parser.parse_args()

    mapping: Mapping[str, float] | None = None
    if args.mapping:
        with open(args.mapping, encoding="utf-8") as f:
            if args.mapping.suffix.lower() in {".yaml", ".yml"}:
                mapping = yaml.safe_load(f) or {}
            else:
                mapping = json.load(f)

    update_trailing_stops(
        trail_percent=args.trail_percent,
        symbol_trail_pct=mapping,
        paper=(not args.live),
    )


if __name__ == "__main__":
    main()
