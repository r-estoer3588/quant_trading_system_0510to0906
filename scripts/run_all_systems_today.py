from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from config.settings import get_settings
from common.utils import get_cached_data
from common.utils_spy import get_spy_with_indicators, get_latest_nyse_trading_day
from common.today_signals import LONG_SYSTEMS, SHORT_SYSTEMS
from common import broker_alpaca as ba
from common.notifier import Notifier

# strategies
from strategies.system1_strategy import System1Strategy
from strategies.system2_strategy import System2Strategy
from strategies.system3_strategy import System3Strategy
from strategies.system4_strategy import System4Strategy
from strategies.system5_strategy import System5Strategy
from strategies.system6_strategy import System6Strategy
from strategies.system7_strategy import System7Strategy
from common.signal_merge import Signal, merge_signals


def _log(msg: str):
    print(msg, flush=True)


def _load_raw_data(symbols: List[str], cache_dir: str) -> Dict[str, pd.DataFrame]:
    data: Dict[str, pd.DataFrame] = {}
    total = len(symbols)
    for i, sym in enumerate(symbols, 1):
        df = get_cached_data(sym, folder=cache_dir)
        if df is None or df.empty:
            continue
        data[sym] = df
        if i % 50 == 0 or i == total:
            _log(f"📦 キャッシュ読み込み {i}/{total}件 完了")
    return data


def _asc_by_score_key(score_key: str | None) -> bool:
    return bool(score_key and score_key.upper() in {"RSI4"})


def _amount_pick(
    per_system: Dict[str, pd.DataFrame],
    strategies: Dict[str, object],
    total_budget: float,
    weights: Dict[str, float],
    side: str,
) -> pd.DataFrame:
    """資金配分に基づいて候補を採用。shares と position_value を付与して返す。"""
    chosen = []
    chosen_symbols = set()

    # システムごとの割当予算
    budgets = {name: float(total_budget) * float(weights.get(name, 0.0)) for name in weights}
    remaining = budgets.copy()

    # システムごとにスコア順で採用
    # 複数周回して1件ずつ拾う（偏りを軽減）
    still = True
    while still:
        still = False
        for name in weights.keys():
            df = per_system.get(name, pd.DataFrame())
            if df is None or df.empty or remaining.get(name, 0.0) <= 0.0:
                continue
            stg = strategies[name]
            # 順に探索
            for idx, row in df.iterrows():
                sym = row["symbol"]
                if sym in chosen_symbols:
                    continue
                entry = float(row["entry_price"]) if not pd.isna(row.get("entry_price")) else None
                stop = float(row["stop_price"]) if not pd.isna(row.get("stop_price")) else None
                if not entry or not stop or entry <= 0:
                    continue

                # 望ましい枚数（全システム割当基準）
                try:
                    desired_shares = stg.calculate_position_size(
                        budgets[name], entry, stop,
                        risk_pct=float(getattr(stg, "config", {}).get("risk_pct", 0.02)),
                        max_pct=float(getattr(stg, "config", {}).get("max_pct", 0.10)),
                    )
                except Exception:
                    desired_shares = 0
                if desired_shares <= 0:
                    continue

                # 予算内に収まるよう調整
                max_by_cash = int(remaining[name] // abs(entry))
                shares = min(desired_shares, max_by_cash)
                if shares <= 0:
                    continue
                position_value = shares * abs(entry)
                if position_value <= 0:
                    continue

                # 採用
                rec = row.to_dict()
                rec["shares"] = int(shares)
                rec["position_value"] = float(round(position_value, 2))
                rec["system_budget"] = float(round(budgets[name], 2))
                rec["remaining_after"] = float(round(remaining[name] - position_value, 2))
                chosen.append(rec)
                chosen_symbols.add(sym)
                remaining[name] -= position_value
                still = True
                break  # 1件ずつ拾って次のシステムへ

    if not chosen:
        return pd.DataFrame()
    out = pd.DataFrame(chosen)
    out["side"] = side
    return out


def _submit_orders(
    final_df: pd.DataFrame,
    *,
    paper: bool = True,
    order_type: str = "market",
    tif: str = "GTC",
    retries: int = 2,
    delay: float = 0.5,
) -> pd.DataFrame:
    """final_df をもとに Alpaca へ注文送信（shares 必須）。
    返り値: 実行結果の DataFrame（order_id/status/error を含む）
    """
    if final_df is None or final_df.empty:
        _log("(submit) final_df is empty; skip")
        return pd.DataFrame()
    if "shares" not in final_df.columns:
        _log("(submit) shares 列がありません。資金配分モードで実行してください。")
        return pd.DataFrame()
    try:
        client = ba.get_client(paper=paper)
    except Exception as e:
        _log(f"(submit) Alpaca接続エラー: {e}")
        return pd.DataFrame()

    results = []
    for _, r in final_df.iterrows():
        sym = str(r.get("symbol"))
        qty = int(r.get("shares") or 0)
        side = "buy" if str(r.get("side")).lower() == "long" else "sell"
        if not sym or qty <= 0:
            continue
        limit_price = float(r.get("entry_price")) if order_type == "limit" else None
        try:
            order = ba.submit_order_with_retry(
                client,
                sym,
                qty,
                side=side,
                order_type=order_type,
                limit_price=limit_price,
                time_in_force=tif,
                retries=max(0, int(retries)),
                backoff_seconds=max(0.0, float(delay)),
                rate_limit_seconds=max(0.0, float(delay)),
                log_callback=_log,
            )
            results.append({
                "symbol": sym,
                "side": side,
                "qty": qty,
                "order_id": getattr(order, "id", None),
                "status": getattr(order, "status", None),
            })
        except Exception as e:
            results.append({
                "symbol": sym,
                "side": side,
                "qty": qty,
                "error": str(e),
            })
    if results:
        out = pd.DataFrame(results)
        _log("\n=== Alpaca submission results ===")
        _log(out.to_string(index=False))
        notifier = Notifier(platform="auto")
        notifier.send_trade_report("integrated", results)
        return out
    return pd.DataFrame()


def _apply_filters(df: pd.DataFrame, *, only_long: bool = False, only_short: bool = False, top_per_system: int = 0) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "side" in out.columns:
        if only_long and not only_short:
            out = out[out["side"].str.lower() == "long"]
        if only_short and not only_long:
            out = out[out["side"].str.lower() == "short"]
    if top_per_system and top_per_system > 0 and "system" in out.columns:
        by = ["system"] + (["side"] if "side" in out.columns else [])
        out = out.groupby(by, as_index=False, group_keys=False).head(int(top_per_system))
    return out


def compute_today_signals(
    symbols: List[str] | None,
    *,
    slots_long: int | None = None,
    slots_short: int | None = None,
    capital_long: float | None = None,
    capital_short: float | None = None,
    save_csv: bool = False,
    notify: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """当日シグナル抽出＋配分の本体。

    戻り値: (final_df, per_system_df_dict)
    """
    settings = get_settings(create_dirs=True)
    cache_dir = str(settings.DATA_CACHE_DIR)
    signals_dir = Path(settings.outputs.signals_dir)
    signals_dir.mkdir(parents=True, exist_ok=True)

    # 最新営業日（NYSE）
    today = get_latest_nyse_trading_day().normalize()
    _log(f"📅 最新営業日（NYSE）: {today.date()}")

    # シンボル決定
    if symbols and len(symbols) > 0:
        symbols = [s.upper() for s in symbols]
    else:
        from common.universe import build_universe_from_cache, load_universe_file

        universe = load_universe_file()
        if not universe:
            universe = build_universe_from_cache(limit=None)
        symbols = [s.upper() for s in universe]
        if not symbols:
            try:
                files = list(Path(cache_dir).glob("*.csv"))
                primaries = [p.stem for p in files if p.stem.upper() == "SPY"]
                others = sorted({p.stem for p in files if len(p.stem) <= 5})[:200]
                symbols = list(dict.fromkeys(primaries + others))
            except Exception:
                symbols = []
    if "SPY" not in symbols:
        symbols.append("SPY")

    _log(f"🎯 対象シンボル数: {len(symbols)}（例: {', '.join(symbols[:10])}{'...' if len(symbols)>10 else ''}）")

    # データ読み込み
    raw_data = _load_raw_data(symbols, cache_dir)
    if "SPY" not in raw_data:
        _log("⚠️ SPY が data_cache に見つかりません。SPY.csv を用意してください。")
        spy_df = None
    else:
        spy_df = get_spy_with_indicators(raw_data["SPY"])  # type: ignore[arg-type]

    # ストラテジ初期化
    strategy_objs = [
        System1Strategy(), System2Strategy(), System3Strategy(),
        System4Strategy(), System5Strategy(), System6Strategy(), System7Strategy(),
    ]
    strategies = {getattr(s, "SYSTEM_NAME", "").lower(): s for s in strategy_objs}

    # 当日シグナル収集
    per_system: Dict[str, pd.DataFrame] = {}
    for name, stg in strategies.items():
        if name == "system4" and spy_df is None:
            _log("⚠️ System4 は SPY 指標が必要ですが SPY データがありません。スキップします。")
            per_system[name] = pd.DataFrame()
            continue
        base = {"SPY": raw_data.get("SPY")} if name == "system7" else raw_data
        _log(f"🔎 {name}: シグナル抽出を開始")
        df = stg.get_today_signals(base, market_df=spy_df, today=today)
        if not df.empty:
            asc = _asc_by_score_key(df["score_key"].iloc[0] if "score_key" in df.columns and len(df) else None)
            df = df.sort_values("score", ascending=asc, na_position="last").reset_index(drop=True)
        per_system[name] = df
        _log(f"✅ {name}: {len(df)} 件")

    # 1) 枠配分（スロット）モード or 2) 金額配分モード
    long_alloc = {"system1": 0.25, "system3": 0.25, "system4": 0.25, "system5": 0.25}
    short_alloc = {"system2": 0.40, "system6": 0.40, "system7": 0.20}

    if capital_long is None and capital_short is None:
        # 旧スロット方式（後方互換）
        max_pos = int(settings.risk.max_positions)
        slots_long = slots_long if slots_long is not None else max_pos
        slots_short = slots_short if slots_short is not None else max_pos

        def _distribute_slots(weights: Dict[str, float], total_slots: int, counts: Dict[str, int]) -> Dict[str, int]:
            base = {k: int(total_slots * weights.get(k, 0.0)) for k in weights}
            for k in list(base.keys()):
                if counts.get(k, 0) <= 0:
                    base[k] = 0
                elif base[k] == 0:
                    base[k] = 1
            used = sum(base.values())
            remain = max(0, total_slots - used)
            if remain > 0:
                order = sorted(weights.keys(), key=lambda k: (counts.get(k, 0), weights.get(k, 0.0)), reverse=True)
                idx = 0
                while remain > 0 and order:
                    k = order[idx % len(order)]
                    if counts.get(k, 0) > base.get(k, 0):
                        base[k] += 1
                        remain -= 1
                    idx += 1
                    if idx > 10000:
                        break
            for k in list(base.keys()):
                base[k] = min(base[k], counts.get(k, 0))
            return base

        long_counts = {k: len(per_system.get(k, pd.DataFrame())) for k in long_alloc}
        short_counts = {k: len(per_system.get(k, pd.DataFrame())) for k in short_alloc}
        long_slots = _distribute_slots(long_alloc, slots_long, long_counts)
        short_slots = _distribute_slots(short_alloc, slots_short, short_counts)

        chosen_frames: List[pd.DataFrame] = []
        for name, slot in {**long_slots, **short_slots}.items():
            df = per_system.get(name, pd.DataFrame())
            if df is None or df.empty or slot <= 0:
                continue
            take = df.head(slot).copy()
            take["alloc_weight"] = (long_alloc.get(name) or short_alloc.get(name) or 0.0)
            chosen_frames.append(take)
        final_df = pd.concat(chosen_frames, ignore_index=True) if chosen_frames else pd.DataFrame()
    else:
        # 金額配分モード
        if capital_long is None:
            capital_long = float(get_settings(create_dirs=False).backtest.initial_capital)
        if capital_short is None:
            capital_short = float(get_settings(create_dirs=False).backtest.initial_capital)

        strategies_map = {k: v for k, v in strategies.items()}
        long_df = _amount_pick(
            {k: per_system.get(k, pd.DataFrame()) for k in long_alloc},
            strategies_map,
            float(capital_long),
            long_alloc,
            side="long",
        )
        short_df = _amount_pick(
            {k: per_system.get(k, pd.DataFrame()) for k in short_alloc},
            strategies_map,
            float(capital_short),
            short_alloc,
            side="short",
        )
        parts = [df for df in [long_df, short_df] if df is not None and not df.empty]
        final_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    if not final_df.empty:
        sort_cols = [c for c in ["side", "system", "score"] if c in final_df.columns]
        final_df = final_df.sort_values(sort_cols, ascending=[True, True, True][: len(sort_cols)]).reset_index(drop=True)

        if notify:
            try:
                from tools.notify_signals import send_signal_notification

                send_signal_notification(final_df)
            except Exception:
                _log("⚠️ 通知に失敗しました。")

    # CSV 保存（任意）
    if save_csv and not final_df.empty:
        date_str = today.strftime("%Y-%m-%d")
        out_all = signals_dir / f"signals_final_{date_str}.csv"
        final_df.to_csv(out_all, index=False)
        # システム別
        for name, df in per_system.items():
            if df is None or df.empty:
                continue
            out = signals_dir / f"signals_{name}_{date_str}.csv"
            df.to_csv(out, index=False)
        _log(f"💾 保存: {signals_dir} にCSVを書き出しました")

    return final_df, per_system


def main():
    parser = argparse.ArgumentParser(description="全システム当日シグナル抽出・集約")
    parser.add_argument("--symbols", nargs="*", help="対象シンボル。未指定なら設定のauto_tickersを使用")
    parser.add_argument("--slots-long", type=int, default=None, help="買いサイドの最大採用数（スロット方式）")
    parser.add_argument("--slots-short", type=int, default=None, help="売りサイドの最大採用数（スロット方式）")
    parser.add_argument("--capital-long", type=float, default=None, help="買いサイド予算（ドル）。指定時は金額配分モード")
    parser.add_argument("--capital-short", type=float, default=None, help="売りサイド予算（ドル）。指定時は金額配分モード")
    parser.add_argument("--save-csv", action="store_true", help="signalsディレクトリにCSVを保存する")
    # Alpaca 自動発注オプション
    parser.add_argument("--alpaca-submit", action="store_true", help="Alpaca に自動発注（shares 必須）")
    parser.add_argument("--order-type", choices=["market", "limit"], default="market", help="注文種別")
    parser.add_argument("--tif", choices=["GTC", "DAY"], default="GTC", help="Time In Force")
    parser.add_argument("--live", action="store_true", help="ライブ口座で発注（デフォルトはPaper）")
    args = parser.parse_args()

    final_df, per_system = compute_today_signals(
        args.symbols,
        slots_long=args.slots_long,
        slots_short=args.slots_short,
        capital_long=args.capital_long,
        capital_short=args.capital_short,
        save_csv=args.save_csv,
    )

    if final_df.empty:
        _log("📭 本日の最終候補はありません。")
    else:
        _log("\n=== 最終候補（推奨） ===")
        cols = [
            "symbol",
            "system",
            "side",
            "signal_type",
            "entry_date",
            "entry_price",
            "stop_price",
            "shares",
            "position_value",
            "score_key",
            "score",
        ]
        show = [c for c in cols if c in final_df.columns]
        _log(final_df[show].to_string(index=False))
        signals_for_merge = [
            Signal(
                system_id=int(str(r.get("system")).replace("system", "") or 0),
                symbol=str(r.get("symbol")),
                side="BUY" if str(r.get("side")).lower() == "long" else "SELL",
                strength=float(r.get("score") or 0.0),
                meta={},
            )
            for _, r in final_df.iterrows()
        ]
        merge_signals([signals_for_merge], portfolio_state={}, market_state={})
        if args.alpaca_submit:
            _submit_orders(final_df, paper=(not args.live), order_type=args.order_type, tif=args.tif)


if __name__ == "__main__":
    main()
