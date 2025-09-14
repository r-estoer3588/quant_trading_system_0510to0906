"""Alpaca注文関連の共通ユーティリティとミックスイン。

このモジュールは alpaca-py に直接依存せず、すべて `common.broker_alpaca`
を経由して呼び出します（テスト環境で SDK が無くても読み込めるため）。
"""

from __future__ import annotations

import logging
from typing import Any
from pathlib import Path
import json

import pandas as pd

from common import broker_alpaca as ba
from common.notifier import Notifier
from common.position_age import load_entry_dates, save_entry_dates

if False:  # typing guard
    pass

logger = logging.getLogger(__name__)


class AlpacaOrderMixin:
    """
    Strategy 用の軽量ミックスイン。

    - `__init__` を持たないことで、`StrategyBase` の初期化順序（MRO）に影響しないようにする。
    - 具体的な発注処理は `common.broker_alpaca` の関数へ委譲する。

    既存コード（AlpacaOrderManager）と同等のメソッド名を提供し、
    互換 API を維持する。
    """

    # 既存の Manager と同名・同シグネチャのメソッドを提供
    def submit_order(
        self,
        client: Any,
        symbol: str,
        qty: int,
        side: str = "buy",
        order_type: str = "market",
        *,
        limit_price: float | None = None,
        stop_price: float | None = None,
        take_profit: float | None = None,
        stop_loss: float | None = None,
        trail_percent: float | None = None,
        log_callback=None,
    ):
        # TimeInForce は Manager と同様に GTC をデフォルト使用
        return ba.submit_order(
            client,
            symbol,
            int(qty),
            side=side,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            trail_percent=trail_percent,
            time_in_force="GTC",
            log_callback=log_callback,
        )

    def log_orders_positions(self, client: Any, log_callback=None):
        orders, positions = ba.log_orders_positions(client)
        if log_callback:
            try:
                for o in orders:
                    oid = getattr(o, "id", None)
                    sym = getattr(o, "symbol", None)
                    side = getattr(o, "side", None)
                    status = getattr(o, "status", None)
                    filled = getattr(o, "filled_qty", None)
                    log_callback(f"Order {oid} {sym} {side} {status} filled={filled}")
                for p in positions:
                    psym = getattr(p, "symbol", None)
                    pqty = getattr(p, "qty", None)
                    avg = getattr(p, "avg_entry_price", None)
                    log_callback(f"Position {psym} qty={pqty} avg_entry={avg}")
            except Exception:
                pass

    def subscribe_order_updates(self, log_callback=None):
        """
        可能なら環境変数からクライアントを生成して購読を開始（ブロッキング）。
        既存の Manager と同名メソッドを提供するためのラッパー。
        """
        try:
            client = ba.get_client(paper=None)
        except Exception as err:
            # クライアントが作れない場合はエラーを明示
            raise RuntimeError(
                "Alpaca クライアントを初期化できませんでした（API キー要確認）。",
            ) from err
        # ブロッキング実行（戻り値は None）
        return ba.subscribe_order_updates(client, log_callback=log_callback)


__all__ = [
    "AlpacaOrderMixin",
    "submit_orders_df",
    "submit_exit_orders_df",
]


# --- lightweight symbol<->system mapping persistence ---
_SYMBOL_SYSTEM_MAP_PATH = Path("data/symbol_system_map.json")


def _load_symbol_system_map() -> dict[str, str]:
    try:
        return json.loads(_SYMBOL_SYSTEM_MAP_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_symbol_system_map(mapping: dict[str, str]) -> None:
    try:
        _SYMBOL_SYSTEM_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SYMBOL_SYSTEM_MAP_PATH.write_text(
            json.dumps(mapping, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


def submit_orders_df(
    final_df: pd.DataFrame,
    *,
    paper: bool = True,
    order_type: str | None = None,
    system_order_type: dict[str, str] | None = None,
    tif: str = "GTC",
    retries: int = 2,
    delay: float = 0.5,
    log_callback: Any | None = None,
    notify: bool = False,
) -> pd.DataFrame:
    """DataFrameからAlpacaへ注文を一括送信する共通ヘルパー。

        - `final_df` は列に少なくとも `symbol`, `system`, `side`,
            `shares`, `entry_date`, `entry_price` を想定。
        - `order_type` が指定されれば全件に適用。未指定なら
            `system_order_type` を参照。`system_order_type` も無い場合は
            既定マップを使用（system1/3/4/5: market, system2/6/7: limit）。
    - 重複送信防止のため (symbol, system, entry_date) でユニーク化。
    - 返り値は結果の DataFrame（order_id/status/error を含む）。
    """
    if final_df is None or final_df.empty:
        return pd.DataFrame()
    if "shares" not in final_df.columns:
        # shares が無い場合はスキップ
        return pd.DataFrame()

    try:
        client = ba.get_client(paper=paper)
    except Exception:
        return pd.DataFrame()

    # 既定のシステム別オーダータイプ
    default_sys_map = {
        "system1": "market",
        "system3": "market",
        "system4": "market",
        "system5": "market",
        "system2": "limit",
        "system6": "limit",
        "system7": "limit",
    }
    sys_map = (system_order_type or {}) | default_sys_map  # defaultを下敷き

    # 重複注文防止
    unique: dict[tuple[str, str, str], Any] = {}
    for _, r in final_df.iterrows():
        key = (
            str(r.get("symbol")),
            str(r.get("system")).lower(),
            str(r.get("entry_date")),
        )
        if key in unique:
            continue
        unique[key] = r

    # load existing symbol->system mapping (for positions dashboard / exit mapping)
    sys_map_store = _load_symbol_system_map()

    results: list[dict[str, Any]] = []
    for (_sym, _sys, _dt), r in unique.items():
        sym = str(r.get("symbol"))
        qty = int(r.get("shares") or 0)
        side = "buy" if str(r.get("side")).lower() == "long" else "sell"
        system = str(r.get("system")).lower()
        if not sym or qty <= 0:
            continue
        ot = order_type or sys_map.get(system, "market")
        entry_price_raw = r.get("entry_price")
        # limit の場合のみ limit_price を推定
        if ot == "limit" and entry_price_raw not in (None, ""):
            try:
                limit_price = float(entry_price_raw)
            except (TypeError, ValueError):
                limit_price = None
        else:
            limit_price = None
        # 表示用価格
        price_val = None
        try:
            if entry_price_raw not in (None, ""):
                price_val = float(entry_price_raw)
        except (TypeError, ValueError):
            price_val = None
        if limit_price is not None:
            price_val = limit_price
        try:
            order = ba.submit_order_with_retry(
                client,
                sym,
                qty,
                side=side,
                order_type=ot,
                limit_price=limit_price,
                time_in_force=tif,
                retries=max(0, int(retries)),
                backoff_seconds=max(0.0, float(delay)),
                rate_limit_seconds=max(0.0, float(delay)),
                log_callback=log_callback,
            )
            results.append(
                {
                    "symbol": sym,
                    "side": side,
                    "qty": qty,
                    "price": price_val,
                    "order_id": getattr(order, "id", None),
                    "status": getattr(order, "status", None),
                    "system": system,
                    "order_type": ot,
                    "time_in_force": tif,
                    "entry_date": r.get("entry_date"),
                }
            )
        except Exception as e:  # noqa: BLE001
            results.append(
                {
                    "symbol": sym,
                    "side": side,
                    "qty": qty,
                    "price": price_val,
                    "error": str(e),
                    "system": system,
                    "order_type": ot,
                    "time_in_force": tif,
                    "entry_date": r.get("entry_date"),
                }
            )

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    # エントリー日記録とシンボル<->システムの更新
    try:
        entry_map = load_entry_dates()
        for _, row in out.iterrows():
            sym = str(row.get("symbol"))
            # final_df はエントリー用途のみを想定しているため、
            # ロング・ショート双方でエントリー日を保存する
            if row.get("entry_date"):
                entry_map[sym] = str(row.get("entry_date"))
            # シンボル→システムも記録（ダッシュボード/exit 判定で使用）
            sys_val = str(row.get("system") or "").lower()
            if sym and sys_val:
                sys_map_store[sym] = sys_val
        save_entry_dates(entry_map)
        _save_symbol_system_map(sys_map_store)
    except Exception:
        pass

    # 通知（任意）
    if notify:
        try:
            Notifier(platform="auto").send_trade_report("integrated", results)
        except Exception:
            pass

    return out


def submit_exit_orders_df(
    exit_df: pd.DataFrame,
    *,
    paper: bool = True,
    tif: str = "CLS",
    retries: int = 2,
    delay: float = 0.5,
    log_callback: Any | None = None,
    notify: bool = True,
) -> pd.DataFrame:
    """Submit exit orders (close existing positions).

    Expected columns in ``exit_df``:
      - ``symbol``: ticker
      - ``qty``: absolute quantity to close
      - ``position_side``: "long" | "short" (to infer order side)
      - ``system``: system name like "system3" (optional but recommended)
      - ``when``: one of "today_close" | "tomorrow_close" | "tomorrow_open"

    Only "today_close" will be sent immediately as Market-On-Close (MOC) via
    time_in_force="cls". Others are returned as planned but not submitted.
    """
    if exit_df is None or exit_df.empty:
        return pd.DataFrame()
    try:
        client = ba.get_client(paper=paper)
    except Exception:
        return pd.DataFrame()

    # Persisted mappings for cleanup after successful exits
    entry_map = load_entry_dates()
    sys_map_store = _load_symbol_system_map()

    results: list[dict[str, Any]] = []
    for _, r in exit_df.iterrows():
        sym = str(r.get("symbol"))
        qty = int(r.get("qty") or 0)
        pos_side = str(r.get("position_side", "")).lower()
        system = str(r.get("system", "")).lower()
        when = str(r.get("when", "")).lower()
        if not sym or qty <= 0:
            continue
        if when != "today_close":
            results.append(
                {
                    "symbol": sym,
                    "side": "planned",
                    "qty": qty,
                    "when": when,
                    "system": system,
                    "status": "planned",
                }
            )
            continue
        order_side = "sell" if pos_side == "long" else "buy"
        try:
            order = ba.submit_order_with_retry(
                client,
                sym,
                qty,
                side=order_side,
                order_type="market",
                time_in_force=(tif or "CLS").upper(),
                retries=max(0, int(retries)),
                backoff_seconds=max(0.0, float(delay)),
                rate_limit_seconds=max(0.0, float(delay)),
                log_callback=log_callback,
            )
            results.append(
                {
                    "symbol": sym,
                    "side": order_side,
                    "qty": qty,
                    "when": when,
                    "system": system,
                    "order_id": getattr(order, "id", None),
                    "status": getattr(order, "status", None),
                }
            )
            # cleanup local tracking for the symbol upon exit
            entry_map.pop(sym, None)
            sys_map_store.pop(sym, None)
        except Exception as e:  # noqa: BLE001
            results.append(
                {
                    "symbol": sym,
                    "side": order_side,
                    "qty": qty,
                    "when": when,
                    "system": system,
                    "error": str(e),
                }
            )

    # persist updated mappings
    try:
        save_entry_dates(entry_map)
        _save_symbol_system_map(sys_map_store)
    except Exception:
        pass

    out = pd.DataFrame(results)
    if notify and not out.empty:
        try:
            Notifier(platform="auto").send_trade_report("exits", results)
        except Exception:
            pass
    return out
