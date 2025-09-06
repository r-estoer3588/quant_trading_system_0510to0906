from __future__ import annotations

import os
from typing import Any, Optional, Tuple, Dict, Iterable
import time

from dotenv import load_dotenv

try:  # pragma: no cover - SDK 未導入環境でも壊れないように
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, OrderClass, TimeInForce
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        TakeProfitRequest,
        StopLossRequest,
        TrailingStopOrderRequest,
    )
    from alpaca.trading.stream import TradingStream
except Exception:  # pragma: no cover
    TradingClient = None  # type: ignore
    OrderSide = OrderClass = TimeInForce = None  # type: ignore
    MarketOrderRequest = LimitOrderRequest = TakeProfitRequest = StopLossRequest = TrailingStopOrderRequest = None  # type: ignore
    TradingStream = None  # type: ignore


def _require_sdk() -> None:
    if TradingClient is None:
        raise RuntimeError("alpaca-py がインストールされていません。requirements に追加/インストールしてください。")


def _load_env_once() -> None:
    # 設定側で読み込み済みでも harm はない
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)


def get_client(*, paper: Optional[bool] = None, api_key: Optional[str] = None, secret_key: Optional[str] = None):
    """TradingClient を生成して返す。

    - paper: None の場合は `ALPACA_PAPER` を '1/true/yes/on' として解釈（デフォルト True）
    - api_key/secret_key: 未指定なら .env から `ALPACA_API_KEY`/`ALPACA_SECRET_KEY`
    """
    _require_sdk()
    _load_env_once()

    if paper is None:
        paper = os.getenv("ALPACA_PAPER", "true").lower() in ("1", "true", "yes", "y", "on")
    api_key = api_key or os.getenv("ALPACA_API_KEY")
    secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("ALPACA_API_KEY/ALPACA_SECRET_KEY が .env に設定されていません。")
    return TradingClient(api_key, secret_key, paper=bool(paper))  # type: ignore[arg-type]


def submit_order(
    client,
    symbol: str,
    qty: int,
    *,
    side: str = "buy",
    order_type: str = "market",
    limit_price: float | None = None,
    stop_price: float | None = None,
    take_profit: float | None = None,
    stop_loss: float | None = None,
    trail_percent: float | None = None,
    time_in_force: str = "GTC",
    log_callback=None,
):
    """注文を送信する共通関数。

    order_type: "market" | "limit" | "oco" | "trailing_stop"
    """
    _require_sdk()

    side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL  # type: ignore[attr-defined]
    tif = getattr(TimeInForce, time_in_force.upper()) if hasattr(TimeInForce, time_in_force.upper()) else TimeInForce.GTC  # type: ignore[attr-defined]

    if order_type == "market":
        req = MarketOrderRequest(  # type: ignore[call-arg]
            symbol=symbol,
            qty=qty,
            side=side_enum,
            time_in_force=tif,
        )
    elif order_type == "limit":
        if limit_price is None:
            raise ValueError("limit_price が必要です。")
        req = LimitOrderRequest(  # type: ignore[call-arg]
            symbol=symbol,
            qty=qty,
            side=side_enum,
            limit_price=limit_price,
            time_in_force=tif,
        )
    elif order_type == "oco":
        if take_profit is None or stop_loss is None:
            raise ValueError("take_profit と stop_loss が必要です。")
        req = LimitOrderRequest(  # type: ignore[call-arg]
            symbol=symbol,
            qty=qty,
            side=side_enum,
            time_in_force=tif,
            order_class=OrderClass.OCO,  # type: ignore[attr-defined]
            take_profit=TakeProfitRequest(limit_price=take_profit),  # type: ignore[call-arg]
            stop_loss=StopLossRequest(stop_price=stop_loss),  # type: ignore[call-arg]
        )
    elif order_type == "trailing_stop":
        if trail_percent is None and stop_price is None:
            raise ValueError("trail_percent か trail_price のいずれかが必要です。")
        req = TrailingStopOrderRequest(  # type: ignore[call-arg]
            symbol=symbol,
            qty=qty,
            side=side_enum,
            time_in_force=tif,
            trail_percent=trail_percent,
            trail_price=stop_price,
        )
    else:
        raise ValueError(f"未知の order_type: {order_type}")

    order = client.submit_order(order_data=req)
    if log_callback:
        try:
            log_callback(f"Submitted {order_type} order {order.id} {symbol} qty={qty} side={side_enum.name}")
        except Exception:
            pass
    return order


def submit_order_with_retry(
    client,
    symbol: str,
    qty: int,
    *,
    side: str = "buy",
    order_type: str = "market",
    limit_price: float | None = None,
    stop_price: float | None = None,
    take_profit: float | None = None,
    stop_loss: float | None = None,
    trail_percent: float | None = None,
    time_in_force: str = "GTC",
    retries: int = 2,
    backoff_seconds: float = 1.0,
    rate_limit_seconds: float = 0.0,
    log_callback=None,
):
    """submit_order をリトライ付きで実行。
    - retries: 失敗時の再試行回数
    - backoff_seconds: 失敗毎に待機する秒数（指数ではなく線形）
    - rate_limit_seconds: 成功/失敗に関わらず各試行後に待機
    """
    last_exc: Optional[Exception] = None
    for i in range(retries + 1):
        try:
            order = submit_order(
                client,
                symbol,
                qty,
                side=side,
                order_type=order_type,
                limit_price=limit_price,
                stop_price=stop_price,
                take_profit=take_profit,
                stop_loss=stop_loss,
                trail_percent=trail_percent,
                time_in_force=time_in_force,
                log_callback=log_callback,
            )
            if rate_limit_seconds > 0:
                time.sleep(rate_limit_seconds)
            return order
        except Exception as e:  # pragma: no cover - ネットワーク/SDK例外
            last_exc = e
            if log_callback:
                try:
                    log_callback(f"submit retry {i+1}/{retries}: {symbol} qty={qty} error={e}")
                except Exception:
                    pass
            if i < retries:
                time.sleep(max(0.0, backoff_seconds))
            if rate_limit_seconds > 0:
                time.sleep(rate_limit_seconds)
    assert last_exc is not None
    raise last_exc


def get_orders_status_map(client, order_ids: Iterable[str]) -> Dict[str, Any]:
    """order_id -> status の簡易マップを返す。"""
    id_set = {oid for oid in order_ids if oid}
    out: Dict[str, Any] = {}
    if not id_set:
        return out
    for oid in id_set:
        try:
            o = client.get_order_by_id(oid)
            out[oid] = getattr(o, "status", None)
        except Exception:
            out[oid] = None
    return out


def log_orders_positions(client) -> Tuple[Any, Any]:
    """現在の注文とポジションを取得し、必要ならログ出力。"""
    orders = client.get_orders(status="all")
    positions = client.get_all_positions()
    return orders, positions


def subscribe_order_updates(client, log_callback=None):
    """注文更新の WebSocket を購読して即時実行（ブロッキング）。"""
    _require_sdk()
    if TradingStream is None:
        raise RuntimeError("alpaca-py がインストールされていません。")

    stream = TradingStream(client.api_key, client.secret_key, paper=client.paper)  # type: ignore[attr-defined]

    @stream.on_order_update
    async def _(data):  # noqa: ANN001 - SDK 固有
        if log_callback:
            try:
                log_callback(f"WS update {data.event} id={data.order.id} status={data.order.status}")
            except Exception:
                pass

    stream.run()
    return stream


__all__ = [
    "get_client",
    "submit_order",
    "log_orders_positions",
    "subscribe_order_updates",
]
