"""Alpaca注文関連の共通ミックスイン。"""
from __future__ import annotations

import os

try:  # pragma: no cover - alpaca-py が未導入でもインポート失敗を防ぐ
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
    MarketOrderRequest = (
        LimitOrderRequest
    ) = TakeProfitRequest = StopLossRequest = TrailingStopOrderRequest = None  # type: ignore
    OrderSide = OrderClass = TimeInForce = None  # type: ignore
    TradingStream = None  # type: ignore


class AlpacaOrderMixin:
    """Alpaca API を利用した注文ユーティリティを提供するミックスイン。"""

    def init_trading_client(self, *, paper: bool = True) -> TradingClient:
        """.env から API キーを読み込み `TradingClient` を生成する。"""
        if TradingClient is None:
            raise RuntimeError(
                "alpaca-py がインストールされていません。requirements に追加してください。"
            )
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise RuntimeError("ALPACA_API_KEY/ALPACA_SECRET_KEY が .env に設定されていません。")
        return TradingClient(api_key, secret_key, paper=paper)

    def _build_market_order(self, *, symbol, qty, side_enum, tif, **_):
        return MarketOrderRequest(symbol=symbol, qty=qty, side=side_enum, time_in_force=tif)

    def _build_limit_order(
        self,
        *,
        symbol,
        qty,
        side_enum,
        tif,
        limit_price=None,
        **_,
    ):
        if limit_price is None:
            raise ValueError("limit_price が必要です。")
        return LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side_enum,
            limit_price=limit_price,
            time_in_force=tif,
        )

    def _build_oco_order(
        self,
        *,
        symbol,
        qty,
        side_enum,
        tif,
        take_profit=None,
        stop_loss=None,
        **_,
    ):
        if take_profit is None or stop_loss is None:
            raise ValueError("take_profit と stop_loss が必要です。")
        return LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side_enum,
            time_in_force=tif,
            order_class=OrderClass.OCO,
            take_profit=TakeProfitRequest(limit_price=take_profit),
            stop_loss=StopLossRequest(stop_price=stop_loss),
        )

    def _build_trailing_stop_order(
        self,
        *,
        symbol,
        qty,
        side_enum,
        tif,
        trail_percent=None,
        stop_price=None,
        **_,
    ):
        if trail_percent is None and stop_price is None:
            raise ValueError("trail_percent か trail_price のいずれかが必要です。")
        return TrailingStopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side_enum,
            time_in_force=tif,
            trail_percent=trail_percent,
            trail_price=stop_price,
        )

    def submit_order(
        self,
        client: TradingClient,
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
        """Alpaca へ注文を送信するユーティリティ。"""
        if TradingClient is None:
            raise RuntimeError("alpaca-py がインストールされていません。")
        side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif = TimeInForce.GTC

        builders = {
            "market": self._build_market_order,
            "limit": self._build_limit_order,
            "oco": self._build_oco_order,
            "trailing_stop": self._build_trailing_stop_order,
        }
        builder = builders.get(order_type)
        if builder is None:
            raise ValueError(f"未知の order_type: {order_type}")

        req = builder(
            symbol=symbol,
            qty=qty,
            side_enum=side_enum,
            tif=tif,
            limit_price=limit_price,
            stop_price=stop_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            trail_percent=trail_percent,
        )

        order = client.submit_order(order_data=req)
        if log_callback:
            log_callback(
                f"Submitted {order_type} order {order.id} {symbol} qty={qty} side={side_enum.name}"
            )
        return order

    def log_orders_positions(self, client: TradingClient, log_callback=None):
        """現在の注文・ポジションを取得してログ出力する。"""
        orders = client.get_orders(status="all")
        positions = client.get_all_positions()
        if log_callback:
            for o in orders:
                log_callback(
                    f"Order {o.id} {o.symbol} {o.side} {o.status} filled={o.filled_qty}"
                )
            for p in positions:
                log_callback(
                    f"Position {p.symbol} qty={p.qty} avg_entry={p.avg_entry_price}"
                )
        return orders, positions

    def subscribe_order_updates(
        self, client: TradingClient, log_callback=None
    ) -> TradingStream:
        """注文更新の WebSocket を購読し、更新時にログ出力する。"""
        if TradingStream is None:
            raise RuntimeError("alpaca-py がインストールされていません。")

        stream = TradingStream(client.api_key, client.secret_key, paper=client.paper)

        @stream.on_order_update
        async def _(data):  # noqa: ANN001 - Alpaca SDK 固有シグネチャ
            if log_callback:
                log_callback(
                    f"WS update {data.event} id={data.order.id} status={data.order.status}"
                )

        stream.run()  # 実行はブロッキング。適宜スレッド化/async 対応が必要。
        return stream
