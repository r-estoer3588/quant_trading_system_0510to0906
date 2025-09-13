"""Alpaca注文関連の共通ミックスイン。"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import (
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TrailingStopOrderRequest,
)

from common import broker_alpaca as ba

if TYPE_CHECKING:
    from alpaca.trading.stream import TradingStream

logger = logging.getLogger(__name__)


class AlpacaOrderManager:
    """Alpaca API を利用した注文ユーティリティを提供するミックスイン。"""

    def __init__(
        self,
        trading_client: TradingClient,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool | None = None,
    ):
        self.trading_client = trading_client
        self._api_key = api_key
        self._secret_key = secret_key
        # TradingStream の paper 引数に渡すために記録
        # （未指定時は trading_client から推定、なければ True）
        default_paper = getattr(trading_client, "paper", True)
        self._paper = paper if paper is not None else default_paper

    @classmethod
    def create_from_env(cls, paper: bool = True) -> AlpacaOrderManager:
        """.env から API キーを読み込み `TradingClient` を生成する。"""
        if TradingClient is None:
            raise RuntimeError(
                "alpaca-py がインストールされていません。requirements に追加してください。"
            )
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise RuntimeError("ALPACA_API_KEY/ALPACA_SECRET_KEY が .env に設定されていません。")
        return cls(
            TradingClient(api_key, secret_key, paper=paper),
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )

    def _build_market_order(self, *, symbol, qty, side_enum, tif, **_):
        return MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side_enum,
            time_in_force=tif,
        )

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
        limit_price=None,
        stop_loss=None,
        **_,
    ):
        # OCO は親のリミット注文 + 子のストップ注文の組み合わせ
        if limit_price is None or stop_loss is None:
            raise ValueError("limit_price と stop_loss が必要です。")
        return LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side_enum,
            time_in_force=tif,
            limit_price=limit_price,
            order_class=OrderClass.OCO,
            # 親がリミット（利確）なので take_profit は不要。子のストップのみ指定。
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
        order_id = getattr(order, "id", None)
        if log_callback:
            msg = (
                f"Submitted {order_type} order {order_id} {symbol} "
                f"qty={qty} side={side_enum.name}"
            )
            log_callback(msg)
        return order

    def log_orders_positions(self, client: TradingClient, log_callback=None):
        """現在の注文・ポジションを取得してログ出力する。"""
        orders, positions = ba.log_orders_positions(client)
        if log_callback:
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

    def subscribe_order_updates(self, log_callback=None) -> TradingStream:
        """注文更新の WebSocket を購読し、更新時にログ出力する。"""
        try:
            from alpaca.trading.stream import TradingStream
        except ImportError as err:
            raise RuntimeError(
                "alpaca-py がインストールされていません。",
            ) from err

        # TradingClient からはキーを取得できないため、保持しているキー（無ければ環境変数）を使用する
        api_key = self._api_key or os.getenv("ALPACA_API_KEY")
        secret_key = self._secret_key or os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise RuntimeError("ALPACA_API_KEY/ALPACA_SECRET_KEY が利用できません。")

        stream = TradingStream(
            api_key,
            secret_key,
            paper=self._paper,  # trading_clientのpaper属性が存在しない場合のデフォルト
        )

        async def _on_trade_update(data):  # noqa: ANN001 - Alpaca SDK 固有シグネチャ
            if log_callback:
                event = getattr(data, "event", None)
                order_obj = getattr(data, "order", None)
                oid = getattr(order_obj, "id", None)
                status = getattr(order_obj, "status", None)
                log_callback(f"Order update: event={event} id={oid} status={status}")

        stream.subscribe_trade_updates(_on_trade_update)

        self._trading_stream = stream
        stream.run()  # 実行はブロッキング。適宜スレッド化/async 対応が必要。
        return stream


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

    def log_orders_positions(self, client: TradingClient, log_callback=None):
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
    "AlpacaOrderManager",
    "AlpacaOrderMixin",
]
