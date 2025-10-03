from __future__ import annotations

import os
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from dotenv import load_dotenv

try:  # pragma: no cover - SDK 未導入環境でも壊れないように
    from alpaca.trading.client import TradingClient

    try:
        from alpaca.trading.enums import (
            OrderClass,
            OrderSide,
            QueryOrderStatus,
            TimeInForce,
        )
    except ImportError:
        from alpaca.trading.models.enums import (
            OrderClass,
            OrderSide,
            QueryOrderStatus,
            TimeInForce,
        )
    from alpaca.trading.requests import (
        GetOrdersRequest,
        LimitOrderRequest,
        MarketOrderRequest,
        StopLossRequest,
        TakeProfitRequest,
        TrailingStopOrderRequest,
    )
    from alpaca.trading.stream import TradingStream
except Exception:  # pragma: no cover
    TradingClient = None
    OrderSide = OrderClass = TimeInForce = QueryOrderStatus = None
    MarketOrderRequest = None
    LimitOrderRequest = None
    GetOrdersRequest = None
    TakeProfitRequest = None
    StopLossRequest = None
    TrailingStopOrderRequest = None
    TradingStream = None


def _require_sdk() -> None:
    if TradingClient is None:
        raise RuntimeError(
            "alpaca-py がインストールされていません。requirements に追加/インストールしてください。"
        )


def _load_env_once() -> None:
    # 設定側で読み込み済みでも harm はない
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"), override=False)


def get_client(
    *,
    paper: bool | None = None,
    api_key: str | None = None,
    secret_key: str | None = None,
):
    """TradingClient を生成して返す。

    - paper: None の場合は `ALPACA_PAPER` を '1/true/yes/on' として解釈（デフォルト True）
    - api_key/secret_key: 未指定なら .env から `APCA_API_KEY_ID`/`APCA_API_SECRET_KEY`
    """
    _require_sdk()
    _load_env_once()

    if paper is None:
        paper = os.getenv("ALPACA_PAPER", "true").lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
    api_key = api_key or os.getenv("APCA_API_KEY_ID")
    secret_key = secret_key or os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "APCA_API_KEY_ID/APCA_API_SECRET_KEY が .env に設定されていません。",
        )

    return TradingClient(  # type: ignore[misc]
        api_key,
        secret_key,
        paper=bool(paper),
    )


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

    if OrderSide is None:
        raise RuntimeError("Alpaca SDK is not installed or OrderSide is unavailable.")
    side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    if TimeInForce is None:
        raise RuntimeError("Alpaca SDK is not installed or TimeInForce is unavailable.")
    tif = (
        getattr(TimeInForce, time_in_force.upper())
        if hasattr(TimeInForce, time_in_force.upper())
        else TimeInForce.GTC
    )

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
            take_profit=TakeProfitRequest(
                limit_price=take_profit,
            ),  # type: ignore[call-arg]
            stop_loss=StopLossRequest(
                stop_price=stop_loss,
            ),  # type: ignore[call-arg]
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
            msg = f"Submitted {order_type} order {order.id} {symbol} qty={qty} side={side_enum.name}"
            log_callback(msg)
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
    last_exc: Exception | None = None
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
                    msg = " ".join(
                        [
                            f"submit retry {i + 1}/{retries}: {symbol}",
                            f"qty={qty} error={e}",
                        ]
                    )
                    log_callback(msg)
                except Exception:
                    pass
            if i < retries:
                time.sleep(max(0.0, backoff_seconds))
            if rate_limit_seconds > 0:
                time.sleep(rate_limit_seconds)
    assert last_exc is not None
    raise last_exc


def get_orders_status_map(client, order_ids: Iterable[str]) -> dict[str, Any]:
    """order_id -> status の簡易マップを返す."""
    id_set = {oid for oid in order_ids if oid}
    out: dict[str, Any] = {}
    if not id_set:
        return out
    for oid in id_set:
        try:
            o = client.get_order_by_id(oid)
            out[oid] = getattr(o, "status", None)
        except Exception:
            out[oid] = None
    return out


def log_orders_positions(client) -> tuple[Any, Any]:
    """現在の注文とポジションを同時に取得する."""
    if GetOrdersRequest is None or QueryOrderStatus is None:
        raise RuntimeError("alpaca-py がインストールされていません。")

    # 並列取得で待ち時間を短縮
    with ThreadPoolExecutor(max_workers=2) as executor:
        orders_future = executor.submit(
            client.get_orders, GetOrdersRequest(status=QueryOrderStatus.ALL)
        )
        positions_future = executor.submit(client.get_all_positions)
        orders = orders_future.result()
        positions = positions_future.result()
    return orders, positions


def get_open_orders(client) -> Any:
    """未約定注文を取得する."""
    if GetOrdersRequest is None or QueryOrderStatus is None:
        raise RuntimeError("alpaca-py がインストールされていません。")

    return client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))


def cancel_all_orders(client) -> None:
    """すべての未約定注文をキャンセルする."""
    try:
        client.cancel_orders()
    except Exception:
        # SDK のバージョン差異に対応
        client.cancel_all_orders()


def subscribe_order_updates(client, log_callback=None):
    """注文更新の WebSocket を購読して即時実行（ブロッキング）。"""
    _require_sdk()
    if TradingStream is None:
        raise RuntimeError("alpaca-py がインストールされていません。")

    stream = TradingStream(client.api_key, client.secret_key, paper=client.paper)  # type: ignore[attr-defined]
    stream_any: Any = stream  # type: ignore

    @stream_any.on_order_update
    async def _(data):  # noqa: ANN001 - SDK 固有
        if log_callback:
            try:
                msg = " ".join(
                    [
                        f"WS update {data.event} id={data.order.id}",
                        f"status={data.order.status}",
                    ]
                )
                log_callback(msg)
            except Exception:
                pass

    stream.run()


__all__ = [
    "get_client",
    "submit_order",
    "log_orders_positions",
    "cancel_all_orders",
    "subscribe_order_updates",
    "reset_paper_cash",
]


def get_shortable_map(client, symbols: Iterable[str]) -> dict[str, bool]:
    """Return a map of symbol->shortable via Alpaca Assets API.

    - On any SDK/API error, returns empty map (caller should skip filtering).
    - Symbols are upper-cased for lookup consistency.
    """
    out: dict[str, bool] = {}
    try:
        syms = [str(s).upper() for s in symbols if str(s).strip()]
    except Exception:
        return out
    if not syms:
        return out
    for sym in syms:
        try:
            asset = client.get_asset(sym)
            shortable = bool(getattr(asset, "shortable", False))
            out[sym] = shortable
        except Exception:
            # On error per-symbol, leave it out (unknown)
            continue
    return out


# ----------------------------------------------------------------------------
# Paper cash reset (best-effort / unofficial)
# ----------------------------------------------------------------------------
def reset_paper_cash(
    target_cash: float | int,
    *,
    api_key: str | None = None,
    secret_key: str | None = None,
    log_callback=None,
    dry_run: bool = False,
) -> dict[str, Any]:  # pragma: no cover - ネットワーク依存機能
    """ペーパーアカウントの現金残高を再設定しようと試みる補助関数。

    注意: Alpaca 現行 API / SDK では公式に *任意金額へ直接リセット* する公開エンドポイントは
    提供されていません。過去 (tradeapi.REST.patch_account) の挙動に倣った互換的ベストエフォートです。

    実装方針:
      1. paper 環境のみ許可 (live では即座に例外)
      2. 非公式 PATCH /v2/account を試行 (失敗時はエラーを返却)
      3. 失敗理由を含む dict を返す (呼び出し側は UI で表示)

    戻り値:
      {"ok": bool, "status": int | None, "error": str | None, "raw": dict | None}

    dry_run=True の場合は実行せず成功想定のみ返します。
    """

    # 型と値の正規化
    try:
        cash_val = float(target_cash)
        if cash_val <= 0:
            raise ValueError("target_cash は正の数である必要があります")
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": None,
            "error": f"invalid target_cash: {exc}",
            "raw": None,
        }

    # SDK が存在しない環境ではネットワーク呼び出しを避ける
    if TradingClient is None:
        return {
            "ok": False,
            "status": None,
            "error": "alpaca-py 未インストール",
            "raw": None,
        }

    _load_env_once()
    api_key = api_key or os.getenv("APCA_API_KEY_ID")
    secret_key = secret_key or os.getenv("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        return {"ok": False, "status": None, "error": "APIキー未設定", "raw": None}

    # paper 以外禁止 (保護)
    if os.getenv("ALPACA_PAPER", "true").lower() not in ("1", "true", "yes", "on", "y"):
        return {
            "ok": False,
            "status": None,
            "error": "live環境では実行禁止です",
            "raw": None,
        }

    # dry-run モード
    if dry_run:
        if log_callback:
            try:
                log_callback(f"[dry-run] Reset paper cash -> ${cash_val:,.2f}")
            except Exception:  # noqa: BLE001
                pass
        return {
            "ok": True,
            "status": None,
            "error": None,
            "raw": {"dry_run": True, "cash": cash_val},
        }

    try:
        import requests  # type: ignore
    except Exception:  # pragma: no cover - requests 無し環境
        return {
            "ok": False,
            "status": None,
            "error": "requests 未インストール",
            "raw": None,
        }

    url = "https://paper-api.alpaca.markets/v2/account"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {"cash": f"{cash_val:.2f}"}
    try:
        resp = requests.patch(url, json=payload, headers=headers, timeout=10)
    except Exception as exc:  # pragma: no cover - ネットワーク例外
        return {
            "ok": False,
            "status": None,
            "error": f"request error: {exc}",
            "raw": None,
        }

    status = resp.status_code
    raw_json: Any | None
    try:
        raw_json = resp.json()
    except Exception:  # noqa: BLE001
        raw_json = None

    ok = 200 <= status < 300
    if ok:
        if log_callback:
            try:
                log_callback(
                    f"✅ Paper cash reset success -> ${cash_val:,.2f} (status {status})"
                )
            except Exception:  # noqa: BLE001
                pass
        return {"ok": True, "status": status, "error": None, "raw": raw_json}
    else:
        err = None
        if isinstance(raw_json, dict):
            err = raw_json.get("message") or raw_json.get("error")
        if not err:
            err = f"HTTP {status}"
        if log_callback:
            try:
                log_callback(f"❌ Paper cash reset failed: {err}")
            except Exception:  # noqa: BLE001
                pass
        return {"ok": False, "status": status, "error": err, "raw": raw_json}
