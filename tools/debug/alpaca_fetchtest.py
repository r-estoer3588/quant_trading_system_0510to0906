import importlib
import os
from typing import Any

from dotenv import load_dotenv

# .env を読み込む
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_API_BASE_URL")
USE_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"


def _load_alpaca_modules() -> tuple[Any, Any, Any] | None:
    """alpaca-py を動的に読み込む。未導入なら None を返す。"""
    try:
        trading_client_mod = importlib.import_module("alpaca.trading.client")
        trading_requests_mod = importlib.import_module("alpaca.trading.requests")
        trading_enums_mod = importlib.import_module("alpaca.trading.enums")
        return trading_client_mod, trading_requests_mod, trading_enums_mod
    except ModuleNotFoundError:
        print(
            "Alpaca SDK (alpaca-py) が見つかりません。`pip install alpaca-py` を実行してください。"
        )
        return None


def main() -> None:
    mods = _load_alpaca_modules()
    if mods is None:
        return
    trading_client_mod, trading_requests_mod, trading_enums_mod = mods

    # クラス参照を取得
    TradingClient = trading_client_mod.TradingClient
    GetOrdersRequest = trading_requests_mod.GetOrdersRequest
    QueryOrderStatus = trading_enums_mod.QueryOrderStatus

    # TradingClient を初期化
    trading_client = TradingClient(API_KEY, API_SECRET, paper=USE_PAPER)

    # アカウント情報を取得
    account = trading_client.get_account()

    print("現金残高:", getattr(account, "cash", "取得不可"))
    print("購買力:", getattr(account, "buying_power", "取得不可"))
    print("総資産価値:", getattr(account, "equity", "取得不可"))
    print("口座ステータス:", getattr(account, "status", "取得不可"))

    # 保有ポジションを取得
    positions = trading_client.get_all_positions()
    if not positions:
        print("保有ポジションはありません。")
    else:
        print("保有ポジション:")
        for position in positions:
            if isinstance(position, dict):
                print(
                    f"シンボル: {position.get('symbol')}, "
                    f"数量: {position.get('qty')}, "
                    f"現在価格: {position.get('current_price')}, "
                    f"評価額: {position.get('market_value')}"
                )
                print(
                    f"  平均取得価格: {position.get('avg_entry_price')}, "
                    f"損益: {position.get('unrealized_pl')}, "
                    f"損益率: {float(position.get('unrealized_plpc', 0)) * 100:.2f}%"
                )
            elif hasattr(position, "symbol") and not isinstance(position, str):
                print(
                    f"シンボル: {position.symbol}, 数量: {position.qty}, "
                    f"現在価格: {position.current_price}, 評価額: {position.market_value}"
                )
                unrealized_plpc_percent = (
                    position.unrealized_plpc * 100
                    if getattr(position, "unrealized_plpc", None) is not None
                    else 0.0
                )
                print(
                    f"  平均取得価格: {position.avg_entry_price}, 損益: {position.unrealized_pl}, "
                    f"損益率: {unrealized_plpc_percent:.2f}%"
                )
            else:
                print(f"ポジション情報: {position}")

    # 直近の注文を取得
    request_params = GetOrdersRequest(
        status=QueryOrderStatus.ALL, limit=5
    )  # 例: ALL / OPEN / CLOSED
    orders = trading_client.get_orders(filter=request_params)

    if not orders:
        print("注文履歴はありません。")
    else:
        for o in orders:
            if (
                not isinstance(o, str)
                and hasattr(o, "id")
                and hasattr(o, "symbol")
                and hasattr(o, "qty")
                and hasattr(o, "status")
            ):
                print(f"ID: {o.id}, 銘柄: {o.symbol}, 数量: {o.qty}, 状態: {o.status}")
            else:
                print(f"注文情報: {o}")


if __name__ == "__main__":
    main()
