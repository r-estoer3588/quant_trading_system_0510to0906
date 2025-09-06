import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus

# .env を読み込む
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_API_BASE_URL")
USE_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# TradingClient を初期化
trading_client = TradingClient(API_KEY, API_SECRET, paper=USE_PAPER)

# アカウント情報を取得
account = trading_client.get_account()

print("現金残高:", account.cash)
print("購買力:", account.buying_power)
print("総資産価値:", account.portfolio_value)
print("口座ステータス:", account.status)

# 保有ポジションを取得
positions = trading_client.get_all_positions()
if not positions:
    print("保有ポジションはありません。")
else:
    print("保有ポジション:")
    for position in positions:
        print(
            f"シンボル: {position.symbol}, 数量: {position.qty}, 現在価格: {position.current_price}, 評価額: {position.market_value}"
        )
        print(
            f"  平均取得価格: {position.avg_entry_price}, 損益: {position.unrealized_pl}, 損益率: {position.unrealized_plpc * 100:.2f}%"
        )
# 直近の注文を取得
request_params = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=5)  # 例: ALL / OPEN / CLOSED

orders = trading_client.get_orders(filter=request_params)

if not orders:
    print("注文履歴はありません。")
else:
    for o in orders:
        print(f"ID: {o.id}, 銘柄: {o.symbol}, 数量: {o.qty}, 状態: {o.status}")
