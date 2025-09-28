#!/usr/bin/env python3
"""実際のAlpacaクライアントでget_open_ordersをテスト"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_real_alpaca_connection():
    """実際のAlpacaクライアントで接続テスト"""
    try:
        print("=== 実際のAlpaca接続テスト ===")
        
        from common import broker_alpaca as ba
        from dotenv import load_dotenv
        load_dotenv()
        
        # API キーの確認
        api_key = os.environ.get('APCA_API_KEY_ID')
        secret_key = os.environ.get('APCA_API_SECRET_KEY')
        
        print(f"API Key: {api_key[:8] if api_key else 'None'}...")
        print(f"Secret Key: {secret_key[:8] if secret_key else 'None'}...")
        
        # クライアント取得
        print("1. Alpacaクライアント取得...")
        client = ba.get_client(paper=True)  # ペーパートレードモード
        print(f"  ✅ クライアント取得成功: {type(client)}")
        
        # get_open_orders呼び出し
        print("2. 未約定注文取得...")
        orders = ba.get_open_orders(client)
        print(f"  ✅ get_open_orders実行成功")
        print(f"  - 結果の型: {type(orders)}")
        print(f"  - 注文数: {len(list(orders)) if orders else 0}")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー発生: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_alpaca_connection()
    sys.exit(0 if success else 1)