#!/usr/bin/env python3
"""broker_alpaca get_open_orders のデバッグテスト"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def debug_broker_alpaca():
    """broker_alpacaモジュールの詳細デバッグ"""
    try:
        print("=== broker_alpaca モジュールのデバッグ開始 ===")

        # 1. モジュールのインポート確認
        print("1. モジュールインポート...")
        from common import broker_alpaca as ba

        print("  ✅ broker_alpaca インポート成功")

        # 2. get_open_orders関数の存在確認
        print("2. get_open_orders関数の確認...")
        if hasattr(ba, "get_open_orders"):
            print("  ✅ get_open_orders関数が存在します")
            func = getattr(ba, "get_open_orders")
            print(f"  - 関数オブジェクト: {func}")
            print(f"  - 関数の型: {type(func)}")
            print(f"  - ファイル: {func.__module__}")
        else:
            print("  ❌ get_open_orders関数が見つかりません")
            return False

        # 3. 必要な依存関係の確認
        print("3. 依存関係の確認...")
        dependencies = ["TradingClient", "GetOrdersRequest", "QueryOrderStatus"]
        for dep in dependencies:
            if hasattr(ba, dep):
                dep_obj = getattr(ba, dep)
                print(f"  ✅ {dep}: {dep_obj}")
                if dep_obj is None:
                    print(f"    ⚠️  {dep}はNoneです（alpaca-pyがインストールされていない可能性）")
            else:
                print(f"  ❌ {dep}が見つかりません")

        # 4. 実際の関数呼び出し（モックclientで）
        print("4. 関数の実行テスト...")
        try:
            # モッククライアントを作成（実際のAPIを呼び出さない）
            class MockClient:
                def get_orders(self, request):
                    return []

            mock_client = MockClient()
            result = ba.get_open_orders(mock_client)
            print(f"  ✅ get_open_orders実行成功: {result}")

        except Exception as e:
            print(f"  ❌ get_open_orders実行エラー: {type(e).__name__}: {e}")
            return False

        print("=== すべてのテストが成功しました ===")
        return True

    except Exception as e:
        print(f"❌ デバッグ中にエラーが発生: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = debug_broker_alpaca()
    sys.exit(0 if success else 1)
