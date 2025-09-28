#!/usr/bin/env python3
"""Streamlit環境でのbroker_alpaca import問題の調査"""

import os
import sys

# パスの設定
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def test_streamlit_import():
    """Streamlit環境を模倣してインポートテスト"""
    print("=== Streamlit環境でのbroker_alpacaインポートテスト ===")

    try:
        # Streamlit関連の警告を抑制
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning)

        print("1. 基本インポート...")
        from common import broker_alpaca as ba

        print("  ✅ broker_alpaca インポート成功")

        print("2. get_open_orders存在確認...")
        if hasattr(ba, "get_open_orders"):
            print("  ✅ get_open_orders関数が存在")
            print(f"  - 型: {type(ba.get_open_orders)}")
            print(f"  - モジュール: {ba.get_open_orders.__module__}")
        else:
            print("  ❌ get_open_orders関数が見つからない")

        print("3. 利用可能な属性一覧:")
        attrs = [attr for attr in dir(ba) if not attr.startswith("_")]
        for attr in sorted(attrs):
            if callable(getattr(ba, attr)):
                print(f"  📝 {attr}: {type(getattr(ba, attr))}")

        print("4. 実際の呼び出しテスト...")
        try:
            # 実際のクライアント作成をスキップして関数の呼び出し自体をテスト
            func = getattr(ba, "get_open_orders", None)
            if func:
                print(f"  ✅ get_open_orders関数オブジェクト取得成功: {func}")
            else:
                print("  ❌ get_open_orders関数オブジェクト取得失敗")
        except Exception as e:
            print(f"  ❌ エラー: {e}")

        print("\n=== テスト完了 ===")

    except Exception as e:
        print(f"❌ インポートエラー: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_streamlit_import()
