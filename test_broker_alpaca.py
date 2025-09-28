#!/usr/bin/env python3
"""broker_alpacaモジュールのget_open_orders関数の存在確認テスト"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from common import broker_alpaca as ba


def test_get_open_orders_exists():
    """get_open_orders関数が存在するかテストする"""
    print("broker_alpacaモジュールの属性一覧:")
    attrs = [attr for attr in dir(ba) if not attr.startswith("_")]
    for attr in sorted(attrs):
        print(f"  - {attr}")

    print(f"\nget_open_orders関数の存在確認: {hasattr(ba, 'get_open_orders')}")

    if hasattr(ba, "get_open_orders"):
        print(f"get_open_orders関数の型: {type(ba.get_open_orders)}")
        print(f"get_open_orders関数のdocstring: {ba.get_open_orders.__doc__}")
    else:
        print("❌ get_open_orders関数が見つかりません")


if __name__ == "__main__":
    test_get_open_orders_exists()
