#!/usr/bin/env python
"""symbol_system_mapの問題を修正するツール

現在のマップが単一システム割り当てになっているため、
配分時に候補が除外される問題を解決します。
"""

import json
from pathlib import Path


def fix_symbol_system_map():
    """symbol_system_mapを確認・修正"""
    # プロジェクトルートを直接取得
    current_dir = Path(__file__).parent.parent
    map_path = current_dir / "data" / "symbol_system_map.json"

    print(f"🔍 symbol_system_map.jsonを調査中: {map_path}")

    if not map_path.exists():
        print(f"❌ symbol_system_map.json が見つかりません: {map_path}")
        # デフォルトマップを作成
        default_map: dict[str, list[str]] = {}
        map_path.parent.mkdir(exist_ok=True)
        with open(map_path, "w") as f:
            json.dump(default_map, f, indent=2)
        print("✅ 空のデフォルトマップを作成しました")
        return

    with open(map_path, "r") as f:
        symbol_map = json.load(f)

    print(f"📊 現在のマップ: {len(symbol_map)}銘柄登録済み")

    # 現在のマップ形式をチェック
    single_system_symbols: list[tuple[str, str | list[str]]] = []
    all_systems = ["system1", "system2", "system3", "system4", "system5", "system6", "system7"]

    for symbol, systems in symbol_map.items():
        if isinstance(systems, str):
            # 単一システム形式（問題）
            single_system_symbols.append((symbol, systems))
        elif isinstance(systems, list) and len(systems) < 7:
            # リスト形式だが制限あり
            single_system_symbols.append((symbol, systems))

    if single_system_symbols:
        print(f"❌ 問題発見！{len(single_system_symbols)}銘柄が制限されています")
        print("   サンプル:")
        for symbol, systems in single_system_symbols[:10]:
            print(f"     {symbol}: {systems}")

        print()
        response = input("🔧 全ての銘柄を全システムで使用可能にしますか？ (y/n): ")

        if response.lower() == "y":
            # 全てのシンボルを全システムに参加可能にする
            new_map = {}
            for symbol in symbol_map:
                new_map[symbol] = all_systems

            # バックアップ作成
            backup_path = map_path.with_suffix(".json.backup")
            with open(backup_path, "w") as f:
                json.dump(symbol_map, f, indent=2)
            print(f"📄 バックアップ作成: {backup_path}")

            # 新しいマップを保存
            with open(map_path, "w") as f:
                json.dump(new_map, f, indent=2)

            print("✅ 修正完了！全ての銘柄が全システムで使用可能になりました")
            print(f"   影響銘柄数: {len(new_map)}")
        else:
            print("❌ 修正をキャンセルしました")
    else:
        print("✅ 制限のあるシンボルはありません")


if __name__ == "__main__":
    fix_symbol_system_map()
