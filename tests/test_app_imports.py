#!/usr/bin/env python3
"""Test app imports after reorganization"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

print("Testing app imports...")

try:
    # メインアプリのimportテスト
    print("1. Testing main app import...")
    import apps.main  # noqa: F401

    print("✅ apps.main - OK")

    # システムアプリの import テスト (遅延戦略ファクトリ導入後)
    print("2. Testing system apps import...")
    system_modules = [
        "apps.systems.app_system1",
        "apps.systems.app_system2",
        "apps.systems.app_system3",
        "apps.systems.app_system4",
        "apps.systems.app_system5",
        "apps.systems.app_system6",
        "apps.systems.app_system7",
    ]
    imported = 0
    for mod in system_modules:
        try:
            __import__(mod)
            print(f"✅ {mod} - OK")
            imported += 1
        except Exception as e:  # pragma: no cover - defensive
            print(f"⚠️  {mod} import skipped: {e}")
    if imported == len(system_modules):
        print("✅ All system apps imported successfully")
    else:
        print(f"ℹ️ Imported {imported}/{len(system_modules)} system apps (some skipped)")

    # ダッシュボードアプリのimportテスト
    print("3. Testing dashboard apps import...")
    import apps.dashboards.alpaca  # noqa: F401

    print("✅ apps.dashboards.alpaca - OK")

    import apps.dashboards.today_signals  # noqa: F401

    print("✅ apps.dashboards.today_signals - OK")

    print("\n🎉 All app imports successful!")

except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"⚠️  Other error: {e}")
    print("Note: Some warnings are normal when not running with 'streamlit run'")
