#!/usr/bin/env python
"""一時的な差分テスト用スクリプト"""
import json

# 元のスナップショットを読み込み
with open(
    "results_csv_test/diagnostics_test/diagnostics_snapshot_20251004_114935.json",
    "r",
    encoding="utf-8",
) as f:
    data = json.load(f)

# system1のfinal_top_n_countを変更
data["systems"][0]["diagnostics"]["final_top_n_count"] = 2

# 変更版を保存
with open(
    "results_csv_test/diagnostics_test/test_modified.json", "w", encoding="utf-8"
) as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("テスト用変更スナップショット作成完了")
