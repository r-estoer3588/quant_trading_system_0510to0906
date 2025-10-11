# Diagnostics API リファレンス

## 概要

各システムの候補生成関数は、シグナル候補リストと一緒に「diagnostics 辞書」を返すようになっています。この辞書には、候補がどのように絞り込まれたかを示す情報が含まれており、パイプラインの動作確認やトラブルシューティングに役立ちます。

Phase0-7 で導入した統一キーにより、すべてのシステムで共通の形式を持つようになりました。

---

## 統一キー一覧

### 全システム共通

すべてのシステム(System1-7)が以下のキーを返します:

| キー                    | 型  | 説明                       | 取り得る値の例                 |
| ----------------------- | --- | -------------------------- | ------------------------------ |
| `ranking_source`        | str | ランキング対象データ       | `"latest_only"`, `"full_scan"` |
| `setup_predicate_count` | int | Setup 条件を満たした行数   | `5`, `0`, `-1`(未実装)         |
| `ranked_top_n_count`    | int | 最終候補件数(ランキング後) | `3`, `0`                       |

**キーの意味**:

- `ranking_source`: 候補を探すときに「最新の 1 行だけ」を見たか(`latest_only`)、全データをスキャンしたか(`full_scan`)を示します。
- `setup_predicate_count`: Setup 条件(predicate 関数)を満たした行の総数です。これが多ければフィルタが広く、少なければ狭いことを意味します。
- `ranked_top_n_count`: ランキングを経て最終的に候補として選ばれた銘柄の件数です。

### System1 専用キー

System1 は two-phase フィルタの詳細を追加で返します:

| キー                        | 型   | 説明                              |
| --------------------------- | ---- | --------------------------------- |
| `predicate_only_pass_count` | int  | Setup predicate のみ通過した件数  |
| `mismatch_flag`             | bool | Setup 列との不一致があれば `True` |
| `count_a`                   | int  | フィルタ a 通過件数(レガシー)     |
| `count_b`                   | int  | フィルタ b 通過件数(レガシー)     |
| `count_c` ~ `count_f`       | int  | 各フィルタ段階の通過件数          |

**注意**: `mismatch_flag` が `True` になっている場合は、Setup 列と predicate 関数の結果が一致していない可能性があります。`VALIDATE_SETUP_PREDICATE=1` でログを確認してください。

---

## 使用例

### System1 candidates 生成

```python
from core.system1 import generate_system1_candidates

# 日次パイプラインなどで呼び出す
candidates, diagnostics = generate_system1_candidates(
    df, current_date, latest_only=True
)

print(diagnostics)
# 出力例:
# {
#   "ranking_source": "latest_only",
#   "setup_predicate_count": 5,
#   "ranked_top_n_count": 3,
#   "predicate_only_pass_count": 5,
#   "mismatch_flag": False,
#   "count_a": 10,
#   "count_b": 8,
#   ...
# }
```

### Diagnostics フォールバック(安全なアクセス)

もし diagnostics が古い形式や欠損している場合でも、安全にアクセスできるヘルパーがあります:

```python
from common.system_diagnostics import get_diagnostics_with_fallback

# raw_diagnostics がない場合でも -1 でフォールバック
safe_diag = get_diagnostics_with_fallback(raw_diagnostics, "system1")

print(safe_diag["setup_predicate_count"])
# 欠損していれば -1 が返る
```

### Snapshot Export

Mini パイプライン実行後に diagnostics をまとめて JSON 出力できます:

```bash
python scripts/run_all_systems_today.py --test-mode mini --skip-external
cat results_csv_test/diagnostics_snapshot_*.json | jq '.systems'
```

出力例:

```json
{
  "systems": [
    {
      "system_id": "system1",
      "diagnostics": {
        "ranking_source": "latest_only",
        "setup_predicate_count": 5,
  "ranked_top_n_count": 3,
        ...
      }
    },
    ...
  ]
}
```

---

## トラブルシューティング

### `setup_predicate_count` が `-1` になる

**原因**: Diagnostics が欠損しているか、未実装のシステムです。  
**対処**: `get_diagnostics_with_fallback()` でラップして安全にアクセスしてください。Phase6 までに System1-7 すべてで実装済みですが、古いコードが残っている可能性があります。

### `mismatch_flag` が `True` になる

**原因**: Setup 列(`setup_system1` など)と shared predicate 関数の結果が一致していません。  
**対処**: 環境変数 `VALIDATE_SETUP_PREDICATE=1` を設定して詳細ログを確認し、predicate ロジックを修正してください。

### `ranked_top_n_count` が `0` になる

**原因**: 候補がランキング前の段階で存在しなかったか、ランキング後にすべて除外されました。  
**対処**:

1. `setup_predicate_count` を確認し、Setup 条件を満たす行が存在するか確認
2. ランキングキー(ROC200, ADX7 など)が正しく計算されているか確認
3. Two-phase フィルタのログを詳しく見て、どの段階で除外されたかを特定

### `ranking_source` が期待と異なる

**原因**: `latest_only` フラグが意図通りに渡されていません。  
**対処**: 呼び出し元で `latest_only=True/False` を明示的に指定してください。デフォルトは `False`(full_scan)です。

---

## 関連ファイル

- `common/system_setup_predicates.py`: 各システムの Setup predicate 関数を実装
- `common/system_diagnostics.py`: Diagnostics フォールバック用ヘルパー
- `core/system1.py` ~ `core/system7.py`: Diagnostics 生成箇所
- `tools/export_diagnostics_snapshot.py`: Snapshot export ツール
- `tools/compare_diagnostics_snapshots.py`: Snapshot 差分比較ツール

---

## 開発者向けメモ

### 新しい diagnostics キーを追加する場合

1. `common/system_diagnostics.py` の `COMMON_DIAGNOSTICS_KEYS` または各システム専用の辞書に追加
2. `core/systemX.py` の候補生成関数で該当キーを返す
3. `tools/export_diagnostics_snapshot.py` でキーが正しく出力されることを確認
4. 本ドキュメント(diagnostics.md)に仕様を追記

### 統一キーを削除・変更する場合

既存のスナップショットやテストに影響するため、慎重に以下を実施してください:

- `tests/diagnostics/` 配下のテストを更新
- `tools/compare_diagnostics_snapshots.py` の差分比較ロジックを調整
- 本ドキュメントと CHANGELOG.md を更新

---

## まとめ

Diagnostics API を使うことで、各システムの候補生成プロセスを透明化し、テストや運用時のトラブルシューティングを容易にします。統一キーにより、システム横断での比較や分析も簡単になりました。

詳細な使用例や応用については、`docs/README.md` のナビゲーションから関連文書も参照してください。
