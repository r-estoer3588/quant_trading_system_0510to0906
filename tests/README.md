# テストガイド

このディレクトリには、quant_trading_system の全自動テストが含まれています。

## 📁 ディレクトリ構造

```
tests/
├── conftest.py                       # 共通フィクスチャと設定
├── test_system7_branches.py          # System7 分岐網羅テスト (16テスト)
├── test_system7_latest_only.py       # System7 当日シグナル高速パス (10テスト)
├── test_system7_error_cases.py       # System7 エラーハンドリング (9テスト)
├── test_system7_full_scan.py         # System7 履歴スキャンモード (6テスト)
├── experimental/
│   └── system7_archive/              # System7旧テストファイル (16ファイル)
└── ... (その他のテストファイル)
```

### System7 公式テスト構成 (66%カバレッジ達成)

**統合結果** (2025 年 10 月 11 日更新):

- **総テスト数**: 41 テスト (全てパス)
- **達成カバレッジ**: 66% (162/247 行)
- **目標**: 65% → **超過達成** ✅

| ファイル名                  | テスト数 | カバレッジ | カバー範囲                        |
| --------------------------- | -------- | ---------- | --------------------------------- |
| test_system7_branches.py    | 16       | 89%        | 分岐条件、エッジケース            |
| test_system7_latest_only.py | 10       | 87%        | Lines 219-262 (latest_only=True)  |
| test_system7_error_cases.py | 9        | 96%        | エラー処理、ATR50 フォールバック  |
| test_system7_full_scan.py   | 6        | 98%        | Lines 275-401 (latest_only=False) |

**アーカイブファイル**: 16 個の古い実験的ファイルは `experimental/system7_archive/` に移動済み。
詳細は [system7_archive/README.md](experimental/system7_archive/README.md) を参照。

## 🎯 テスト実行方法

### 全テスト実行

```powershell
# 基本実行
python -m pytest -q

# 詳細出力
python -m pytest -v

# カバレッジ付き
python -m pytest --cov=core --cov=common --cov-report=term-missing
```

### 特定システムのテスト

```powershell
# System7のみ (公式4ファイル)
python -m pytest tests/test_system7_branches.py tests/test_system7_latest_only.py tests/test_system7_error_cases.py tests/test_system7_full_scan.py

# カバレッジ測定 (66%達成確認)
python -m pytest tests/test_system7_branches.py tests/test_system7_latest_only.py tests/test_system7_error_cases.py tests/test_system7_full_scan.py --cov=core.system7 --cov-report=term-missing -q
```

### 高速実行（並列）

```powershell
# pytest-xdistで並列実行
python -m pytest -n auto
```

## 🔧 重要な設定

### テスト決定性フィクスチャ (conftest.py)

すべてのテストは自動的に決定性（再現可能性）が保証されます。

```python
@pytest.fixture(autouse=True)
def ensure_test_determinism(request):
    """各テストに一意の決定性シードを設定

    機能:
    - 各テストにユニークなランダムシードを割り当て
    - pytest-xdist並列実行時も決定的動作を保証
    - テスト間の干渉を防止

    仕組み:
    - テスト名のハッシュ値からシード生成 (0 to 2^31-1)
    - common.testing.set_test_determinism()を呼び出し
    - numpy.random、random、pandas乱数を統一的に初期化
    """
    test_name = request.node.nodeid
    seed = abs(hash(test_name)) % (2**31)
    set_test_determinism(seed=seed)
```

**重要**: このフィクスチャにより、以下が保証されます:

- ✅ テストの再現性 (同じテストは常に同じ結果)
- ✅ 並列実行の安全性 (pytest-xdist 互換)
- ✅ テスト順序への非依存性
- ✅ ランダム値の決定性 (np.random、random)

### キャッシュクリーンアップ (conftest.py)

System7 固有のキャッシュは自動的にクリーンアップされます:

```python
@pytest.fixture(scope="function")
def cleanup_system7_cache():
    """System7のキャッシュディレクトリを削除"""
    cache_dir = "data_cache/indicators_system7_cache"
    # テスト前後で自動クリーンアップ
```

## 📊 カバレッジ目標

| モジュール              | 目標カバレッジ | 現在    | 状態                |
| ----------------------- | -------------- | ------- | ------------------- |
| core/system7.py         | 65%            | **66%** | ✅ 達成 (41 テスト) |
| common/cache_manager.py | 70%+           | -       | 📝 計画中           |
| core/system1-6.py       | 60%+           | -       | 📝 計画中           |

**System7 達成詳細** (2025 年 10 月 11 日):

- 開始: 53% (132/247 行)
- 最終: 66% (162/247 行)
- 改善: +13 ポイント
- 公式テストファイル: 4 ファイル、41 テスト

## 🐛 トラブルシューティング

### テストが不安定（時々失敗する）

**原因**: ランダムシードが固定されていない可能性

**解決策**: `ensure_test_determinism`フィクスチャが自動適用されているか確認

```powershell
# conftest.pyの@pytest.fixture(autouse=True)を確認
```

### pandas/numpy 互換性エラー

**症状**: `TypeError: int() argument must be a string... not '_NoValueType'`

**解決策**: `df.drop(columns=[...], errors="ignore")` を使わない

```python
# ❌ 非推奨
df = df.drop(columns=["col"], errors="ignore")

# ✅ 推奨
df = pd.DataFrame({c: df[c] for c in df.columns if c != "col"})
```

### pytest-cov との干渉

**症状**: 単独実行では PASS、カバレッジ測定時に FAIL

**原因**: pytest-cov が import フックを変更し、モジュールの動作に影響

**解決策**:

1. テストデータ生成を完全に独立させる
2. 外部キャッシュへの依存を最小化
3. 必要に応じて`@pytest.mark.no_cover`を使用

## 📝 テスト記述のベストプラクティス

### 1. テストクラスでグループ化

```python
class TestSystem7CandidateGeneration:
    """候補生成機能のテスト群"""

    def test_basic_generation(self):
        """基本的な候補生成"""
        # ...

    def test_empty_data_handling(self):
        """空データのハンドリング"""
        # ...
```

### 2. 明確な docstring

```python
def test_latest_only_missing_atr50(self):
    """Test latest_only when ATR50 is missing (line 233-235)."""
    # 対応するコード行数を明記
```

### 3. 決定性の確保

```python
# ✅ 推奨: フィクスチャが自動設定
def test_random_behavior(self):
    # ensure_test_determinismが自動適用
    result = np.random.randint(0, 100)
    assert result >= 0  # 常に再現可能

# ❌ 非推奨: 手動シード設定
def test_manual_seed(self):
    np.random.seed(42)  # フィクスチャと競合する可能性
```

### 4. テストデータの独立性

```python
# ✅ 推奨: 各テストで独自データ作成
def test_with_independent_data(self):
    df = pd.DataFrame({"Close": [100, 101, 102]})
    # ...

# ❌ 非推奨: グローバル共有データ
SHARED_DATA = pd.DataFrame(...)  # テスト間で干渉の可能性
```

## 🔄 CI/CD 統合

### GitHub Actions での実行

```yaml
- name: Run Tests
  run: |
    python -m pytest -v --cov=core --cov=common --cov-report=xml

- name: Upload Coverage
  uses: codecov/codecov-action@v3
```

### ローカル pre-commit フック

```bash
# .git/hooks/pre-commit
pytest tests/test_*system7*.py -q
```

## 📚 関連ドキュメント

- [プロジェクトルート README](../README.md)
- [システム仕様書](../docs/systems/)
- [技術詳細](../docs/technical/)
- [Copilot Instructions](../.github/copilot-instructions.md)

## 🆘 サポート

テスト関連の問題や質問:

1. エラーログを確認
2. `pytest -vv --tb=long` で詳細出力
3. conftest.py のフィクスチャ動作を確認
4. 必要に応じて issue 報告

---

最終更新: 2025 年 10 月 11 日
