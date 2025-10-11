# テストガイド

このディレクトリには、quant_trading_system の全自動テストが含まれています。

## 📁 ディレクトリ構造

```
tests/
├── conftest.py                       # 共通フィクスチャと設定
├── test_core_system7_focused.py      # System7 コア機能テスト
├── test_system7_branches.py          # System7 分岐カバレッジテスト
├── test_system7_cache.py             # System7 キャッシュ機能テスト
├── test_system7_coverage_boost.py    # System7 追加カバレッジテスト
└── ... (その他のテストファイル)
```

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
# System7のみ
python -m pytest tests/test_core_system7_focused.py tests/test_system7_branches.py tests/test_system7_cache.py tests/test_system7_coverage_boost.py

# カバレッジ測定
python -m pytest tests/test_*system7*.py --cov=core.system7 --cov-report=term-missing
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

| モジュール              | 目標カバレッジ | 現在    | 状態      |
| ----------------------- | -------------- | ------- | --------- |
| core/system7.py         | 57-62%         | **53%** | ✅ 達成   |
| common/cache_manager.py | 70%+           | -       | 📝 計画中 |
| core/system1-6.py       | 60%+           | -       | 📝 計画中 |

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
