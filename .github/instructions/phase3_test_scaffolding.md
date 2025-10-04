# Phase3: Test Scaffolding & Fixture Migration

## 🎯 目的

既存テストを diagnostics 対応に移行し、全システムのパラメトリックテストを拡充。最小 DataFrame フィクスチャを共通化して保守性を向上。

## 📋 前提条件（Phase2 完了済み）

- ✅ 共通 setup predicate 実装済み（`common/system_setup_predicates.py`）
- ✅ Diagnostics 統一キー導入済み（ranking_source, setup_predicate_count, final_top_n_count, predicate_only_pass_count, mismatch_flag）
- ✅ パラメトリックテスト実装済み（`tests/diagnostics/test_diagnostics_param_all_systems.py`）

## 🔧 実装タスク

### Task 3.1: Fixture 共通化（優先度: 中）

**目的**: 最小 DataFrame 生成ロジックを `conftest.py` に集約

**対象ファイル**:

- `tests/conftest.py`（新規 fixture 追加）
- `tests/diagnostics/test_diagnostics_param_all_systems.py`（fixture 利用に書き換え）

**実装内容**:

```python
# tests/conftest.py に追加
@pytest.fixture
def minimal_system1_df():
    """System1 の setup を満たす最小 DataFrame を返す。"""
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    data = {
        "Open": [10.0, 10.0],
        "High": [10.5, 10.6],
        "Low": [9.8, 9.9],
        "Close": [10.0, 10.5],
        "Volume": [5_000_000, 5_500_000],
        "dollarvolume20": [30_000_000, 35_000_000],
        "sma200": [9.0, 9.5],
        "roc200": [0.1, 0.2],
        "sma25": [9.7, 10.0],
        "sma50": [9.6, 9.9],
        "atr20": [0.2, 0.2],
    }
    df = pd.DataFrame(data, index=dates)
    df["filter"] = (df["Close"] >= 5.0) & (df["dollarvolume20"] > 25_000_000)
    df["setup"] = df["filter"] & (df["Close"] > df["sma200"]) & (df["roc200"] > 0)
    return df

# System2-7 も同様に実装
```

**検証**:

- 既存パラメトリックテストを fixture 利用版に書き換え
- `pytest tests/diagnostics/ -v` で全テストが green を維持

---

### Task 3.2: 既存テストの Diagnostics 対応移行（優先度: 低）

**目的**: `tests/` 配下の既存テストを diagnostics 対応に更新

**対象候補**:

- `tests/experimental/test_integration.py`
- `tests/experimental/test_ultra_phase2.py`
- その他、`generate_candidates_systemX()` を呼び出すテスト

**実装内容**:

1. `include_diagnostics=True` を追加
2. 戻り値のタプル展開を `by_date, merged, diag = result` に変更
3. 必須キーの存在を assert（ranking_source, setup_predicate_count 等）

**例**:

```python
# Before
result = generate_candidates_system1(prepared, latest_only=True, top_n=5)
by_date, merged = result

# After
result = generate_candidates_system1(prepared, latest_only=True, include_diagnostics=True, top_n=5)
if isinstance(result, tuple) and len(result) == 3:
    by_date, merged, diag = result
else:
    by_date, merged = result
    diag = {}
assert "ranking_source" in diag
```

**検証**:

- `pytest tests/ -k "integration or ultra" -v` で全テストが green

---

### Task 3.3: 全システムのパラメトリックテスト拡充（優先度: 中）

**目的**: latest_only だけでなく full_scan モードもカバー

**対象ファイル**:

- `tests/diagnostics/test_diagnostics_param_all_systems.py`（新規テスト追加）

**実装内容**:

```python
@pytest.mark.parametrize(
    "system_id, maker, func_import, top_n",
    [
        ("system1", _df_s1, "core.system1:generate_candidates_system1", 5),
        # ... 他のシステム
    ],
)
def test_diagnostics_shape_full_scan(system_id: str, maker, func_import: str, top_n: int):
    """Full scan モードでの diagnostics 形状を検証。"""
    module_name, func_name = func_import.split(":", 1)
    mod = __import__(module_name, fromlist=[func_name])
    gen_func = getattr(mod, func_name)

    # 複数日分の DataFrame を用意
    prepared = {("SPY" if system_id == "system7" else "AAA"): maker(True)}
    result = gen_func(prepared, latest_only=False, include_diagnostics=True, top_n=top_n)

    if isinstance(result, tuple) and len(result) == 3:
        by_date, merged, diag = result
    else:
        by_date, merged = result
        diag = {}

    assert isinstance(diag, dict)
    assert diag.get("ranking_source") == "full_scan"
    # setup_predicate_count は full_scan 時は複数日分の合計
    assert isinstance(diag.get("setup_predicate_count"), int)
```

**検証**:

- `pytest tests/diagnostics/ -v` で新規テストも含めて全 green

---

## 📊 完了条件

- [ ] `tests/conftest.py` に minimal_systemX_df fixture を実装（System1-7）
- [ ] パラメトリックテストを fixture 利用版に書き換え
- [ ] 既存テスト（integration, ultra）を diagnostics 対応に移行
- [ ] Full scan モードのパラメトリックテストを追加
- [ ] `pytest tests/ -v` で全テスト green

## 🔗 関連ドキュメント

- `docs/technical/diagnostics_api.md`（Phase7 で作成予定）
- `common/system_setup_predicates.py`（共通 predicate 実装）
- `tests/diagnostics/test_diagnostics_param_all_systems.py`（既存パラメトリックテスト）

## 🚀 開始コマンド

```bash
# テスト実行（現在の状態確認）
pytest tests/diagnostics/ -v

# 特定システムのみ
pytest tests/diagnostics/ -k "system1" -v

# Coverage 付き
pytest tests/diagnostics/ --cov=core --cov=common -v
```

## 📝 注意事項

- System6 は共通 predicate 未統合（別タスクで対応済み）なので、テストは既存ロジックのまま
- テストデータは現実的な値を使用（極端な値でテストをパスさせない）
- Fixture の引数で pass_setup=True/False を切り替え可能にする（setup 不成立のケースもテスト）
