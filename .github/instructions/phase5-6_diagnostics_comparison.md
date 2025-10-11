# Phase5-6: Diagnostics Fallback & Comparison Utilities

## 🎯 目的

Diagnostics が欠損・異常値の際のフォールバック処理を実装し、システム間や履歴との比較を容易にする JSON ベースのツールを構築。

## 📋 前提条件（Phase2–4 完了済み）

- ✅ Diagnostics 統一キー導入済み（ranking_source, setup_predicate_count, ranked_top_n_count など）
- ✅ Mini パイプライン検証済み
- ✅ TRD リスト長検証ツール実装済み

## 🔧 実装タスク

### Phase5: Diagnostics Fallback & Escalation

#### Task 5.1: ゼロ TRD 時のエスカレーション（優先度: 中）

**目的**: 全システムで候補ゼロの際に通知を送る

**対象ファイル**:

- `scripts/run_all_systems_today.py`（メインパイプライン）
- `common/notification.py`（通知ヘルパー、新規作成）

**実装内容**:

```python
# common/notification.py
def notify_zero_trd_all_systems(ctx, all_signals: list) -> None:
    """全システムで候補ゼロの際にエスカレーション通知を送る。

    Args:
        ctx: ExecContext（通知設定を含む）
        all_signals: compute_today_signals() の返り値リスト
    """
    if not all_signals or len(all_signals) == 0:
        message = (
            "⚠️ Zero TRD Alert: All systems returned zero candidates.\n"
            f"Mode: {ctx.test_mode or 'production'}\n"
            f"Date: {ctx.current_date}\n"
            "Action: Check filters, data freshness, and indicator calculation."
        )

        # ログに警告
        logging.warning(message)

        # 通知を送信（ctx.notify_enabled が True の場合）
        if ctx.notify_enabled:
            # ここで Discord/Slack/Email 等に通知
            # 実装は既存の通知機能を流用
            pass
```

**統合**:

- `scripts/run_all_systems_today.py` の `main()` 内で `all_signals` を検査
- 全システムで候補ゼロの場合に `notify_zero_trd_all_systems()` を呼び出し

**検証**:

```bash
# フィルタを厳しくして全システムゼロを再現
python scripts/run_all_systems_today.py --test-mode mini --skip-external
# ログに "⚠️ Zero TRD Alert" が出力されることを確認
```

---

#### Task 5.2: Diagnostics Missing 時のフォールバック（優先度: 低）

**目的**: diagnostics が None または空辞書の際にデフォルト値を返す

**対象**:

- `common/system_diagnostics.py`（新規ユーティリティ）

**実装内容**:

```python
# common/system_diagnostics.py
def get_diagnostics_with_fallback(diag: dict | None, system_id: str) -> dict:
    """Diagnostics が欠損している場合にデフォルト値を返す。

    Args:
        diag: 元の diagnostics 辞書（None 可）
        system_id: システム ID（ログ用）

    Returns:
        統一キーを含む辞書（欠損時はデフォルト値）
    """
    if diag is None or not isinstance(diag, dict):
        logging.warning(f"{system_id}: diagnostics is None or invalid, using fallback")
        diag = {}

    return {
        "ranking_source": diag.get("ranking_source", "unknown"),
        "setup_predicate_count": int(diag.get("setup_predicate_count", -1)),
        "ranked_top_n_count": int(diag.get("ranked_top_n_count", -1)),
        "predicate_only_pass_count": int(diag.get("predicate_only_pass_count", -1)),
        "mismatch_flag": bool(diag.get("mismatch_flag", False)),
        # System1 専用キー（他システムでは -1 でフォールバック）
        "count_a": int(diag.get("count_a", -1)),
        "count_b": int(diag.get("count_b", -1)),
        "count_c": int(diag.get("count_c", -1)),
        "count_d": int(diag.get("count_d", -1)),
        "count_e": int(diag.get("count_e", -1)),
        "count_f": int(diag.get("count_f", -1)),
    }
```

**統合**:

- `core/system1.py` ～ `core/system7.py` の候補生成関数内で使用
- Diagnostics を返す前に `get_diagnostics_with_fallback()` でラップ

**検証**:

```python
# tests/test_diagnostics_fallback.py
def test_fallback_none():
    result = get_diagnostics_with_fallback(None, "system1")
    assert result["ranking_source"] == "unknown"
    assert result["setup_predicate_count"] == -1

def test_fallback_partial():
    partial = {"ranking_source": "latest_only"}
    result = get_diagnostics_with_fallback(partial, "system2")
    assert result["ranking_source"] == "latest_only"
    assert result["setup_predicate_count"] == -1
```

---

### Phase6: Comparison Utilities

#### Task 6.1: Diagnostics Snapshot Export（優先度: 高）

**目的**: Mini パイプライン実行後に diagnostics を JSON としてエクスポート

**対象**:

- `scripts/run_all_systems_today.py`（export ロジック追加）
- 新規: `tools/export_diagnostics_snapshot.py`（スタンドアロン版）

**実装内容**:

```python
# tools/export_diagnostics_snapshot.py
def export_diagnostics_snapshot(all_signals: list, output_path: Path) -> None:
    """全システムの diagnostics をスナップショット JSON として保存。

    Args:
        all_signals: compute_today_signals() の返り値
        output_path: 出力先 JSON パス（例: results_csv_test/diagnostics_snapshot.json）
    """
    snapshot = {
        "export_date": datetime.now().isoformat(),
        "systems": [],
    }

    for sig in all_signals:
        system_id = sig.get("system_id", "unknown")
        diag = sig.get("diagnostics", {})

        # フォールバック適用
        diag_safe = get_diagnostics_with_fallback(diag, system_id)

        snapshot["systems"].append({
            "system_id": system_id,
            "diagnostics": diag_safe,
            "candidate_count": len(sig.get("candidates", [])),
        })

    # JSON 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    logging.info(f"Diagnostics snapshot exported to {output_path}")
```

**統合**:

- `scripts/run_all_systems_today.py` の `main()` 内で `all_signals` を export
- `results_csv_test/diagnostics_snapshot_YYYYMMDD.json` に保存（テストモード時のみ）

**検証**:

```bash
python scripts/run_all_systems_today.py --test-mode mini --skip-external
# results_csv_test/diagnostics_snapshot_*.json が生成されることを確認
cat results_csv_test/diagnostics_snapshot_*.json | jq '.systems[] | {system_id, diagnostics}'
```

---

#### Task 6.2: Diff Comparison Tool（優先度: 中）

**目的**: 2 つの diagnostics スナップショットを比較し、差分を分類

**対象**:

- 新規: `tools/compare_diagnostics_snapshots.py`

**実装内容**:

```python
# tools/compare_diagnostics_snapshots.py
def compare_snapshots(baseline_path: Path, current_path: Path) -> dict:
    """2 つの diagnostics スナップショットを比較。

    Args:
        baseline_path: ベースライン JSON
        current_path: 現在の JSON

    Returns:
        差分の辞書（システムごとの増減、カテゴリ分類を含む）
    """
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(current_path, "r", encoding="utf-8") as f:
        current = json.load(f)

    # システム ID でマッピング
    baseline_systems = {s["system_id"]: s for s in baseline["systems"]}
    current_systems = {s["system_id"]: s for s in current["systems"]}

    diffs = []
    for sys_id in sorted(set(baseline_systems.keys()) | set(current_systems.keys())):
        b_diag = baseline_systems.get(sys_id, {}).get("diagnostics", {})
        c_diag = current_systems.get(sys_id, {}).get("diagnostics", {})

        diff = {
            "system_id": sys_id,
            "setup_predicate_count": {
                "baseline": b_diag.get("setup_predicate_count", -1),
                "current": c_diag.get("setup_predicate_count", -1),
                "diff": c_diag.get("setup_predicate_count", -1)
                - b_diag.get("setup_predicate_count", -1),
            },
            "ranked_top_n_count": {
                "baseline": b_diag.get("ranked_top_n_count", -1),
                "current": c_diag.get("ranked_top_n_count", -1),
                "diff": c_diag.get("ranked_top_n_count", -1)
                - b_diag.get("ranked_top_n_count", -1),
            },
            "category": _classify_diff(b_diag, c_diag),
        }
        diffs.append(diff)

    return {
        "baseline_date": baseline.get("export_date"),
        "current_date": current.get("export_date"),
        "diffs": diffs,
    }


def _classify_diff(baseline: dict, current: dict) -> str:
    """差分をカテゴリ分類（no_change, increase, decrease, new, removed）。"""
    b_final = baseline.get("ranked_top_n_count", -1)
    c_final = current.get("ranked_top_n_count", -1)

    if b_final == -1 and c_final >= 0:
        return "new"
    elif b_final >= 0 and c_final == -1:
        return "removed"
    elif b_final == c_final:
        return "no_change"
    elif c_final > b_final:
        return "increase"
    else:
        return "decrease"
```

**CLI インターフェース**:

```bash
# 比較実行
python tools/compare_diagnostics_snapshots.py \
  --baseline results_csv_test/diagnostics_snapshot_20250913.json \
  --current results_csv_test/diagnostics_snapshot_20250914.json \
  --output results_csv_test/diagnostics_diff.json

# 結果表示
cat results_csv_test/diagnostics_diff.json | jq '.diffs[] | select(.category != "no_change")'
```

---

#### Task 6.3: Diff Category Summary（優先度: 低）

**目的**: 差分カテゴリごとの集計を表示

**対象**:

- `tools/compare_diagnostics_snapshots.py`（summary 関数追加）

**実装内容**:

```python
# tools/compare_diagnostics_snapshots.py（追加）
def summarize_diff(diff_result: dict) -> dict:
    """差分カテゴリごとの集計を返す。

    Returns:
        {category: count} の辞書
    """
    from collections import Counter
    categories = [d["category"] for d in diff_result["diffs"]]
    return dict(Counter(categories))
```

**CLI**:

```bash
python tools/compare_diagnostics_snapshots.py \
  --baseline baseline.json \
  --current current.json \
  --summary
# Output:
# {
#   "no_change": 5,
#   "increase": 1,
#   "decrease": 1
# }
```

---

## 📊 完了条件

- [ ] ゼロ TRD エスカレーション実装（`common/notification.py`）
- [ ] Diagnostics フォールバック実装（`common/system_diagnostics.py`）
- [ ] Snapshot export ツール実装（`tools/export_diagnostics_snapshot.py`）
- [ ] Diff 比較ツール実装（`tools/compare_diagnostics_snapshots.py`）
- [ ] Diff summary 集計機能追加
- [ ] Mini パイプライン実行後に snapshot JSON が生成される
- [ ] ベースラインと current の比較が正常に動作

## 🔗 関連ドキュメント

- `docs/technical/diagnostics.md`（Phase7 で作成予定）
- `scripts/run_all_systems_today.py`（メインパイプライン）
- `common/system_diagnostics.py`（診断ユーティリティ）

## 🚀 開始コマンド

```bash
# Mini パイプライン実行 + snapshot export
python scripts/run_all_systems_today.py --test-mode mini --skip-external

# Snapshot 確認
cat results_csv_test/diagnostics_snapshot_*.json | jq '.systems[] | {system_id, diagnostics}'

# 差分比較（2 回目実行後）
python tools/compare_diagnostics_snapshots.py \
  --baseline results_csv_test/diagnostics_snapshot_20250913.json \
  --current results_csv_test/diagnostics_snapshot_20250914.json \
  --output results_csv_test/diagnostics_diff.json \
  --summary
```

## 📝 注意事項

- ゼロ TRD エスカレーションは production モードでのみ通知送信
- Diagnostics フォールバックは -1 を「欠損」のマーカーとして使用
- Snapshot JSON はテストモード時のみ生成（production では不要）
- Diff 比較は手動実行を想定（CI での自動化は Phase8 で検討）
