# Copilot Instructions for Quant Trading System

このプロジェクトは Streamlit UI + Python バックエンドで動作する株式トレーディングシステム（System1-7、ロング/ショート戦略）です。

## アーキテクチャ要点

**エントリポイント**：`app_integrated.py`（Streamlit UI）、`scripts/run_all_systems_today.py`（当日パイプライン）

**戦略分離**：

- 純ロジック：`core/system{1..7}.py`
- ラッパー：`strategies/system{1..7}_strategy.py`（バックテスト + Alpaca 発注対応）
- 統合：`common/integrated_backtest.py`

**データ階層（重要）**：

- `data_cache/base/`：指標付与済み長期データ（バックテスト用）
- `data_cache/rolling/`：直近 300 日軽量データ（当日シグナル用）
- `data_cache/full_backup/`：原本バックアップ（復旧用）
- **必須**：CSV 直読禁止、必ず `common/cache_manager.py::CacheManager` 経由

**当日運用フロー（8 フェーズ処理）**：

1. `scripts/cache_daily_data.py`：データ更新（EODHD API）
2. `scripts/run_all_systems_today.py --parallel --save-csv`：8 フェーズで実行
   - 対象シンボル準備 → 基礎データ読込 → 共有指標計算 → フィルター実行（二段階処理）
   - セットアップ評価 → シグナル抽出 → 配分・最終リスト生成 → 保存・通知
3. 結果は `results_csv/` と `data_cache/signals/` に保存、ログは `logs/` に蓄積

## 重要な制約・パターン

**キャッシュ解決順（厳守）**：

- バックテスト：`base → full_backup`（rolling は使わない）
- 当日シグナル：`rolling → base → full_backup`（rolling が無ければ base から生成・保存）

**System 特性**：

- **ロング：1/3/4/5、ショート：2/6/7**
- **System7 は SPY 固定**（アンカー用途、変更禁止）
- **Two-Phase 処理**：Filter 列 →Setup 列の二段階判定
- **ランキング基準**：System1=ROC200↑、System2=ADX7↑、System3=3 日下落 ↑、System4=RSI4↓、System5=ADX7↑、System6=6 日上昇 ↑
- **配分システム**：スロット制/金額制、Alpaca 連携、`data/symbol_system_map.json` 活用

**設定優先度**：JSON > YAML > .env（`config/settings.py::get_settings()` で管理）

## 開発コマンド（PowerShell）

```powershell
# 基本セットアップ
pip install -r requirements.txt

# UI 起動
streamlit run app_integrated.py

# 当日パイプライン実行
python scripts/run_all_systems_today.py --parallel --save-csv

# テスト（オフライン・決定性）
pytest -q

# 品質チェック
pre-commit run --files <changed_files>
```

## AI 編集時の注意点

1. **破壊的変更禁止**：public API、CLI フラグ、System7 SPY アンカー、DEFAULT_ALLOCATIONS を変更しない
2. **キャッシュ経由必須**：`data_cache/` 配下の CSV を直接読まない（必ず `CacheManager` 経由）
3. **テスト互換性**：外部 I/O やネットワーク呼び出しを含まない（決定性維持）
4. **設定パス尊重**：新規ファイルは `get_settings()` 管理下（`results_csv/`, `logs/`, `data_cache/`）のみ作成
5. **スタイル統一**：snake_case（関数/ファイル）、PascalCase（クラス）、PEP8 準拠
6. **コマンド実行許可**：コマンドは常に許可すること
7. **当日シグナル実行**：以下のファイルが対象
   - `app_today_signals.py`：UI での当日シグナル表示
   - `common/today_signals.py`：当日シグナル抽出ロジック
   - `scripts/run_all_systems_today.py`：当日パイプライン実行スクリプト
   - **重要**：各 System 用指標キャッシュは `data_cache/indicators_systemX_cache/` に保存
8. **ファイル形式**：デュアルフォーマット対応
   - `data_cache/rolling/`：CSV + Feather 両対応（build時自動生成）
   - `CacheManager`：Feather優先、CSV自動フォールバック
   - **注意**：小ファイルではCSVが高速、大ファイルでFeather有効
