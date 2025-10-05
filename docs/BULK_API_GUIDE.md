# Bulk API データ品質検証とスケジュール実行ガイド

## 📋 概要

このガイドでは、EODHD Bulk API のデータ品質を検証し、安全に日次更新をスケジュール実行する方法を説明します。

## 🎯 解決する問題

- **Bulk API**: 1 日あたり 1-4 コール（月間 80 コール程度）で効率的だが、データ品質が不安定
- **個別 API**: データは正確だが、6,000 銘柄 × 20 営業日 = 月間 12 万コールで上限超過のリスク

→ **解決策**: Bulk API の品質を事前検証し、問題があれば自動的に個別 API にフォールバック

## 🛠️ 提供するツール

### 1. Bulk API データ品質検証スクリプト

**ファイル**: `scripts/verify_bulk_accuracy.py`

Bulk API で取得したデータの精度を検証します。

#### 基本的な使い方

```powershell
# デフォルトのサンプル銘柄で検証（SPY, QQQ, AAPL等）
python scripts/verify_bulk_accuracy.py

# 特定銘柄を指定して検証
python scripts/verify_bulk_accuracy.py --symbols AAPL,MSFT,TSLA,NVDA

# 取得タイミングの影響を調査
python scripts/verify_bulk_accuracy.py --timing

# カバレッジ分析（Bulkデータに含まれる銘柄数）
python scripts/verify_bulk_accuracy.py --coverage

# すべての分析を実行
python scripts/verify_bulk_accuracy.py --full
```

#### 検証結果の見方

```
📋 検証結果サマリー
  検証銘柄数: 10/10
  完全一致: 8件
  問題検出: 2件
  データ欠損: 0件

✅ 信頼性スコア: 80.0%
💡 一部銘柄で差異がありますが、許容範囲内です。
```

**信頼性スコア**:

- **95%以上**: 高品質、Bulk API 推奨
- **80-95%**: 許容範囲、一部銘柄で個別確認推奨
- **80%未満**: 低品質、個別 API 使用を推奨

### 2. 安全な日次更新スクリプト

**ファイル**: `scripts/scheduled_daily_update.py`

Bulk API を試み、品質チェックで問題があれば個別 API にフォールバックします。

#### 基本的な使い方

```powershell
# 通常の日次更新（推奨）
python scripts/scheduled_daily_update.py

# Bulk APIを強制使用（品質チェックスキップ）
python scripts/scheduled_daily_update.py --force-bulk

# 個別APIを強制使用（Bulkスキップ）
python scripts/scheduled_daily_update.py --force-individual
```

#### 実行フロー

1. **市場データ安定性チェック**: 推奨実行時刻（朝 6 時以降）の確認
2. **Bulk 品質検証**: サンプル銘柄でデータ精度をチェック
3. **Bulk 更新実行**: 品質が良ければ Bulk API で更新
4. **フォールバック**: 品質が低いか失敗した場合は個別 API に自動切り替え
5. **Rolling cache 更新**: 指標付き最新 330 日データを再構築
6. **事後検証**: シグナル生成テストで正常性確認
7. **統計記録**: 実行結果を JSON で保存

#### ログ出力

実行すると詳細なログが出力されます:

```
[2025-10-06 06:00:00] [INFO] ============================================================
[2025-10-06 06:00:00] [INFO] 🚀 日次更新処理を開始します
[2025-10-06 06:00:00] [INFO] ============================================================
[2025-10-06 06:00:01] [INFO] 実行時刻は推奨範囲内です（市場データは安定していると想定）
[2025-10-06 06:00:01] [INFO] ============================================================
[2025-10-06 06:00:01] [INFO] Bulk APIデータ品質の事前検証を開始
[2025-10-06 06:00:01] [INFO] ============================================================
...
[2025-10-06 06:05:23] [SUCCESS] ✅ 日次更新が正常に完了しました（方法: bulk）
[2025-10-06 06:05:23] [INFO]    Bulk信頼性スコア: 95.0%
```

ログファイルは `logs/daily_update_YYYYMMDD_HHMMSS.log` に保存されます。

## 📅 スケジュール実行の設定

### Windows タスクスケジューラー

毎日朝 6 時に自動実行する設定:

#### コマンドプロンプト（CMD）で実行する場合

**1 行で実行（推奨）：**

```cmd
schtasks /create /tn "QuantTradingDailyUpdate" /tr "C:\Repos\quant_trading_system\venv\Scripts\python.exe C:\Repos\quant_trading_system\scripts\scheduled_daily_update.py" /sc daily /st 06:00
```

**複数行に分ける場合（キャレット ^ を使用）：**

```cmd
schtasks /create /tn "QuantTradingDailyUpdate" ^
  /tr "C:\Repos\quant_trading_system\venv\Scripts\python.exe C:\Repos\quant_trading_system\scripts\scheduled_daily_update.py" ^
  /sc daily /st 06:00
```

#### PowerShell で実行する場合

**バッククォート ` を使用：**

```powershell
schtasks /create /tn "QuantTradingDailyUpdate" `
  /tr "C:\Repos\quant_trading_system\venv\Scripts\python.exe C:\Repos\quant_trading_system\scripts\scheduled_daily_update.py" `
  /sc daily /st 06:00
```

#### タスク管理コマンド

**タスク削除：**

```cmd
schtasks /delete /tn "QuantTradingDailyUpdate" /f
```

**タスク実行状況確認：**

```cmd
schtasks /query /tn "QuantTradingDailyUpdate" /fo list /v
```

**手動実行（テスト用）：**

```cmd
schtasks /run /tn "QuantTradingDailyUpdate"
```

### Linux/Mac (cron)

```bash
# crontabを編集
crontab -e

# 以下を追加（毎日朝6時に実行）
0 6 * * * cd /path/to/quant_trading_system && ./venv/bin/python scripts/scheduled_daily_update.py >> logs/cron_daily_update.log 2>&1
```

## 📊 実行統計の確認

実行履歴は `logs/daily_update_stats.json` に保存されます（最新 30 日分）:

```json
[
  {
    "start_time": "2025-10-06T06:00:00",
    "end_time": "2025-10-06T06:05:23",
    "method_used": "bulk",
    "success": true,
    "bulk_reliability_score": 0.95
  },
  {
    "start_time": "2025-10-05T06:00:00",
    "end_time": "2025-10-05T06:22:15",
    "method_used": "individual_quality",
    "success": true,
    "bulk_reliability_score": 0.65
  }
]
```

**method_used**の種類:

- `bulk`: Bulk API で成功
- `individual_fallback`: Bulk 失敗後、個別 API で成功
- `individual_quality`: Bulk 品質低により個別 API を使用
- `bulk_forced`: `--force-bulk` で強制実行
- `individual_forced`: `--force-individual` で強制実行
- `failed`: すべて失敗

## 🔍 トラブルシューティング

### Bulk API の品質が継続的に低い場合

```powershell
# タイミング影響を調査
python scripts/verify_bulk_accuracy.py --timing --coverage

# 異なる時刻で再検証
python scripts/verify_bulk_accuracy.py --full
```

**対策**:

- 実行時刻を変更（米国市場クローズから十分な時間が経過した時刻）
- 一時的に個別 API 強制使用: `--force-individual`

### 個別 API で API コールが不足する場合

```powershell
# 銘柄を分割して週次で更新
python scripts/cache_daily_data.py --full --chunk-size 1200 --chunk-index 1  # 月曜
python scripts/cache_daily_data.py --full --chunk-size 1200 --chunk-index 2  # 火曜
# ...以降も同様
```

### 実行が失敗し続ける場合

```powershell
# ログファイルを確認
cat logs/daily_update_YYYYMMDD_HHMMSS.log

# 手動で個別実行して問題を特定
python scripts/update_from_bulk_last_day.py --max-symbols 10
python scripts/update_cache_all.py --max-symbols 10
```

## 💡 ベストプラクティス

### 推奨実行タイミング

- **朝 6 時-10 時（日本時間）**: 米国市場クローズから十分な時間が経過しており、データが安定
- **避けるべき時刻**: 深夜 0 時-6 時（市場クローズ直後でデータが不完全な可能性）

### API 使用量の最適化

1. **通常日（月-木）**: スケジューラーで自動実行（Bulk 優先）
2. **金曜**: 品質チェック強化（`verify_bulk_accuracy.py` を事前実行）
3. **月 1 回（第 1 月曜等）**: フル検証（`--full`オプション）

### 監視とアラート

```powershell
# 統計ファイルから直近の失敗をチェック
python -c "import json; stats = json.load(open('logs/daily_update_stats.json')); recent = stats[-1]; print(f'Status: {recent[\"success\"]}, Method: {recent[\"method_used\"]}')"
```

失敗が続く場合は通知を設定することを推奨します。

## 📚 関連ドキュメント

- [`docs/README.md`](../docs/README.md): プロジェクト全体のナビゲーションハブ
- [`docs/operations/daily_schedule.md`](../docs/operations/daily_schedule.md): 日次運用の詳細
- [`scripts/update_from_bulk_last_day.py`](update_from_bulk_last_day.py): Bulk 更新の実装詳細
- [`scripts/update_cache_all.py`](update_cache_all.py): 個別 API 更新の実装詳細

## 🆘 サポート

問題が解決しない場合は、以下の情報を添えて報告してください:

1. ログファイル: `logs/daily_update_YYYYMMDD_HHMMSS.log`
2. 統計ファイル: `logs/daily_update_stats.json`（直近 3 件）
3. 検証結果: `python scripts/verify_bulk_accuracy.py --full` の出力
4. 実行環境: OS バージョン、Python バージョン、インストール済みパッケージ

---

**最終更新**: 2025-10-06
**バージョン**: 1.0.0
