# 🔥 Alpaca Paper Trading 連携ガイド

Alpaca Paper Tradingを使った仮想トレード実績の蓄積方法を解説します。

## 📋 目次

1. [セットアップ](#セットアップ)
2. [使い方](#使い方)
3. [トレード履歴の確認](#トレード履歴の確認)
4. [自動化](#自動化)
5. [トラブルシューティング](#トラブルシューティング)

---

## セットアップ

### 1. Alpacaアカウント作成

1. [Alpaca](https://alpaca.markets/)にアクセス
2. アカウントを作成（無料）
3. Paper Trading用のAPIキーを取得

### 2. APIキー設定

`.env`ファイルに以下を追加：

```bash
# Alpaca Paper Trading
APCA_API_KEY_ID=your_key_id_here
APCA_API_SECRET_KEY=your_secret_key_here
ALPACA_PAPER=true  # ペーパートレード（仮想）モード
```

### 3. 接続テスト

```bash
python tools/test_alpaca_connection.py
```

成功すると、アカウント情報とポジションが表示されます。

---

## 使い方

### 方法1: Streamlit UI（推奨）🌟

最も簡単で視覚的な方法です：

```bash
# シグナル生成＋Alpaca送信
streamlit run apps/app_today_signals.py
```

**手順：**
1. サイドバーで設定（資金配分、ペーパートレードモードなど）
2. 「Generate Signals」ボタンをクリック
3. シグナルが表示されたら「Alpaca自動発注」セクションで送信
4. **トレード履歴**セクションで過去の注文を確認

**UI の主な機能：**
- ✅ リアルタイムシグナル生成
- ✅ Alpaca注文送信（成功/失敗のサマリー表示）
- ✅ トレード履歴の可視化
- ✅ 統計情報（成功率、システム別内訳など）
- ✅ CSVエクスポート

### 方法2: コマンドライン（自動化向け）

```bash
# シグナル生成＋Alpaca送信（ペーパートレード）
python scripts/run_all_systems_today.py --alpaca-submit --save-csv

# 本番取引（要注意！）
python scripts/run_all_systems_today.py --alpaca-submit --live --save-csv
```

**主なオプション：**
- `--alpaca-submit`: Alpacaへ注文送信を有効化
- `--live`: 本番取引モード（デフォルトはペーパートレード）
- `--tif DAY`: Time In Force（DAY, GTC, CLS など）
- `--order-type market`: 注文タイプ（market, limit）
- `--save-csv`: CSVに結果を保存

### 方法3: Alpacaダッシュボード（ポジション管理）

既存ポジションの監視・手動決済：

```bash
streamlit run apps/dashboards/app_alpaca_dashboard.py
```

**機能：**
- 📊 ポジション一覧
- 📈 損益サマリー
- ⏰ 保有日数管理
- 🚀 手動決済ボタン
- 🤖 自動ルール設定

---

## トレード履歴の確認

### UIで確認（最も簡単）

`apps/app_today_signals.py` の **📊 トレード履歴** セクションで：
- 過去の注文履歴を表示
- 期間フィルタ（7日/30日/90日など）
- 成功/失敗の統計
- システム別内訳
- CSVエクスポート

### ログファイルで確認

トレード履歴は `data/trade_history.jsonl` に自動保存されます：

```bash
# 最新10件を表示
Get-Content data/trade_history.jsonl | Select-Object -Last 10
```

### Pythonで分析

```python
from common.trade_history import get_trade_history_logger

logger = get_trade_history_logger()

# 過去30日の統計
stats = logger.get_stats(days=30, paper_only=True)
print(f"成功率: {stats['successful_orders'] / stats['total_orders'] * 100:.1f}%")

# 履歴DataFrame取得
df = logger.get_recent_trades(limit=100)
print(df)
```

---

## 自動化

### 1. Windows タスクスケジューラ

市場終了15分前（15:45 ET）に自動実行する例：

```powershell
# タスク作成スクリプト
$action = New-ScheduledTaskAction `
    -Execute "python" `
    -Argument "c:\Repos\quant_trading_system\scripts\run_all_systems_today.py --alpaca-submit --save-csv" `
    -WorkingDirectory "c:\Repos\quant_trading_system"

$trigger = New-ScheduledTaskTrigger `
    -Daily -At "15:45"

Register-ScheduledTask `
    -TaskName "AlpacaPaperTrade" `
    -Action $action `
    -Trigger $trigger `
    -Description "Daily Alpaca Paper Trading"
```

### 2. GitHub Actions

`.github/workflows/daily_trade.yml`:

```yaml
name: Daily Paper Trading

on:
  schedule:
    - cron: '45 19 * * 1-5'  # Mon-Fri 15:45 ET (19:45 UTC)
  workflow_dispatch:  # 手動実行も可能

jobs:
  trade:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python scripts/run_all_systems_today.py --alpaca-submit --save-csv
        env:
          APCA_API_KEY_ID: ${{ secrets.APCA_API_KEY_ID }}
          APCA_API_SECRET_KEY: ${{ secrets.APCA_API_SECRET_KEY }}
          ALPACA_PAPER: true
```

---

## トラブルシューティング

### ❌ "Alpaca API credentials not configured"

**原因**: `.env`にAPIキーが設定されていない

**解決策**:
```bash
# .env ファイルを確認
cat .env | grep APCA

# 必要な変数が設定されているか確認
APCA_API_KEY_ID=PK...
APCA_API_SECRET_KEY=...
```

### ❌ "No signals generated"

**原因**: 該当日にシグナルが無い（正常）

**確認**:
```bash
# 詳細ログで確認
python scripts/run_all_systems_today.py --test-mode mini --save-csv
```

### ❌ 注文が拒否される

**原因**:
- 市場時間外
- 銘柄が取引不可
- 資金不足（ペーパーでは通常100万ドル）

**確認**:
```bash
# Alpacaのステータス確認
python tools/test_alpaca_connection.py
```

### ❌ 履歴が表示されない

**原因**: `data/trade_history.jsonl` が存在しない、または空

**解決策**:
```bash
# ファイルの存在確認
ls data/trade_history.jsonl

# 手動で初回注文を送信
python scripts/run_all_systems_today.py --alpaca-submit --test-mode mini
```

---

## 📊 実績確認

### ダッシュボードで確認

```bash
streamlit run apps/app_integrated.py
```

→ "ポジション管理"タブで現在のポジションを確認

### ログファイル確認

```bash
# 最新のトレードログ
Get-ChildItem results_csv/*.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# 内容確認
Import-Csv results_csv/today_signals_2025-11-03.csv
```

---

## 🔗 参考リンク

- [Alpaca Paper Trading](https://alpaca.markets/docs/trading/paper-trading/)
- [Alpaca Python SDK](https://github.com/alpacahq/alpaca-py)
- [プロジェクトREADME](../README.md)

---

**🎉 Happy Paper Trading!**
