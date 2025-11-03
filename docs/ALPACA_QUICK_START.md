# Alpaca Paper Trading - クイックスタート 🚀

このガイドに従って、5分でAlpaca Paper Tradingを開始できます。

## ステップ1: Alpacaアカウント作成 (2分)

1. [Alpaca Markets](https://alpaca.markets/) にアクセス
2. 「Sign Up」でアカウント作成（無料）
3. ログイン後、「Dashboard」→「Paper Trading」→「Generate API Keys」
4. 以下の2つをコピー：
   - API Key ID (例: `PKxxxxxxxxxxxxxxxx`)
   - Secret Key (例: `xxxxx...xxxxx`)

⚠️ **Secret Keyは一度しか表示されません！** 必ずコピーしてください。

---

## ステップ2: 環境変数設定 (1分)

`.env`ファイルを開いて以下を追加：

```bash
# Alpaca Paper Trading
APCA_API_KEY_ID=PKxxxxxxxxxxxxxxxx  # ← ここにコピーしたKey IDを貼り付け
APCA_API_SECRET_KEY=xxxxxxxxxx      # ← ここにSecret Keyを貼り付け
ALPACA_PAPER=true                   # ペーパートレードモード
```

---

## ステップ3: 接続テスト (30秒)

```bash
python tools/test_alpaca_connection.py
```

成功すると以下のように表示されます：

```
✅ Client initialized successfully

📊 Account Information:
  Status: ACTIVE
  Cash: $100,000.00
  Buying Power: $400,000.00
  Portfolio Value: $100,000.00
```

---

## ステップ4: Dry-Run テスト (30秒)

実際に注文を送信せず、動作確認：

```bash
python scripts/daily_paper_trade.py --dry-run
```

シグナルが表示されれば成功！

---

## ステップ5: 実際にペーパートレード実行 (1分)

```bash
python scripts/daily_paper_trade.py
```

成功すると：
- Alpacaに注文が送信される
- `results_csv/paper_trade_log_*.csv` にログが保存される
- Slack/Discord通知（設定済みの場合）

---

## トラブルシューティング

### ❌ "Alpaca API credentials not configured"

→ `.env`ファイルの設定を確認してください

```bash
# Windowsの場合
Get-Content .env | Select-String "APCA"

# Linux/Macの場合
grep APCA .env
```

### ❌ "No signals generated"

→ 正常です！該当日にシグナルが無いだけです。

別の日付でテスト：
```bash
python scripts/daily_paper_trade.py --dry-run --verbose
```

### ❌ 注文が拒否される

→ 市場時間を確認してください（月-金 9:30-16:00 ET）

Alpacaのステータス確認：
```bash
python -c "from common.broker_alpaca import get_client; print(get_client().get_clock())"
```

---

## 次のステップ

✅ **自動化**: 毎日自動実行（[詳細ガイド](ALPACA_PAPER_TRADING.md#自動化)）
✅ **ダッシュボード**: `streamlit run apps/app_integrated.py` でポジション確認
✅ **実績分析**: トレード履歴の分析とパフォーマンス評価

---

**🎉 おめでとうございます！Alpaca Paper Tradingの準備が完了しました！**

詳細な使い方は [ALPACA_PAPER_TRADING.md](ALPACA_PAPER_TRADING.md) を参照してください。
