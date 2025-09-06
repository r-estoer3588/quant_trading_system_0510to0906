## ドキュメント総覧（System1〜7）

### 共通前提
- データ: EODHD 日足（初回 /eod、以後 /eod-bulk-last-day/US）
- 必須: SPY キャッシュ、ブラックリスト適用、Asia/Tokyo運用
- ベース指標: SMA, ATR, RSI, ROC, HV … (`common/cache_manager.py`)

### 設定の優先順位
- JSON > YAML > .env (`config/settings.py`)

### バックテスト前提
- 手数料/スリッページ/遅延/出来高・価格制約、失敗時挙動を明記

### KPI
- Win%, PF, Sharpe, MDD(USD), CAGR, MaxDD期間, Trades, Hit連続 など
