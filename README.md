# Quant Trading System (Streamlit)

Streamlit ベースのアプリで 7 つの売買システムを可視化・バックテストします。

## セットアップ
1. 仮想環境を作成し依存関係をインストール:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. `.env` を用意し `EODHD_API_KEY` に加え、Alpaca 連携を行う場合は
   `ALPACA_API_KEY` と `ALPACA_SECRET_KEY` を設定します。

## 実行例
- UI: `streamlit run app_integrated.py`
- Alpaca ダッシュボード: `streamlit run app_alpaca_dashboard.py`
- 日次キャッシュ: `python scripts/cache_daily_data.py`
- 簡易スケジューラ: `python -m schedulers.runner`

## テスト
```bash
pytest -q
```

## コード安全性
### zip() 関数の使用について
Python 3.10+ では、`zip()` 関数を使用する際は必ず `strict=True` パラメータを指定してください：

```python
# ❌ 危険: 長さが異なる場合に無音でデータが失われる
result = zip(list1, list2)

# ✅ 安全: 長さが異なる場合に ValueError が発生
result = zip(list1, list2, strict=True)
```

これにより、イテレータの長さが異なる場合に無音のデータ破損を防ぐことができます。プリコミットフックが自動的にこのルールをチェックします。

## 設定
優先順位は **JSON > YAML > .env**（`config/settings.py` 実装に準拠）。
推奨: `config.yaml` をベースに、秘匿値は `.env`、上書きは JSON で。

## ログ運用
`logging_utils` にてローテーション設定。容量上限と日次ローテの使い分けを明記し、
古いログのクリーンアップ方針を追加。

## ディレクトリ構成
- `app_integrated.py` – 統合UI
- `strategies/` – 戦略ラッパ
- `core/` – 各システム純ロジック
- `common/` – 共通ユーティリティ
- `config/` – 設定
- `docs/` – ドキュメント
- `tests/` – テスト

## 貢献ガイド
- コミットメッセージは命令形・現在形で72文字以内。
- 変更後は `pytest -q` を実行してテストを確認してください。
