# Bulk API 品質設定 - クイックスタート

**⏱️ 所要時間**: 2 分  
**📅 最終更新**: 2025-10-12

## 🎯 この設定でできること

Bulk API のデータ品質検証を環境に応じて調整し、API コール削減と信頼性を両立します。

## ✅ デフォルト設定（推奨）

**何もしなくて OK！** 以下の最適値が自動適用されます：

- Volume 差異許容: **5.0%**（速報値誤差を許容）
- 価格差異許容: **0.5%**（戦略への影響を最小化）
- 最低信頼性スコア: **70.0%**（Bulk API 使用基準）

**結果**: 信頼性スコア **100%** で Bulk API を活用 → API コール大幅削減 ✨

## 🔧 カスタマイズが必要な場合

### 1. `.env` ファイルを作成（初回のみ）

```bash
# プロジェクトルートに .env ファイルを作成
cp .env.example .env
```

### 2. 環境別設定例を追加

#### 🏭 本番環境（推奨）

```bash
# .env に追加
BULK_API_VOLUME_TOLERANCE=5.0
BULK_API_PRICE_TOLERANCE=0.5
BULK_API_MIN_RELIABILITY=70.0
```

#### 🧪 開発/テスト環境

```bash
# .env に追加
BULK_API_VOLUME_TOLERANCE=10.0    # 緩和
BULK_API_PRICE_TOLERANCE=1.0      # やや緩和
BULK_API_MIN_RELIABILITY=60.0     # 低基準
```

#### ⚡ 高頻度取引

```bash
# .env に追加
BULK_API_VOLUME_TOLERANCE=3.0     # 厳格
BULK_API_PRICE_TOLERANCE=0.3      # 極めて厳格
BULK_API_MIN_RELIABILITY=90.0     # 高基準
```

## 📊 動作確認（2 分で完了）

### ステップ 1: 現在の設定を確認

```powershell
python -c "from config.environment import get_env_config; env = get_env_config(); print(f'Volume: {env.bulk_api_volume_tolerance}%\n価格: {env.bulk_api_price_tolerance}%\n信頼性: {env.bulk_api_min_reliability}%')"
```

**期待される出力**:

```
Volume: 5.0%
価格: 0.5%
信頼性: 70.0%
```

### ステップ 2: 品質検証を実行

```powershell
python scripts/verify_bulk_accuracy.py
```

**期待される出力**:

```
✅ 信頼性スコア: 100.0%
👍 Bulk APIは高品質です。安心して使用できます。
```

## 🚨 トラブルシューティング

### 信頼性スコアが低い（70%未満）

**症状**:

```
❌ 信頼性スコア: 65.0%
🚨 Bulk APIの品質が低いです。個別API使用を推奨します。
```

**対策 1**: 実行時刻を確認

```powershell
python scripts/verify_bulk_accuracy.py --timing
```

→ **推奨時間帯**: 日本時間 6:00〜10:00

**対策 2**: Volume 許容範囲を緩和

```bash
# .env に追加
BULK_API_VOLUME_TOLERANCE=7.0
```

**対策 3**: 信頼性基準を調整

```bash
# .env に追加
BULK_API_MIN_RELIABILITY=65.0
```

### 個別 API にフォールバックされる

**ログ確認**:

```
[WARNING] ⚠️ Bulk APIの品質が低いため、個別APIを使用します
```

**即座の対処**（一時的）:

```powershell
$env:BULK_API_MIN_RELIABILITY="65.0"
python scripts/run_all_systems_today.py --parallel --save-csv
```

**恒久的対処**: `.env` ファイルに設定を追加（上記参照）

## 📖 詳細ドキュメント

より詳しい説明が必要な場合:

- **包括ガイド**: [Bulk API 品質ガイド](operations/bulk_api_quality_guide.md)
- **環境変数一覧**: [環境変数ドキュメント](technical/environment_variables.md#7-bulk-apiデータ品質検証)
- **変更履歴**: [2025-10-12 改善サマリー](changes/2025-10-12_bulk_api_quality_improvement.md)

## 💡 よくある質問

**Q: デフォルト設定のままで大丈夫？**  
A: はい！ほとんどの環境で最適です。変更不要で信頼性 100% を達成します。

**Q: Volume を 5% 許容して戦略に影響は？**  
A: 価格データ（OHLC）は 0.5% で厳格に検証しているため、影響はありません。Volume のみの緩和です。

**Q: 設定を間違えたら？**  
A: `.env` から該当行を削除すれば、デフォルト値（推奨設定）に戻ります。

**Q: 環境変数が反映されない**  
A: Python プロセスを再起動してください。`@lru_cache` でキャッシュされている場合があります。

---

**🎉 設定完了！** 日次更新処理で Bulk API が効率的に使用されます。
