# Warnings ポリシー（ドラフト）

## 目的

テスト/CI 実行時に発生する Warning を体系的に把握し、
将来の破断や品質低下を防止しつつログノイズを段階的に削減する。

## スコープ

- pytest 実行時に表示される Python 標準 Warning / ライブラリ DeprecationWarning / FutureWarning
- ランタイム動的生成の UserWarning, RuntimeWarning
- GUI/描画系 (matplotlib, Tkinter) のバックエンド警告

## 分類と優先度

| 種別                      | 例                     | 影響                   | 優先度 | 初期対応                       | 最終目標          |
| ------------------------- | ---------------------- | ---------------------- | ------ | ------------------------------ | ----------------- |
| DeprecationWarning (外部) | websockets.legacy      | 将来削除で破断         | 中     | バージョン調査・移行 TODO 発行 | 該当 API 非使用化 |
| FutureWarning (pandas)    | dtype/infer 警告       | 意図しない型変換リスク | 中〜高 | 最小再現コード作成             | 該当箇所明示処理  |
| GUI Backend Warning       | Tkinter / display 関連 | ノイズのみ             | 低     | backend "Agg" 固定             | 発生 0            |
| UserWarning (内製)        | キャッシュ検証 etc     | 設計上条件差分         | 中     | 発生条件ログに記録             | 仕様化 or 除去    |
| RuntimeWarning            | 数値計算 (ゼロ除算)    | 潜在バグ/欠損          | 高     | 早期フィルタ＋明示チェック     | 非発生            |

## 運用フェーズ

1. 計測: pytest `--disable-warnings` を外し一覧収集。出力を artifact 保存。
2. カタログ化: 重複メッセージを正規化（ファイル:行番号を除去ハッシュ化）。
3. 改善キュー生成: 優先度行列 (影響度 × 頻度)。
4. 段階的ゲート: 高優先度を `-W error` に昇格。低優先度は許容しつつ減少トレンド監視。

## CI 推奨設定（段階）

Phase 1 (現状): 失敗させない。Summary に件数だけ集計。
Phase 2: RuntimeWarning, 内製 UserWarning(特定タグ) を `-W error`。
Phase 3: 主要 DeprecationWarning (削除期日近い) を `-W error`。

## 開発者ローカルガイド

- 集中調査時: `pytest -W error::RuntimeWarning -k <target>`
- 一時抑制（正当化要必須コメント）: `warnings.filterwarnings("ignore", category=..., message=r"...", module="...")`
- 新規コードで `warnings.warn` を使う場合: 代替できるなら構造化ログ or 明示的例外。

## 記録フォーマット案

`logs/warnings_summary.jsonl` に以下 JSON を append:

```json
{"ts": "2025-10-01T12:34:56Z", "count": 57, "by_category": {"DeprecationWarning": 12, "FutureWarning": 18, ...}}
```

## 今後の TODO 候補

- 自動集計スクリプト `tools/collect_warnings.py` 追加
- CI で 直近 N 回平均と比較し増加を PR コメントに表示
- pandas FutureWarning 発生ソース行抽出ユーティリティ

## 決定しない事項（保留）

- 全警告ゼロ目標の期日設定 (初期コスト不明のため保留)
- mypy error を Warning 化するか否か (現状独立運用)

---

(ドラフト段階: 追加/修正歓迎)
