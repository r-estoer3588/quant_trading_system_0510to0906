# mypy 強化ロードマップ（ドラフト）

## 背景
現在 mypy は `|| true` で CI を fail させない informational モード。型未整備領域が広く、一度に厳格化すると開発速度を阻害するリスクがあるため段階的導入を行う。

## 目的
- 段階的に型整合性を高め、バグ検出前倒しとリファクタ安全性向上
- 数値/日付処理・キャッシュ層・戦略ロジックの “仕様表現” としての型明確化

## 現状課題（想定）
- 大量の `Any` 流通（特に pandas DataFrame 絡み）
- 動的属性アクセス / dict payload の未型付
- 例外境界 (cache, broker) での戻り値 Optional/None ハンドリング曖昧
- 並列実行ユーティリティでの型変換（ThreadPoolExecutor 返却型）

## フェーズ計画
| フェーズ | スコープ | 設定 | ゴール | 目安期間 |
|----------|----------|------|--------|----------|
| P0 | 情報収集 | mypy --show-error-codes | エラー総数把握 | 1日 |
| P1 | コア util (cache/utils/一部) | warn-unused-ignores 有効 | 必須関数戻り値型 hint 付与 | 1週 |
| P2 | 戦略 entry points (systemX) | disallow-incomplete-defs | public 関数引数/戻り型確定 | 1-2週 |
| P3 | DataFrame生成部 | pandas-stubs 導入検討 | 主要列名/型コメント化 | 2週 |
| P4 | strict_optional | strict_optional= True | None 安全境界確立 | 2週 |
| P5 | stricter (no implicit Any) | disallow-any-generics 等 | Any 濫用 < 5% | 継続 |

## 推奨 mypy.ini 追加例（段階導入）
```ini
[mypy]
python_version = 3.11
show_error_codes = True
warn_return_any = True
warn_unused_ignores = True

# 後続フェーズで段階的に True に切り替え
# disallow_untyped_defs = True
# disallow_incomplete_defs = True
# no_implicit_optional = True
# strict_optional = True
# disallow_any_generics = True
```

## 運用ポリシー
1. 例外的に `# type: ignore[...]` を使う場合、理由コメント必須（期限付き）。
2. DataFrame 列アクセスは `TypedDict` もしくは Protocol で主要カラム宣言を検討。
3. “テスト専用” ヘルパーは後半フェーズで対象化（初期はノイズ削減）。
4. 新規コードはフェーズ進行度に関わらず P2 レベルの基準を即適用（遡及負債増加防止）。

## メトリクス追跡
- mypy エラー数（全体 / モジュール別）を CI Summary に表示
- 週次で減少傾向を可視化（スプレッドシート or JSONL）
- フェーズ完了条件: エラー閾値 (< N) を下回り 3 回連続維持

## リスクと軽減
| リスク | 緩和策 |
|--------|--------|
| 大量差分でレビュー困難 | フォルダ単位の段階 PR |
| DataFrame 型注釈の冗長化 | 必要列のみ列挙 / コメント型使用 |
| ignore 氾濫 | 週次レポートで ignore 件数集計 |

## 次のアクション候補
- P0 実測エラー収集スクリプト追加 (`tools/mypy_error_snapshot.py` 案)
- mypy.ini にベース設定反映
- CI Summary にエラー件数表示ステップ追加

---
(ドラフト: 変更提案歓迎)
