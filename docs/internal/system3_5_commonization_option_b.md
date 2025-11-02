<!-- docs/internal/system3_5_commonization_option_b.md -->

# System3/5 共通処理 抽出設計（Option B）

この文書は、System3（大陰線/反発系）と System5（ボラ・出来高系）の「候補準備〜ランキング〜診断更新」フローを、最小リスクで共通化するための設計案です。Phase-A で導入した `common/system_candidates_utils.py` を土台に、重複をさらに減らし、診断の一貫性とテスト容易性を高めます。

## ねらい（なぜやるか）

- 実装の重複を減らし、修正の波及を抑える。
- 診断更新（件数・理由・最終件数）を「ランキング確定後」に統一し、ズレを防ぐ。
- System3/5 固有のしきい値や列名は「戦略パラメータ」として明示し、共通関数に渡すだけで使えるようにする。

## スコープ（今回やること／やらないこと）

- やること
  - ランキング前の「入力正規化」
  - しきい値ベースのフィルタリングとゼロ件時の診断出力
  - ランキング確定後の診断更新（`set_diagnostics_after_ranking` の拡張活用）
  - 除外理由の記録（理由名→銘柄リスト）の共通ハンドリング
- やらないこと
  - 戦略ロジック（指標の定義やしきい値の意味）は変更しない
  - 出力スキーマ（UI/最終配分 API）を変更しない

## ミニマム構成（提案する関数群）

- `common/system_candidates_utils.py` に以下を追加（既存非破壊）
  1) `prepare_ranking_input(df, label_date, required_cols) -> tuple[DataFrame, dict]`
     - 目的: ラベル日での行抽出、必要列の存在チェック、件数計測。
     - 返り値: (ランキング対象の DataFrame, 診断補助カウンタ dict)
  2) `apply_thresholds(df, rules) -> tuple[DataFrame, dict, dict]`
     - 目的: しきい値ルールの適用（例: drop3d >= 0.125 など）。
     - 返り値: (フィルタ後 df, {理由→件数}, {理由→シンボル配列})
  3) `finalize_ranking_and_diagnostics(diag, ranked_df, ranking_source, extras=None)`
     - 目的: 既存 `set_diagnostics_after_ranking` を内包拡張し、`diagnostics_extra` 用の集計値（例: ranking_input_counts, thresholds, exclude_reasons）をまとめて付与するユーティリティ。

これらはすべて「オプショナルで利用」できる形にし、System3/5 から段階的に置き換え可能にします。

## I/O 契約（コンパクト）

- 入力
  - DataFrame: 少なくとも `date`, `symbol` と戦略固有の指標列（例: `drop3d`, `atr_ratio` など）を含む
  - `label_date`: ランキング基準日（latest_only の場合はモード日）
  - `rules`: しきい値辞書（例: `{ 'drop3d': { 'op': '>=', 'value': 0.125 } }`）
- 出力
  - ランキング対象 DataFrame / フィルタ後 DataFrame
  - 除外理由の件数と銘柄（辞書）
  - 診断の更新（`setup_predicate_count`, `ranked_top_n_count`, `final_top_n_count`, `ranking_source`）
- エラー/境界
  - 必須列が欠けていれば空 DataFrame とし、`diagnostics_extra` に不足列を記録
  - しきい値適用で 0 件になった場合、ゼロ件理由を `diagnostics_extra.ranking_zero_reason` に記録

## 適用手順（小さく進める）

1) テスト追加（先行）
   - System3/5 の「ゼロ件時の診断」「1 件以上時の件数整合」を検査する軽量テストを追加。

2) System3 に適用（フラグ駆動）
   - 既存コードの前処理〜診断更新を、上記ユーティリティで置換。
   - `diagnostics_extra` の保持項目（`ranking_input_counts`,`thresholds`,`exclude_reasons`）を現状と同等に埋める。

3) System5 に適用（同様）
   - しきい値セットは戦略関数側で定義し、ユーティリティに渡すだけにする。

4) 後始末
   - 重複ヘルパの削除/統合（各 system ファイル内の小ヘルパを utils 側へ）
   - ドキュメントとサンプル更新（この設計書への追記）

## 影響とリスク

- 影響範囲は System3/5 の前処理・診断部分に限定。ランキングスコアの式や列名は維持。
- 既存ログ文言は維持する（文言が変わると E2E のログ照合に影響）
- リスク: しきい値比較の不等号向きミス、日付基準の取り違え。→ 先行テストでガード。

## ロールバック方針

- フラグを `False` に戻せば従来実装に切替え可能な構成にする（小さな if-guard）。
- 置換差分は関数境界で完結するように分割する。

## 既存との整合

- `set_diagnostics_after_ranking` の契約は不変。最終更新は常に「ランキング確定後」。
- `diagnostics_extra` に既存で出力しているキー（System3 の `ranking_input_counts`, `thresholds` 等）はそのまま残す。

---

以上。実装は System3 から段階的に当て込み、テストが緑のままなら System5 に横展開します。
