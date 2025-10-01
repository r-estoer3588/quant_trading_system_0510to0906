# 構造化ログ (NDJSON) 仕様 v1

このドキュメントは `STRUCTURED_UI_LOGS=1` + `STRUCTURED_LOG_NDJSON=1` 有効時に書き出される NDJSON (Newline Delimited JSON) 形式ログのスキーマと運用方法をまとめます。将来の破壊的変更を避けるため **バージョンフィールド `v`** を必須とし、後方互換の追加のみを基本方針とします。

## 1. 目的

1. UI (Streamlit) でのリアルタイム表示と機械可読データを両立。
2. 後工程 (解析 / 回帰比較 / 遅延 UI 再生) 向けに安定したイベントストリームを保存。
3. 既存のプレーンテキストログ量を増やさず、必要最小限のフィールドのみを記録。

## 2. 出力ファイル

- ルート: `logs/structured/` (設定取得不能時フォールバック) もしくは `STRUCTURED_LOG_NDJSON_DIR` 指定ディレクトリ。
- ファイル名: `<PREFIX>_<YYYYMMDD>_<HHMMSS>[_partN].ndjson`
  - `PREFIX`: `STRUCTURED_LOG_NDJSON_PREFIX` (デフォルト `run`)
  - ローテーション発生時: `_part1`, `_part2`, ... を付与。

## 3. 環境変数 (Feature Flags)

| 変数                           | 意味                              | デフォルト                     | 備考                           |
| ------------------------------ | --------------------------------- | ------------------------------ | ------------------------------ |
| STRUCTURED_UI_LOGS             | UI 向けに JSON オブジェクトを生成 | off                            | これのみではファイル保存しない |
| STRUCTURED_LOG_NDJSON          | NDJSON ファイル永続化を有効化     | off                            | 有効時に writer を初期化       |
| STRUCTURED_LOG_NDJSON_DIR      | 出力ディレクトリ                  | (settings.logs_dir/structured) | 存在しなければ作成             |
| STRUCTURED_LOG_NDJSON_PREFIX   | ファイル名プレフィックス          | run                            | run / backtest など切替想定    |
| STRUCTURED_LOG_BUFFER_LINES    | 行数バッファ閾値                  | 0                              | >0 でその行数に達したら flush  |
| STRUCTURED_LOG_BUFFER_FLUSH_MS | 時間バッファ閾値(ms)              | 0                              | >0 で経過時間でも flush        |
| STRUCTURED_LOG_MAX_MB          | ローテーションサイズ(MB)          | 0                              | >0 でサイズ超過後 rotate       |
| STRUCTURED_LOG_MAX_LINES       | ローテーション行数                | 0                              | >0 で行数超過後 rotate         |

### フラッシュポリシー

- バッファ条件:
  - `STRUCTURED_LOG_BUFFER_LINES > 0` の場合: 行数到達 OR フラッシュ時間経過 (`*_FLUSH_MS`) で一括書き出し。
  - `STRUCTURED_LOG_BUFFER_LINES == 0` かつ `STRUCTURED_LOG_BUFFER_FLUSH_MS > 0`: 時間のみで flush。
  - 両方 0: 従来通り行ごと即時 flush。
- `close_global_writer()` 呼び出し時は強制 flush。

### ローテーション条件

- サイズ: `STRUCTURED_LOG_MAX_MB > 0` かつ 現行ファイルサイズ >= 指定 MB → rotate。
- 行数: `STRUCTURED_LOG_MAX_LINES > 0` かつ 現行ファイル行数 >= 指定行数 → rotate。
- どちらも指定された場合は **どちらか一方でも閾値超過** で rotate。

## 4. スキーマ v1

必須フィールド:

| フィールド | 型  | 説明                                   |
| ---------- | --- | -------------------------------------- |
| v          | int | スキーマバージョン (常に 1)            |
| ts         | int | Unix epoch milliseconds (生成時刻)     |
| iso        | str | UTC ISO-8601 (末尾 'Z')                |
| lvl        | str | ログレベル (現状 "INFO" 固定)          |
| msg        | str | 元の表示メッセージ (UI と同一)         |
| elapsed_ms | int | 最初の structured ログからの経過ミリ秒 |

オプション (存在する場合):

| フィールド   | 型  | 条件                                        |
| ------------ | --- | ------------------------------------------- |
| system       | str | `SystemX` 検出時 (例: "system1")            |
| phase        | str | フェーズ推定キーワード検出時 (正規化小文字) |
| phase_status | str | `start` or `end` 推定時                     |

### 拡張方針

- 新規フィールドは「任意」追加のみ。既存必須フィールドの削除・型変更は禁止。
- 破壊的拡張が必要な場合は `v` を 2 に増加し、新旧併存期間を設ける。

## 5. フェーズ / システム推定ロジック (概要)

- `system`: 正規表現 `System(\d+)` → `system<digit>` 小文字化。
- `phase`: 優先順位付きキーワードリストをメッセージ小文字に適用し最初の一致採用。
- `phase_status`: メッセージに `start`/`end` キーワード (大小区別せず) が含まれる場合。
- 直近 `system+phase` の組を記憶し、明示的 `end` 出現時は同じ `phase` に `phase_status=end` を付与。

## 6. 使用上の注意

1. 高速連続出力時はバッファリングを有効化すると I/O が大幅に削減される。
2. 低遅延が最優先 (UI 同期等) の場合はバッファ設定を 0 に保つ。
3. ローテーションは長時間稼働や巨大 backtest でのメモリ/解析容易性のための分割目的。
4. 解析ツールは不明フィールドを無視し警告のみ記録する実装を推奨。

## 7. よくある設定例

| 用途                    | 推奨設定例                                 |
| ----------------------- | ------------------------------------------ |
| 開発デバッグ (即時反映) | BUFFER_LINES=0, FLUSH_MS=0                 |
| 日次運用 (負荷低減)     | BUFFER_LINES=50, FLUSH_MS=2000             |
| 大規模バックテスト      | BUFFER_LINES=200, FLUSH_MS=3000, MAX_MB=50 |
| 厳格分割 (行数基準)     | MAX_LINES=100000                           |

## 8. FAQ

Q. UI が落ちてもファイルは壊れない?  
A. 行単位 JSON なので途中で停止しても最後の行だけ切れる可能性があり、その行をスキップすれば残りは解析可能。

Q. phase/system が誤推定されるケースは?  
A. メッセージにキーワードが部分一致した特殊語句。頻発する場合はキーワードリスト調整で改善可能。

Q. v2 への移行基準は?  
A. 既存必須フィールドの互換性が維持できない重大変更 (例: ts 精度変更や iso タイムゾーン変更) が必要になったとき。

## 9. 今後の改善候補

- phase 推定の多言語対応 / スコアリング
- ライン圧縮 (gzip オプション) と遅延圧縮
- 書き出し統計 (書込行数, バッファ flush 回数) のメタ行追加 (別 v2 候補)

---

(最終更新) 自動生成: buffering & rotation 実装時点
