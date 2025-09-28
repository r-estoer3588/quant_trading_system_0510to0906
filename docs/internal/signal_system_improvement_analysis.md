# シグナル実行システム改善調査レポート

本レポートは、シグナル実行システムに関する9項目の改善テーマについて、現状把握と改善案、優先順位を整理したものです。コード修正は伴わず、実装に向けた検討材料をまとめています。

## 1. エラーメッセージ体系の再構築
### 現状
- CLI ログ関数 `_log` はキーワードフィルタを通した任意文字列を標準出力とファイルへ流す仕組みで、メッセージの型や ID は統一されていません。【F:scripts/run_all_systems_today.py†L560-L663】
- 個別スクリプトでも emoji＋自由形式のエラーメッセージを直接 `print` しており、コード化されていません（例: SPY 復旧スクリプト）。【F:scripts/recover_spy_cache.py†L20-L109】
- `common/structured_logging.py` には JSON 形式の構造化ロガーが存在するものの、today ランナーや CLI ルートでは採用されていません。【F:common/structured_logging.py†L1-L120】

### 改善案
1. **エラーコード命名規則の導入**: `common/exceptions.py` に共通の `TradingError` 階層があるため、ここに "AAA123E" 形式のコード列挙を追加し、`dataclass` ベースの `ErrorDescriptor` で ID・深刻度（I/E/A/W）・既定メッセージを定義する。
2. **ログエンリッチメント**: `_log` の内部でコード付与済みメッセージ (例: `[SGL001E] インジケーター計算失敗 ...`) を強制し、UI/CLI に同じメッセージを届ける。構造化ロガー (`TradingSystemLogger`) を wrap して JSON 出力も同時に残す。
3. **フェーズ別トレース ID**: `StageMetrics` のスナップショットにエラーフェーズを追加し、各例外捕捉箇所で「どのフェーズで失敗したか」を記録。`ContextVar` でフェーズをバインドし `_log` / `_emit_ui_log` に付加する。
4. **移行戦略**: 既存メッセージを網羅したマッピング表を作成し、段階的にコード化。初期段階では従来メッセージに `[LEGACY]` を付けるなどして移行完了を可視化する。

## 2. system6 の実行時間過剰問題
### 現状
- System6 の指標準備は、プロセスプールを用いる経路と逐次経路の双方で `AverageTrueRange` や rolling 計算を毎銘柄ごとに実行しており、キャッシュ再利用時も 50 日分の再計算が発生します。【F:core/system6.py†L145-L399】
- 進捗ログは存在するものの、実測時間やワーカーごとのスループットが記録されず、遅延要因の切り分けが困難です。【F:core/system6.py†L200-L411】
- ProcessPool 経路でも子プロセスからの進捗キュー送信以外に詳細なメトリクス収集は行っていません。【F:core/system6.py†L98-L143】

### 改善案
1. **メトリクス収集の標準化**: `common/structured_logging.MetricsCollector` を system6 に注入し、`prepare_data_vectorized_system6` と `generate_candidates_system6` の各バッチで `elapsed`・`rows_per_second`・`cache_hit率` を JSON ログとして蓄積する。
2. **プロファイル実験の自動化**: プロセスプール利用時は `multiprocessing` の `set_start_method("spawn")` と合わせて CPU／メモリ使用量を `psutil` で取得し、ログへ添付する。ワーカー数ごとの実行時間比較スクリプトを `scripts/bench_system6.py` に追加。
3. **キャッシュ差分の最適化**: Fast-path でも ATR・dollarvolume の計算が全行再計算になるため、キャッシュ側に最新値を保持して差分更新する、もしくは `numba`／`numpy` ベクトル化を検討する。最終的には `CacheManager` に system6 専用指標を永続化し、`reuse_indicators=True` 時に再計算を避ける。
4. **ボトルネック検知**: 上記メトリクスから閾値（例: 1銘柄あたり 150ms 超過）でアラートを出し、`BatchSizeMonitor` の動的バッチ調整結果をログへ可視化する。

## 3. `recover_spy_cache.py` のエラー解析
### 現状
- `EODHD_API_KEY` 未設定や API レスポンス空など、複数の早期 return パスが `print` のみで通知されています。【F:scripts/recover_spy_cache.py†L20-L87】
- CacheManager への書き込み失敗時は `traceback` を標準エラーへ吐きますが、呼び出し元に伝播しないため再試行や後続処理に活かせません。【F:scripts/recover_spy_cache.py†L95-L106】
- 例外種別ごとの再現条件（HTTP タイムアウト、必須列欠落など）が整理されておらず、UI 側でのガイダンスに利用できません。

### 改善案
1. **例外種別の分類**: HTTP レイヤー（`requests.exceptions`）、スキーマ検証、CacheManager 例外を捕捉して `TradingError` にラップし、コード付きで再送する。
2. **再試行ポリシー**: ネットワークエラー時は指数バックオフを適用し、3 回失敗で `[SPY001E]` のようなコードを返却。API レート制限検知時は待機時間をログへ明示する。
3. **UI 連携**: 成功時・失敗時の詳細を `logs/recover_spy.jsonl` へ構造化出力し、Streamlit 側から最新結果を表示できるようにする。

## 4. CLI 出力ログの不足対応
### 現状
- `_log` 関数は `SHOW_INDICATOR_LOGS` が無効な場合、インジケーター関連ログやバッチ時間をフィルタしてしまい、CLI 上の追跡が困難です。【F:scripts/run_all_systems_today.py†L584-L645】
- UILogger でも短時間に重複したメッセージを抑制しており、完全なログ履歴はファイル側にしか残りません。【F:apps/app_today_signals.py†L1450-L1520】

### 改善案
1. **フィルタ挙動の可視化**: `_log` のフィルタ条件をログに出力するデバッグモード (`TODAY_LOG_DEBUG=1`) を用意し、どのメッセージが除外されたかを後追いできるようにする。
2. **CLI 詳細ログレベル**: `--verbose` フラグや環境変数で `SHOW_INDICATOR_LOGS` を強制的に有効にし、段階的に抑制するキーワードをユーザーが制御できるようにする。
3. **ログリングバッファ**: UILogger にリングバッファ（例: 最新 500 行）を保持させ、CLI から `--dump-log-buffer` オプションで出力できるようにする。

## 5. CLI と UI のログ同期問題
### 現状
- CLI→UI へのログ転送は `_emit_ui_log` で ContextVar `_LOG_FORWARDING` を用いて二重出力を防いでいますが、UI 側の `UILogger` が CLI に逆送するため、タイミングによりログが欠落または順序入れ替わる場合があります。【F:scripts/run_all_systems_today.py†L482-L629】【F:apps/app_today_signals.py†L1450-L1520】
- `GLOBAL_STAGE_METRICS` のイベントキューをドレインする `_drain_stage_event_queue` が定義されているものの、呼び出し箇所がなく、ステージ進捗が UI へ反映されないケースがあります。【F:scripts/run_all_systems_today.py†L492-L549】

### 改善案
1. **ログイベントバス**: ContextVar の代わりに `queue.Queue` ベースのイベントストリームを導入し、CLI・UI 双方が購読する。順序保証と back-pressure を実現する。
2. **ステージイベントの自動ドレイン**: today ランナーの主要ループに `_drain_stage_event_queue()` を組み込み、1〜2 秒ごとに UI へ最新進捗を配送する。また、Streamlit 側でポーリングする API を提供する。
3. **メッセージ ID 付与**: `_log` と `UILogger.log` 双方でメッセージ ID を付加し、重複抑止を ID ベースに切り替えることで順序ズレ検知を容易にする。

## 6. UI のログと進捗率の乖離
### 現状
- ステージ更新は `StageTracker.update_stage` で 0.5 秒以内の同一イベントを無視する仕組みがあり、短時間に連続送信されると進捗バーが更新されません。【F:apps/app_today_signals.py†L1218-L1285】
- `common/today_signals.get_today_signals_for_strategy` は 0/25/50/75/100% の離散進捗を送出するため、実際の処理時間とバー表示が乖離しやすい構造になっています。【F:common/today_signals.py†L2361-L2557】

### 改善案
1. **サブステージ導入**: フィルタリング・セットアップ内でさらに細分化した進捗（例: 10% 刻み）を計算し、`StageMetrics` へ記録する。
2. **イベントスロットリング調整**: `StageTracker` の重複抑止間隔を可変（デフォルト 0.2 秒、UI 設定で変更可能）にし、処理速度に応じて調整できるようにする。
3. **ログと進捗のひも付け**: `_log` にフェーズ ID を付け、UI 側で該当フェーズの進捗バー横に最新ログを表示するミニタイルを実装する。

## 7. UI 表示項目欠落 (tgt/filpass/stupass/trdlist/entry/exit)
### 現状
- `StageMetrics.record_stage` は progress=0 のときに target を更新しますが、`_stage` から渡されるシステム名が空文字のため、UI 側では対象システムに紐づく target が登録されません。【F:scripts/run_all_systems_today.py†L3553-L3645】
- 直接 UI コールバック (`cb2`) を呼ぶ箇所でも 0% イベントが送られないため、Tgt が `None` のままとなります。【F:scripts/run_all_systems_today.py†L2983-L3261】
- 進捗 50% 時点でセットアップ件数を更新しても、ターゲットが初期化されていないと `StageTracker` の表示が `-` のままです。【F:apps/app_today_signals.py†L1240-L1285】

### 改善案
1. **システム名伝搬の修正**: `_stage` にシステム名を引数として渡すよう today ランナーを改修し、`GLOBAL_STAGE_METRICS` へ正しいキーで記録する。
2. **初期化イベントの明示化**: フィルター開始時に `cb2(system, 0, total_symbols, …)` を送る共通ユーティリティを追加し、UI 側が Tgt を確実に受け取れるようにする。
3. **ステージスナップショットのフォールバック**: UI 初期化時に `GLOBAL_STAGE_METRICS.all_snapshots()` から target が取得できない場合は、`StageTracker` が `universe_total` をフィルター通過数で推定するロジックを追加する。

## 8. UI 表示とフェーズ進捗の同期改善
### 現状
- `_drain_stage_event_queue` が未呼び出しのため、プロセスプール経由で送信された進捗イベントが UI へ反映されないままキューに滞留する可能性があります。【F:scripts/run_all_systems_today.py†L492-L549】
- UI 側は `StageTracker._apply_snapshot` で `GLOBAL_STAGE_METRICS` のスナップショットを再適用していますが、`refresh_all` を都度呼び出すため描画負荷が高く、リアルタイム更新に不向きです。【F:apps/app_today_signals.py†L1203-L1386】

### 改善案
1. **定期ポーリングの導入**: Streamlit セッション側に `st.session_state` を利用したポーリングタスクを追加し、一定間隔で `GLOBAL_STAGE_METRICS.drain_events()` を呼び出す。
2. **差分更新**: `StageTracker` に差分適用機構（更新されたシステムのみ描画）を実装し、UI リフレッシュコストを削減する。
3. **リアルタイム可視化**: WebSocket／`st.experimental_data_editor` 等を利用して、フェーズ完了スナップショットを即時に描画できる仕組みを検討する。

## 9. SPY 不在時の早期システム停止
### 現状
- SPY データが `basic_data` に無い場合は警告ログを出すのみで処理が継続され、System4 以外は空のデータフレームで最後まで進みます。【F:scripts/run_all_systems_today.py†L3223-L3261】【F:scripts/run_all_systems_today.py†L3289-L3376】
- `System1` のセットアップ内訳では SPY > SMA100 判定を行うものの、SPY 不在時でもゼロ扱いで続行されます。【F:scripts/run_all_systems_today.py†L2942-L3021】

### 改善案
1. **起動時チェック**: `_prepare_symbol_universe` 完了直後に SPY キャッシュの鮮度を検証し、欠落していればフェーズ 0 で停止・復旧ガイドを提示する。
2. **システム別ガード**: System1/4/7 など SPY 依存システムは `StageMetrics` に「SPY 欠落」ステータスを記録し、UI に赤色表示と復旧導線を出す。
3. **時間効率試算**: SPY 欠落時に残りシステムをスキップした場合の平均節約時間（例: 5〜7 分）を過去ログから算出し、早期停止の効果を定量化する。

## 改善ロードマップ（優先度順）
1. **ステージ同期と UI 指標欠落の解消 (項目5/7/8)**
   - 進捗が UI へ届かない問題は他施策の前提となるため、`_drain_stage_event_queue` の実装・システム名伝搬の修正を最優先で行う。
2. **SPY 依存チェックの強化 (項目9)**
   - データ欠落時に無駄な計算を避けることで、残りの調査・改善タスク実行時間を確保する。
3. **ログ体系の標準化 (項目1/4/5)**
   - エラーコード導入とログバス整備により、以降のパフォーマンス・品質改善の検証が容易になる。
4. **system6 パフォーマンス計測基盤整備 (項目2)**
   - 実計測値を取得できるようになった段階で、キャッシュ最適化やアルゴリズム改善を計画する。
5. **SPY 復旧スクリプトと関連オペレーション (項目3)**
   - エラーハンドリングを整備し、運用チームが迅速に復旧できる体制を整える。
6. **UI 進捗表示の精緻化 (項目6)**
   - フェーズ同期が安定後、細分化した進捗やログ連携を実装してユーザー体験を向上させる。

---
本レポートを足掛かりに、各項目の実装計画と工数見積もりを進めてください。
