# トレーディングシステム デバッグガイド

## 1. 基本的なデバッグフロー

### エラーコード体系の活用

当システムは `AAA123E` 形式のエラーコードを採用しています。各部分の意味は：

- **AAA**: エラーカテゴリ（DAT=データ、SIG=シグナル、ALC=アロケーション、SYS=システム、NET=ネットワーク）
- **123**: 連番（カテゴリ内の一意の番号）
- **E/W/I**: 重大度（E=エラー、W=警告、I=情報）

例：`DAT001E`はデータ関連の重大なエラー（キャッシュファイル未発見など）

### CLI フラグによるデバッグ強化

```bash
# 基本的な詳細ログ出力
python scripts/run_all_systems_today.py --verbose

# 完全デバッグモード (詳細なトレースとデバッグ情報)
python scripts/run_all_systems_today.py --debug-mode

# すべてのフェーズでトレースIDを使用 (問題の追跡に最適)
python scripts/run_all_systems_today.py --trace-all

# 特定のシステムのみテスト
python scripts/run_all_systems_today.py --system system3 --verbose

# テストモードでの高速検証 (2秒程度で完了)
python scripts/run_all_systems_today.py --test-mode mini --skip-external --benchmark

# 複数フラグの組み合わせ
python scripts/run_all_systems_today.py --verbose --trace-all --test-mode mini
```

## 2. ログファイル解析

### 構造化ログファイルの場所

```
logs/                      # ログのルートディレクトリ
├── application.jsonl      # メインアプリケーションログ (JSON形式)
├── errors.jsonl           # エラーのみのログ
├── performance.jsonl      # パフォーマンス測定データ
├── progress_today.jsonl   # 進捗イベントログ
└── metrics/               # 詳細メトリクス
    └── metrics.jsonl      # メトリクスデータ
```

### JSON ログの解析

```bash
# 最新のエラーのみ表示
cat logs/errors.jsonl | tail -n 20 | python -m json.tool

# 特定のトレースIDに関連するすべてのログを抽出
cat logs/application.jsonl | grep "trace_52aa4572" | python -m json.tool

# 特定のフェーズのログのみ表示
cat logs/application.jsonl | grep '"phase":"FILTERS"' | python -m json.tool

# 特定のエラーコードの出現回数をカウント
cat logs/errors.jsonl | grep -o 'DAT[0-9]\{3\}[EWI]' | sort | uniq -c

# パフォーマンス分析 - 処理時間の長い操作トップ10を表示
cat logs/performance.jsonl | jq -c 'select(.extra.duration > 1.0)' | \
  jq -s 'sort_by(.extra.duration) | reverse | .[0:10]' | jq '.'
```

## 3. トレース ID 機能の活用

### トレース ID の仕組み

システム実行時に各フェーズに一意のトレース ID が割り当てられます。これにより、関連する処理を追跡できます。

```
🆔 実行トレースID: run_52aa4572  # メイン実行ID
  └── 🔄 SYMBOLS: symbols_7bc31a8e  # フェーズID
      └── 📊 universe_filter: filter_19e45f2d  # サブ処理ID
```

### トレース ID による追跡

```bash
# 特定のトレースIDに関連するすべてのログエントリを表示
cat logs/*.jsonl | grep "52aa4572" | python -m json.tool

# フェーズ遷移を時系列で追跡
cat logs/application.jsonl | jq -c 'select(.event_type == "phase_transition")' | \
  jq -s 'sort_by(.timestamp)'
```

## 4. 一般的なエラーとトラブルシューティング

### データキャッシュ階層に関する問題

**症状**: `[DAT001E] Cache file not found` や `[DAT002E] Cache hierarchy broken`

**解決策**:

1. キャッシュの階層構造を確認: `rolling` → `base` → `full_backup`
2. キャッシュの更新:

```bash
# キャッシュ更新（シリアル実行）
python -m run_task -w "c:\Repos\quant_trading_system" -i "shell: Update Cache All (Serial)"

# パラレル実行（高速）
python -m run_task -w "c:\Repos\quant_trading_system" -i "shell: Update Cache All (Parallel)"

# rollingのみ再構築
python -m run_task -w "c:\Repos\quant_trading_system" -i "shell: Build Rolling Only"
```

### SPY データ関連の問題

**症状**: `[DAT004E] SPY data corruption` または `[NET002E] Data download timeout`

**解決策**:

```bash
# SPY復旧スクリプトの実行
python tools/recover_spy_cache.py

# リトライ回数を増やして再試行
RETRY_MAX_ATTEMPTS=5 python tools/recover_spy_cache.py --verbose
```

### シグナル生成エラー

**症状**: `[SIG003E] Setup condition error` や `[SIG001E] Invalid signal configuration`

**解決策**:

```bash
# 特定のシステムだけを詳細モードで実行
python scripts/run_all_systems_today.py --system system2 --verbose --trace-all

# テストモードでの検証
python scripts/run_all_systems_today.py --system system2 --test-mode mini --debug-mode
```

## 5. デバッグモード別機能一覧

| フラグ             | 環境変数            | 主な機能                                   |
| ------------------ | ------------------- | ------------------------------------------ |
| `--verbose`        | `VERBOSE=1`         | 詳細なログ出力、インジケーターログの表示   |
| `--debug-mode`     | `TODAY_LOG_DEBUG=1` | 内部変数のダンプ、フィルタ条件の詳細表示   |
| `--trace-all`      | `TRACE_ALL=1`       | すべての処理にトレース ID 付与、階層的追跡 |
| `--test-mode mini` | `TEST_MODE=mini`    | 少数銘柄での高速テスト実行                 |
| `--skip-external`  | `SKIP_EXTERNAL=1`   | 外部 API コールをスキップ                  |
| `--benchmark`      | `BENCHMARK=1`       | パフォーマンス計測の有効化                 |

## 6. UI デバッグ

### Streamlit UI のデバッグ

UI にエラーが表示される場合:

1. バックエンドログとフロントエンドログを比較:

```bash
# バックエンドログ確認
cat logs/application.jsonl | tail -n 100 | python -m json.tool

# Streamlit UIログ（別ターミナルで）
streamlit run apps/app_integrated.py --logger.level=debug
```

2. UI リングバッファの確認:

```python
# Pythonインタラクティブシェルでの確認方法
from common.structured_logging import get_trading_logger
logger = get_trading_logger()
last_logs = logger.get_ring_buffer(last_n=50)  # 最新50件
for log in last_logs:
    print(f"{log['timestamp']} [{log['level']}] {log['message']}")
```

### 進捗イベント問題

進捗バーが更新されない場合:

1. イベントキューを確認:

```bash
# 進捗イベントログ確認
cat logs/progress_today.jsonl | tail -n 20 | python -m json.tool
```

2. `progress_events.py`のデバッグモード有効化:

```bash
ENABLE_PROGRESS_EVENTS=1 PROGRESS_DEBUG=1 streamlit run apps/app_today_signals.py
```

## 7. 高度なデバッグ技術

### リトライポリシーのカスタマイズ

特定の操作でリトライ動作を調整:

```python
from common.trading_errors import RetryPolicy, retry_with_backoff

# リトライポリシーのカスタマイズ
custom_policy = RetryPolicy(
    max_attempts=5,        # 最大試行回数
    base_delay=1.0,        # 初回遅延(秒)
    max_delay=30.0,        # 最大遅延(秒)
    backoff_factor=2.0     # バックオフ係数
)

# カスタムポリシーでの関数実行
result = retry_with_backoff(my_function, policy=custom_policy)
```

### CacheManager のデバッグ

キャッシュ関連の問題を診断:

```python
from common.cache_manager import CacheManager
from common.logging_utils import configure_logging

# ロガー設定
configure_logging(debug=True)

# キャッシュ診断
cm = CacheManager(debug=True)
status = cm.diagnose_cache_hierarchy("AAPL")
print(status)

# キャッシュ検証
cm.validate_cache_contents("AAPL", verbose=True)
```

### トレースコンテキストの手動追加

デバッグ用にトレースコンテキストを設定:

```python
from common.trace_context import TraceContext

# トレースコンテキスト作成
with TraceContext.start_trace("debug_session") as ctx:
    # このコンテキスト内のすべてのログはこのトレースIDに関連付けられる
    ctx.add_attribute("system", "system3")
    ctx.add_attribute("debug_mode", True)

    # 問題の処理を実行
    result = problematic_function()

    # トレースID表示
    print(f"デバッグトレースID: {ctx.get_trace_id()}")
```

## 8. テストと検証

### 決定性テスト実行

```bash
# 基本テスト
python -m run_task -w "c:\Repos\quant_trading_system" -i "shell: Quick Test Run"

# 特定テストのみ実行
python -m pytest tests/test_cache_manager.py -v
```

### パフォーマンスベンチマーク

```bash
# ベンチマークモード
python scripts/run_all_systems_today.py --test-mode mini --benchmark

# I/O最適化ベンチマーク
python -m common.io_optimization_benchmark
```

## 9. 問題報告ガイドライン

問題を報告する際は、以下の情報を含めてください:

1. エラーコード (`AAA123E`形式)
2. トレース ID
3. 実行コマンドと使用フラグ
4. エラー発生前のログコンテキスト
5. 再現手順

問題報告テンプレート:

```
## エラー情報
- エラーコード: DAT001E
- トレースID: run_52aa4572
- 実行コマンド: python scripts/run_all_systems_today.py --system system3 --verbose

## エラー内容
[ここにエラーメッセージや現象を記載]

## 再現手順
1. [手順1]
2. [手順2]
3. [手順3]

## 添付情報
[ここに関連するログやスクリーンショットを添付]
```

---

このデバッグガイドが、トレーディングシステムの問題解決に役立つことを願っています。さらに詳細なサポートが必要な場合は、`common/trading_errors.py`や`docs/README.md`を参照してください。
