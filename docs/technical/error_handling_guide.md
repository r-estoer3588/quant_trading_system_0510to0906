# 統一エラーハンドリングフレームワーク

このドキュメントでは、`common/error_handling.py` の使用方法を説明します。

## 目的

従来、システム全体で以下の 3 パターンのエラー出力が混在していました：

1. `logger.error("エラーメッセージ")`
2. `log_callback(f"[ERROR] エラーメッセージ")` （Streamlit UI 通知用）
3. `print(f"Error: {e}")` （デバッグ用の残骸）

これを統一し、以下を実現します：

- **統一されたログ出力**: logger と log_callback の両方に自動出力
- **コンテキスト情報の追加**: 銘柄名・システム番号等を構造化
- **カスタム例外階層**: エラー種別ごとの明確な例外クラス
- **環境変数対応**: `COMPACT_TODAY_LOGS` を考慮した自動切り替え

---

## カスタム例外階層

すべての例外は `QuantTradingError` を基底とします。

```python
from common.error_handling import (
    QuantTradingError,    # 基底例外
    DataError,            # データロード・キャッシュエラー
    CalculationError,     # 指標計算・シグナル生成エラー
    SystemError,          # システム設定・環境変数エラー
    AllocationError,      # 配分計算エラー
    ValidationError,      # データバリデーションエラー
)
```

### 例外の使用例

```python
# DataError: キャッシュファイルが見つからない
raise DataError("キャッシュファイルが見つかりません", symbol="AAPL", cache_type="rolling")

# CalculationError: 指標計算中の数値エラー
raise CalculationError("ROC200の計算に失敗しました", symbol="TSLA", reason="insufficient data")

# ValidationError: Setup/Filter列の不整合
raise ValidationError(
    "Setup列とFilter列が不整合です",
    symbol="NVDA",
    setup_count=10,
    filter_count=12
)
```

すべての例外は **コンテキスト情報** (`**kwargs`) を受け取り、自動的にメッセージに付与されます：

```
DataError: キャッシュファイルが見つかりません (symbol=AAPL, cache_type=rolling)
```

---

## SystemErrorHandler の使用方法

### 基本的な使い方

```python
from common.error_handling import SystemErrorHandler
import logging

logger = logging.getLogger(__name__)

# エラーハンドラーの作成
handler = SystemErrorHandler(
    system_name="System3",
    logger=logger,
    log_callback=None,  # Streamlit UI用コールバック（オプション）
)

# ログ出力（logger + log_callback の両方に出力）
handler.info("処理開始", symbol_count=100)
handler.warning("キャッシュミス", symbol="AAPL")
handler.error("データロード失敗", symbol="TSLA", reason="file not found")
```

### SystemLogger との統合（推奨）

`common/logging_utils.py::SystemLogger` は `SystemErrorHandler` と同等の機能を提供します：

```python
from common.logging_utils import SystemLogger

# SystemLogger を使用（推奨）
sys_logger = SystemLogger.create(
    system_name="System3",
    logger=logger,
    log_callback=log_callback,
)

# 使い方は SystemErrorHandler と同じ
sys_logger.info("処理開始", symbol_count=100)
sys_logger.error("エラー発生", symbol="AAPL")
```

**どちらを使うべきか？**

- **SystemLogger**: ロギングモジュール (`common/logging_utils.py`) で既に使用されている場合
- **SystemErrorHandler**: エラーハンドリングモジュール (`common/error_handling.py`) を使用する場合

両方とも同じ機能を提供し、`COMPACT_TODAY_LOGS` 環境変数を自動考慮します。

### log_callback との統合

Streamlit UI への通知を含める場合：

```python
def my_log_callback(msg: str):
    """Streamlit UI への通知"""
    st.write(msg)

handler = SystemErrorHandler(
    system_name="System3",
    logger=logger,
    log_callback=my_log_callback,  # UIに通知
)

handler.info("処理開始", symbol_count=100)
# 出力先:
# - logger: "System3: 処理開始 (symbol_count=100)"
# - log_callback: "[INFO] System3: 処理開始 (symbol_count=100)"
```

### 環境変数対応（推奨）

`COMPACT_TODAY_LOGS` 環境変数を自動考慮する場合：

```python
from common.error_handling import create_handler_from_env

# 環境変数を考慮して自動生成
handler = create_handler_from_env(
    system_name="System3",
    logger=logger,
    log_callback=log_callback,
)

# COMPACT_TODAY_LOGS=1 の場合、info() は自動的に DEBUG レベルに降格
handler.info("詳細情報", symbol="AAPL")  # DEBUG レベルで出力
handler.error("エラー", symbol="TSLA")   # 常に ERROR レベルで出力
```

---

## 実践的な使用例

### System3 での使用例

```python
# core/system3.py

from common.error_handling import (
    SystemErrorHandler,
    create_handler_from_env,
    DataError,
    CalculationError,
)
import logging

logger = logging.getLogger(__name__)


def prepare_data_vectorized_system3(
    symbols: list[str],
    log_callback: Callable[[str], None] | None = None,
    latest_only: bool = False,
) -> dict[str, pd.DataFrame]:
    """System3のデータ準備"""

    # エラーハンドラー作成（環境変数対応）
    handler = create_handler_from_env(
        system_name="System3",
        logger=logger,
        log_callback=log_callback,
    )

    handler.info("データ準備開始", symbol_count=len(symbols), latest_only=latest_only)

    results = {}
    for symbol in symbols:
        try:
            # データロード
            df = load_data(symbol)
            if df is None:
                raise DataError("データロードに失敗しました", symbol=symbol)

            # 指標計算
            df = calculate_indicators(df)
            results[symbol] = df

            handler.debug("データ準備完了", symbol=symbol, rows=len(df))

        except DataError as e:
            # DataError は警告レベル（スキップして継続）
            handler.warning(str(e))
            continue

        except CalculationError as e:
            # CalculationError はエラーレベル（スキップして継続）
            handler.error(str(e))
            continue

        except Exception as e:
            # 予期しない例外は例外情報付きでログ
            handler.exception(
                "予期しないエラーが発生しました",
                exc_info=e,
                symbol=symbol,
            )
            continue

    handler.info("データ準備完了", success_count=len(results))
    return results
```

### ヘルパー関数の使用

```python
from common.error_handling import (
    handle_data_error,
    handle_calculation_error,
)

# データエラーの統一ハンドリング
try:
    df = load_rolling_cache(symbol)
except Exception as e:
    handle_data_error(
        handler=handler,
        operation="キャッシュ読み込み",
        symbol=symbol,
        exc=e,
    )
    # 出力: "System3: キャッシュ読み込みに失敗しました (symbol=AAPL)"
    # + 例外のスタックトレース

# 計算エラーの統一ハンドリング
try:
    df["ROC200"] = calculate_roc(df, 200)
except Exception as e:
    handle_calculation_error(
        handler=handler,
        operation="指標計算",
        symbol=symbol,
        indicator="ROC200",
        exc=e,
    )
    # 出力: "System3: 指標計算に失敗しました (symbol=AAPL, indicator=ROC200)"
```

---

## 既存コードからの移行

### Before: logger.error のみ

```python
# 旧コード
logger.error(f"Error processing {symbol}")
```

### After: SystemErrorHandler

```python
# 新コード
handler.error("データ処理に失敗しました", symbol=symbol)
```

---

### Before: log_callback のみ

```python
# 旧コード
if log_callback:
    log_callback(f"[ERROR] System3: Failed to load {symbol}")
```

### After: SystemErrorHandler

```python
# 新コード（logger + log_callback の両方に自動出力）
handler.error("データロードに失敗しました", symbol=symbol)
```

---

### Before: print デバッグ

```python
# 旧コード
print(f"Error: {e}")
```

### After: SystemErrorHandler

```python
# 新コード
handler.exception("予期しないエラー", exc_info=e)
```

---

## ログレベルの使い分け

| レベル        | 用途               | 例                             |
| ------------- | ------------------ | ------------------------------ |
| `debug()`     | デバッグ情報       | 各銘柄の処理状況               |
| `info()`      | 通常の進捗情報     | 処理開始・完了、成功件数       |
| `warning()`   | 警告（処理継続）   | キャッシュミス、一部データ欠落 |
| `error()`     | エラー（処理継続） | データロード失敗、計算エラー   |
| `critical()`  | 致命的エラー       | システム全体の停止が必要       |
| `exception()` | 例外付きエラー     | try-except で捕捉した例外      |

---

## 環境変数 `COMPACT_TODAY_LOGS` の影響

| compact_mode                    | `info()` の出力先 | `error()` の出力先 |
| ------------------------------- | ----------------- | ------------------ |
| `False` (デフォルト)            | INFO レベル       | ERROR レベル       |
| `True` (`COMPACT_TODAY_LOGS=1`) | **DEBUG レベル**  | ERROR レベル       |

**使い分け**:

- 本番環境: `COMPACT_TODAY_LOGS=1` でログ量を削減
- 開発環境: `COMPACT_TODAY_LOGS=0` で詳細情報を出力

---

## テストでの使用

```python
import pytest
from common.error_handling import SystemErrorHandler, DataError

def test_error_handler():
    """エラーハンドラーのテスト"""
    messages = []

    def mock_callback(msg: str):
        messages.append(msg)

    handler = SystemErrorHandler(
        system_name="TestSystem",
        log_callback=mock_callback,
    )

    handler.error("テストエラー", symbol="TEST")

    # log_callback が呼ばれたことを確認
    assert len(messages) == 1
    assert "[ERROR]" in messages[0]
    assert "TestSystem" in messages[0]
    assert "symbol=TEST" in messages[0]
```

---

## まとめ

### ✅ 推奨パターン

1. **エラーハンドラーの作成**: `create_handler_from_env()` を使用
2. **ログ出力**: `handler.info()`, `handler.error()` 等を使用
3. **例外発生**: `DataError`, `CalculationError` 等のカスタム例外を使用
4. **コンテキスト情報**: `**kwargs` で銘柄名・システム番号等を渡す

### ❌ 非推奨パターン

1. ~~`logger.error()` の直接使用~~ → `handler.error()` を使用
2. ~~`log_callback()` の直接使用~~ → `handler.error()` が自動対応
3. ~~`print()` でのデバッグ~~ → `handler.debug()` を使用
4. ~~汎用の `Exception`~~ → カスタム例外を使用

---

## 関連ドキュメント

- [環境変数一覧](../technical/environment_variables.md) - `COMPACT_TODAY_LOGS` の詳細
- [実装状況マトリクス](../today_signal_scan/implementation_status.md) - エラーハンドリング不統一の問題
