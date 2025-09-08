# 必要指標リスト

本システム（`docs` フォルダおよび `app_system1.py`〜`app_system7.py`）で参照される指標をまとめた。

1. **SMA25**：25日単純移動平均
2. **SMA50**：50日単純移動平均
3. **SMA100**：100日単純移動平均
4. **SMA150**：150日単純移動平均
5. **SMA200**：200日単純移動平均
6. **ATR3**：3ATR。過去10日・40日・50日など複数期間で使用
7. **ATR1.5**：1.5ATR。過去40日などで使用
8. **ATR1**：1ATR。過去10日などで使用
9. **ATR2.5**：2.5ATR。過去10日などで使用
10. **ATR**：過去10日、過去50日、過去40日、4%ATR など複数期間で使用
11. **ADX7**：7日ADX
12. **Return6D（旧称 RETURN6）**：6日リターン
13. **return_pct**：総リターン
14. **Drop3D**：3日ドロップ
15. **HV50**：50日ヒストリカルボラティリティ（年率換算）

## 指標と使用システム対応表

| 指標 | 使用システム | 実装状況 |
| --- | --- | --- |
| SMA25 | System1 | 列として実装済 (`SMA25`) |
| SMA50 | System1 | 列として実装済 (`SMA50`) |
| SMA100 | System1, System5 | 列として実装済 (`SMA100`) |
| SMA150 | System3 | 列として実装済 (`SMA150`) |
| SMA200 | System4 | 列として実装済 (`SMA200`) |
| ATR3 | System2, System5, System6, System7 | 未実装（`ATR`列の3倍で計算） |
| ATR1.5 | System4 | 未実装（`ATR`列の1.5倍で計算） |
| ATR1 | System5 | 未実装（`ATR`列を使用） |
| ATR2.5 | System3 | 未実装（`ATR`列の2.5倍で計算） |
| ATR（10日・20日・40日・50日など） | System1, System2, System3, System4, System5, System6, System7 | 列として実装済 (`ATR10` 等) |
| ADX7 | System2, System5 | 列として実装済 (`ADX7`) |
| Return6D（旧称 RETURN6） | System6 | 列として実装済 (`Return6D`) |
| return_pct | System1, System2, System3, System4, System5, System6, System7 | 列として実装済 (`return_pct`) |
| Drop3D | System3 | 列として実装済 (`Drop3D`) |
| HV50 | System4 | 列として実装済 (`HV50`) |

## 補足

- ATR は「過去10日」「過去40日」「過去50日」「3ATR」「1.5ATR」「2.5ATR」「1ATR」「4%ATR」など複数パターンが存在する。
- SMA は「25日」「50日」「100日」「150日」「200日」など複数の期間を参照する。
- ADX は 7 日値を使用し、必要に応じて高い順ランキング（`ADX7_High`）や 55 以上の閾値判定を行う。
- Return 系指標は「Return6D（旧称 RETURN6）」「return_pct」などを含む。
- Drop3D は「3日ドロップ」として使用される。

## システム別フィルター

### System1
- `avg_dollar_volume_20 > 50_000_000`
- `low >= 5`

### System2
- `low >= 5`
- `avg_dollar_volume_20 > 25_000_000`
- `ATR10 / close > 0.03`

### System3
- `low >= 1`
- `avg_volume_50 >= 1_000_000`
- `ATR10 / close >= 0.05`

### System4
- `avg_dollar_volume_50 > 100_000_000`
- `0.10 <= HV50 <= 0.40`

### System5
- `avg_volume_50 > 500_000`
- `avg_dollar_volume_50 > 2_500_000`
- `ATR10 / close > 0.04`

### System6
- `low >= 5`
- `avg_dollar_volume_50 > 10_000_000`

### System7
- フィルターなし

## システム別セットアップ

### System1
- `SPY_close > SPY_SMA100`
- `SMA25 > SMA50`

### System2
- `RSI3 > 90`
- `close[-1] > close[-2]` かつ `close[-2] > close[-3]`

### System3
- `close > SMA150`
- `(close[-3] - close) / close[-3] <= -0.125`

### System4
- `SPX_close > SPX_SMA200`
- `close > SMA200`

### System5
- `close > SMA100 + ATR10`
- `ADX7 > 55`
- `RSI3 < 50`

### System6
- `(close / close[-6]) - 1 >= 0.20`
- `close[-1] > close[-2]` かつ `close[-2] > close[-3]`

### System7
- `SPY_low == rolling_min(SPY_low, window=50)`
