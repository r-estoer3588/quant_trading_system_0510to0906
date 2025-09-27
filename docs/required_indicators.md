# 必要な指標一覧# 忁 E�� 指標リスチ E

## 前計算すべき指標（indicators_common.py）本シスチ E���E�Edocs`フォルダおよび`app_system1.py`〜`app_system7.py`�E� で参 �E される指標をまとめた、E

1. **SMA25**: 25 日単純移動平均 1. **SMA25**�E�E5 日単純移動平坁 E

2. **SMA50**: 50 日単純移動平均 2. **SMA50**�E�E0 日単純移動平坁 E

3. **SMA100**: 100 日単純移動平均 3. **SMA100**�E�E00 日単純移動平坁 E

4. **SMA150**: 150 日単純移動平均 4. **SMA150**�E�E50 日単純移動平坁 E

5. **SMA200**: 200 日単純移動平均 5. **SMA200**�E�E00 日単純移動平坁 E

6. **ATR3**: 3 倍 ATR。過去 10 日・40 日・50 日など各期間で使用 6. **ATR3**�E�EATR。過去 10 日・40 日・50 日など褁 E�� 期間で使用

7. **ATR1.5**: 1.5 倍 ATR。過去 40 日などで使用 7. **ATR1.5**�E�E.5ATR。過去 40 日などで使用

8. **ATR1**: 1 倍 ATR。過去 10 日などで使用 8. **ATR1**�E�EATR。過去 10 日などで使用

9. **ATR2.5**: 2.5 倍 ATR。過去 10 日などで使用 9. **ATR2.5**�E�E.5ATR。過去 10 日などで使用

10. **ATR**: 過去 10 日、過去 50 日、過去 40 日、3%ATR など各期間で使用 10. **ATR**�E� 過去 10 日、E�� 去 50 日、E�� 去 40 日、E%ATR など褁 E�� 期間で使用

11. **ADX7**: 7 日 ADX11. **ADX7**�E�E 日 ADX

12. **return_6d**: （旧称 RETURN6D） 6 日間リターン 12. \**return_6d�E� 旧称 RETURN6�E�E*�E�E 日リターン

13. **return_pct**: 総リターン 13. **return_pct**�E� 総リターン

14. **Drop3D**: 3 日ドロップ 14. **Drop3D**�E�E 日ドロチ E�E

15. **HV50**: 50 日ヒストリカルボラティリティ（年率換算）15. **HV50**�E�E0 日ヒストリカルボラチ E�� リチ E���E� 年玁 E�� 算！E

## 指標と使用システム対応表## 持 E�� と使用シスチ E�� 対応表

| 指標 | 使用システム | 実装状況 || 持 E��E | 使用シスチ E�� | 実裁 E�� 況 E |

| ------------------------------------- | ------------------------------------------------------------- | --------------------------------------------- || ------------------------------------- | ------------------------------------------------------------- | --------------------------------------------- |

| SMA25 | System1 | 列として実装済み(`SMA25`) || SMA25 | System1 | 列として実裁 E��E(`SMA25`) |

| SMA50 | System1 | 列として実装済み(`SMA50`) || SMA50 | System1 | 列として実裁 E��E(`SMA50`) |

| SMA100 | System1, System5 | 列として実装済み(`SMA100`) || SMA100 | System1, System5 | 列として実裁 E��E(`SMA100`) |

| SMA150 | System3 | 列として実装済み(`SMA150`) || SMA150 | System3 | 列として実裁 E��E(`SMA150`) |

| SMA200 | System4 | 列として実装済み(`SMA200`) || SMA200 | System4 | 列として実裁 E��E(`SMA200`) |

| ATR3 | System2, System5, System6, System7 | ストップ計算で `stop_atr_multiple=3` を使用 || ATR3 | System2, System5, System6, System7 | ストップ計算で `stop_atr_multiple=3` を使用 |

| ATR1.5 | System4 | ストップ計算で `stop_atr_multiple=1.5` を使用 || ATR1.5 | System4 | ストップ計算で `stop_atr_multiple=1.5` を使用 |

| ATR1 | System5 | `Close > SMA100 + ATR10` 判定で使用 || ATR1 | System5 | `Close > SMA100 + ATR10` 判定で使用 |

| ATR2.5 | System3 | ストップ計算で `stop_atr_multiple=2.5` を使用 || ATR2.5 | System3 | ストップ計算で `stop_atr_multiple=2.5` を使用 |

| ATR（10 日・20 日・40 日・50 日など） | System1, System2, System3, System4, System5, System6, System7 | 列として実装済み(`ATR10` 等) || ATR�E�E0 日・20 日・40 日・50 日など �E�E| System1, System2, System3, System4, System5, System6, System7 | 列として実裁 E��E(`ATR10` 筁 E |

| ADX7 | System2, System5 | 列として実装済み(`ADX7`) || ADX7 | System2, System5 | 列として実裁 E��E(`ADX7`) |

| return_6d（旧称 RETURN6D） | System6 | 列として実装済み(`return_6d`) || return_6d�E� 旧称 RETURN6�E�E | System6 | 列として実裁 E��E(`return_6d`) |

| return_pct | System1, System2, System3, System4, System5, System6, System7 | 列として実装済み(`return_pct`) || return_pct | System1, System2, System3, System4, System5, System6, System7 | 列として実裁 E��E(`return_pct`) |

| Drop3D | System3 | 列として実装済み(`Drop3D`) || Drop3D | System3 | 列として実裁 E��E(`Drop3D`) |

| HV50 | System4 | 列として実装済み(`HV50`) || HV50 | System4 | 列として実裁 E��E(`HV50`) |

## 補足## 補足

- ATR は「過去 10 日」「過去 40 日」「過去 50 日」、「3ATR」、「1.5ATR」、「2.5ATR」、「1ATR」、「3%ATR」など複数パターンが存在する。- ATR は「過去 10 日」「過去 40 日」「過去 50 日」、EATR」、E.5ATR」、E.5ATR」、EATR」、E%ATR」など褁 E�� パターンが存在する、E

- SMA は、「25 日」、「50 日」、「100 日」、「150 日」、「200 日」など複数の期間を参照する。- SMA は、E5 日」、E0 日」、E00 日」、E50 日」、E00 日」など褁 E�� の期間を参照する、E

- ADX は 7 日値を使用し、必要に応じて高い値ランキング（`ADX7_High`）や 55 以上の閾値判定を行う。- ADX は 7 日値を使用し、忁 E�� に応じて高い頁 E�� ンキング �E�EADX7_High`�E� や 55 以上 �E 閾値判定を行う、E

- Return 系指標は「return_6d（旧称 RETURN6D）」「return_pct」などを含む。- Return 系持 E���E「Return6D�E� 旧称 RETURN6�E�」「return_pct」などを含む、E

- Drop3D は、「3 日ドロップ」として使用される。- Drop3D は、E 日ドロチ E�E」として使用される、E

## システム別フィルター## シスチ E�� 別フィルター

### System1: CANSLIM Growth ロング戦略### System1

- フィルター: `Close > SMA25 > SMA50 > SMA100`

- 必要指標: `SMA25`, `SMA50`, `SMA100`, `ATR10`, `return_pct`- `avg_dollar_volume_20 > 50_000_000`

- `low >= 5`

### System2: Small Cap Growth ショート戦略

- フィルター: `ADX7 >= 55`### System2

- 必要指標: `ADX7`, `ATR10`, `return_pct`

- `low >= 5`

### System3: Mean Reversion ロング戦略- `avg_dollar_volume_20 > 25_000_000`

- フィルター: `Close < SMA150`, `Drop3D <= -0.30`- `ATR10 / close > 0.03`

- 必要指標: `SMA150`, `Drop3D`, `ATR10`, `return_pct`

### System3

### System4: Volatility Contraction ロング戦略

- フィルター: `Close > SMA200`, `HV50 <= 30`- `low >= 1`

- 必要指標: `SMA200`, `HV50`, `ATR40`, `return_pct`- `avg_volume_50 >= 1_000_000`

- `ATR10 / close >= 0.05`

### System5: Breakout ロング戦略

- フィルター: `Close > SMA100 + ATR10`, `ADX7 >= 30`### System4

- 必要指標: `SMA100`, `ATR10`, `ADX7`, `return_pct`

- `avg_dollar_volume_50 > 100_000_000`

### System6: Momentum Burst ショート戦略- `0.10 <= HV50 <= 0.40`

- フィルター: `return_6d > 0.20`, `uptwodays == True`

- 必要指標: `return_6d`, `uptwodays`, `ATR10`, `dollarvolume50`### System5

### System7: SPY アンカー ショート戦略- `avg_volume_50 > 500_000`

- 対象: SPY 固定- `avg_dollar_volume_50 > 2_500_000`

- フィルター: なし- `ATR10 / close > 0.04`

- 必要指標: `ATR10`, `return_pct`

### System6

## キャッシュファイル構造

- `low >= 5`

### Base Cache (`data_cache/base/`)- `avg_dollar_volume_50 > 10_000_000`

- 長期データ（バックテスト用）

- 全指標を含む完全データセット### System7

- 日次更新

- フィルターなぁ E

### Rolling Cache (`data_cache/rolling/`)

- 直近 300 日データ（当日シグナル用）## シスチ E�� 別セチ E�� アチ E�E

- 必要指標のみを含む軽量データセット

- リアルタイム更新### System1

### Full Backup Cache (`data_cache/full_backup/`)- `SPY_close > SPY_SMA100`

- 原本バックアップ（復旧用）- `SMA25 > SMA50`

- 指標計算前の生データ

- 緊急時復旧用### System2

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
