# 蠢・ｦ∵欠讓吶Μ繧ｹ繝・

譛ｬ繧ｷ繧ｹ繝・Β・・docs` 繝輔か繝ｫ繝縺翫ｈ縺ｳ `app_system1.py`縲彖app_system7.py`・峨〒蜿ら・縺輔ｌ繧区欠讓吶ｒ縺ｾ縺ｨ繧√◆縲・

1. **SMA25**・・5 譌･蜊倡ｴ皮ｧｻ蜍募ｹｳ蝮・
2. **SMA50**・・0 譌･蜊倡ｴ皮ｧｻ蜍募ｹｳ蝮・
3. **SMA100**・・00 譌･蜊倡ｴ皮ｧｻ蜍募ｹｳ蝮・
4. **SMA150**・・50 譌･蜊倡ｴ皮ｧｻ蜍募ｹｳ蝮・
5. **SMA200**・・00 譌･蜊倡ｴ皮ｧｻ蜍募ｹｳ蝮・
6. **ATR3**・・ATR縲る℃蜴ｻ 10 譌･繝ｻ40 譌･繝ｻ50 譌･縺ｪ縺ｩ隍・焚譛滄俣縺ｧ菴ｿ逕ｨ
7. **ATR1.5**・・.5ATR縲る℃蜴ｻ 40 譌･縺ｪ縺ｩ縺ｧ菴ｿ逕ｨ
8. **ATR1**・・ATR縲る℃蜴ｻ 10 譌･縺ｪ縺ｩ縺ｧ菴ｿ逕ｨ
9. **ATR2.5**・・.5ATR縲る℃蜴ｻ 10 譌･縺ｪ縺ｩ縺ｧ菴ｿ逕ｨ
10. **ATR**・夐℃蜴ｻ 10 譌･縲・℃蜴ｻ 50 譌･縲・℃蜴ｻ 40 譌･縲・%ATR 縺ｪ縺ｩ隍・焚譛滄俣縺ｧ菴ｿ逕ｨ
11. **ADX7**・・ 譌･ ADX
12. **return_6d・域立遘ｰ RETURN6・・*・・ 譌･繝ｪ繧ｿ繝ｼ繝ｳ
13. **return_pct**・夂ｷ上Μ繧ｿ繝ｼ繝ｳ
14. **Drop3D**・・ 譌･繝峨Ο繝・・
15. **HV50**・・0 譌･繝偵せ繝医Μ繧ｫ繝ｫ繝懊Λ繝・ぅ繝ｪ繝・ぅ・亥ｹｴ邇・鋤邂暦ｼ・

## 謖・ｨ吶→菴ｿ逕ｨ繧ｷ繧ｹ繝・Β蟇ｾ蠢懆｡ｨ

| 謖・ｨ・                                 | 菴ｿ逕ｨ繧ｷ繧ｹ繝・Β                                                  | 螳溯｣・憾豕・                                     |
| ------------------------------------- | ------------------------------------------------------------- | --------------------------------------------- |
| SMA25                                 | System1                                                       | 蛻励→縺励※螳溯｣・ｸ・(`SMA25`)                      |
| SMA50                                 | System1                                                       | 蛻励→縺励※螳溯｣・ｸ・(`SMA50`)                      |
| SMA100                                | System1, System5                                              | 蛻励→縺励※螳溯｣・ｸ・(`SMA100`)                     |
| SMA150                                | System3                                                       | 蛻励→縺励※螳溯｣・ｸ・(`SMA150`)                     |
| SMA200                                | System4                                                       | 蛻励→縺励※螳溯｣・ｸ・(`SMA200`)                     |
| ATR3                                  | System2, System5, System6, System7                            | 繧ｹ繝医ャ繝苓ｨ育ｮ励〒 `stop_atr_multiple=3` 繧剃ｽｿ逕ｨ   |
| ATR1.5                                | System4                                                       | 繧ｹ繝医ャ繝苓ｨ育ｮ励〒 `stop_atr_multiple=1.5` 繧剃ｽｿ逕ｨ |
| ATR1                                  | System5                                                       | `Close > SMA100 + ATR10` 蛻､螳壹〒菴ｿ逕ｨ           |
| ATR2.5                                | System3                                                       | 繧ｹ繝医ャ繝苓ｨ育ｮ励〒 `stop_atr_multiple=2.5` 繧剃ｽｿ逕ｨ |
| ATR・・0 譌･繝ｻ20 譌･繝ｻ40 譌･繝ｻ50 譌･縺ｪ縺ｩ・・| System1, System2, System3, System4, System5, System6, System7 | 蛻励→縺励※螳溯｣・ｸ・(`ATR10` 遲・                   |
| ADX7                                  | System2, System5                                              | 蛻励→縺励※螳溯｣・ｸ・(`ADX7`)                       |
| return_6d・域立遘ｰ RETURN6・・             | System6                                                       | 蛻励→縺励※螳溯｣・ｸ・(`return_6d`)                   |
| return_pct                            | System1, System2, System3, System4, System5, System6, System7 | 蛻励→縺励※螳溯｣・ｸ・(`return_pct`)                 |
| Drop3D                                | System3                                                       | 蛻励→縺励※螳溯｣・ｸ・(`Drop3D`)                     |
| HV50                                  | System4                                                       | 蛻励→縺励※螳溯｣・ｸ・(`HV50`)                       |

## 陬懆ｶｳ

- ATR 縺ｯ縲碁℃蜴ｻ 10 譌･縲阪碁℃蜴ｻ 40 譌･縲阪碁℃蜴ｻ 50 譌･縲阪・ATR縲阪・.5ATR縲阪・.5ATR縲阪・ATR縲阪・%ATR縲阪↑縺ｩ隍・焚繝代ち繝ｼ繝ｳ縺悟ｭ伜惠縺吶ｋ縲・
- SMA 縺ｯ縲・5 譌･縲阪・0 譌･縲阪・00 譌･縲阪・50 譌･縲阪・00 譌･縲阪↑縺ｩ隍・焚縺ｮ譛滄俣繧貞盾辣ｧ縺吶ｋ縲・
- ADX 縺ｯ 7 譌･蛟､繧剃ｽｿ逕ｨ縺励∝ｿ・ｦ√↓蠢懊§縺ｦ鬮倥＞鬆・Λ繝ｳ繧ｭ繝ｳ繧ｰ・・ADX7_High`・峨ｄ 55 莉･荳翫・髢ｾ蛟､蛻､螳壹ｒ陦後≧縲・
- Return 邉ｻ謖・ｨ吶・縲軍eturn6D・域立遘ｰ RETURN6・峨阪罫eturn_pct縲阪↑縺ｩ繧貞性繧縲・
- Drop3D 縺ｯ縲・ 譌･繝峨Ο繝・・縲阪→縺励※菴ｿ逕ｨ縺輔ｌ繧九・

## 繧ｷ繧ｹ繝・Β蛻･繝輔ぅ繝ｫ繧ｿ繝ｼ

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

- 繝輔ぅ繝ｫ繧ｿ繝ｼ縺ｪ縺・

## 繧ｷ繧ｹ繝・Β蛻･繧ｻ繝・ヨ繧｢繝・・

### System1

- `SPY_close > SPY_SMA100`
- `SMA25 > SMA50`

### System2

- `RSI3 > 90`
- `close[-1] > close[-2]` 縺九▽ `close[-2] > close[-3]`

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
- `close[-1] > close[-2]` 縺九▽ `close[-2] > close[-3]`

### System7

- `SPY_low == rolling_min(SPY_low, window=50)`
