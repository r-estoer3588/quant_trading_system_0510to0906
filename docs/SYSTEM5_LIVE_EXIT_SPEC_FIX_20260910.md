# System5 live exit を Bensdorp / backtest semantics に戻す (2026-09-10)

## Scope / non-goals

この修正は **System5 の戦略を変更しない**。filter / setup / ranking / entry / risk / stop / target / holding-period の数値は一切変更しない。

正準値は既存の `docs/systems/システム5.txt` / `common/trade_management.py` / `strategies/system5_strategy.py` のまま:

- long
- entry = 前日終値 -3% の指値
- stop = entry - 3 * ATR10
- target trigger = entry + 1 * ATR10
- target に到達したら **翌営業日の寄り付きで成行手仕舞い**
- entry 後 6 営業日を観察し、stop/target のどちらにも到達しなければ **翌営業日（7本目）の寄り付きで成行手仕舞い**
- entry/target/stop に使う ATR10 は entry 時点の参照値

## Confirmed live divergences

### 1. time exit が 1 立会日早い

`count_trading_days(entry, today)` は entry 当日=0、翌立会日=1 の `(entry, today]` 規約。

generic live builder は `holding_days >= max_holding_days` で time exit を出すため、S5 の `max_holding_days=6` は **6本目で成行 close** になる。

一方 `System5Strategy.compute_exit` は `offset=1..6` を stop/target の観察に使い、fallback は `entry_idx + 6 + 1` = **7本目 Open**。

したがって strategy 値を 6→7 に変更するのではなく、S5 live adapter だけが `holding_days > 6` を close 条件に翻訳する。

### 2. +1ATR target が live で別物になっている

generic protection engine は target を broker-resident LIMIT として扱う。通常モードは Alpaca の qty reservation のため `stop > target` 優先で target が発注されず、`PROTECT_USE_OCO=1` では stop+target の OCO になって target 到達時に即約定する。

どちらも S5 の **target touch -> next session open market exit** と一致しない。

この修正では S5 に resting target / OCO を新規生成しない。完了済み日足の High が entry+1ATR に到達したことを検出し、次の立会日に market close を生成する。stop は従来どおり broker-resident GTC を維持する。

### 3. ATR が live で動いていた

旧 live exit path は `paper_exit_check._load_atr_by_symbol()` の **最新行 ATR10** を stop/target に渡していた。

S5 backtest は entry 時に `entry_idx-1` の ATR10 を `_last_entry_atr` として固定する。この差で target/stop 水準が保有中に変わり得た。

S5 adapter は rolling CSV から **entry date より前の最終行 ATR10** を復元して固定する。過去データが無い場合、target は捏造せず未判定にし、downside stop だけ既存の latest-ATR fallback を安全策として許す。

## Existing-position migration

過去に S5 の `protect-target` / `protect-oco` が resting の場合、それは immediate-target semantics なので正準 S5 と一致しない。

S5 adapter は該当 client_order_id だけを `cancel_client_order_ids` に載せ、既存の scoped cancel を使って stop-only に置換する。同一 symbol の手動注文や他 system の注文を blanket cancel しない。

## Regression contract

`tests/test_system5_live_exit_spec_20260910.py` で以下を固定する。

1. strategy constants は 3ATR10 / +1ATR10 / max_holding_days=6 のまま
2. day6 は観察日で time exit しない、day7 で market exit
3. target touch 当日は close せず、翌 session に market exit
4. target/stop は latest ATR ではなく pre-entry ATR を使う
5. legacy S5 OCO/target は stop-only へ限定移行
6. entry history 不足時に latest ATR から profit target を捏造しない

## Separate P0 found during audit

この作業中、System5 だけに限らない別の execution gap も確認した。

`open_auto_run` は `exit_stage -> entry_stage` の順であり、entry fill 後に protection を再-arm する stage がない。したがって、その run で新規約定した whole-share position は同じ run の `paper_exit_check` が見ておらず、次の protection pass まで native stop/trailing が無い可能性がある。

これは本 PR の「S5 exit semantics」と分離して扱う。strategy を変えず、entry 後の protection timing / late DAY-limit fill を安全に arm する実行レイヤ修正として別途 remediation する。
