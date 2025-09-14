from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import inspect
import time as _t
from typing import Any

import pandas as pd

# --- サイド定義（売買区分）---
LONG_SYSTEMS = {"system1", "system3", "system4", "system5"}
SHORT_SYSTEMS = {"system2", "system6", "system7"}


@dataclass(frozen=True)
class TodaySignal:
    symbol: str
    system: str
    side: str  # "long" | "short"
    signal_type: str  # "buy" | "sell"
    entry_date: pd.Timestamp
    entry_price: float
    stop_price: float
    score_key: str | None = None
    score: float | None = None
    reason: str | None = None


def _infer_side(system_name: str) -> str:
    name = (system_name or "").lower()
    if name in SHORT_SYSTEMS:
        return "short"
    return "long"


def _score_from_candidate(
    system_name: str, candidate: dict
) -> tuple[str | None, float | None, bool]:
    """
    候補レコードからスコア項目と並び順（昇順か）を推定して返す。
    戻り値: (score_key, score_value, asc)
    """
    name = (system_name or "").lower()
    # システム別の代表スコア
    key_order: list[tuple[list[str], bool]] = [
        (["ROC200"], False),  # s1: 大きいほど良い
        (["ADX7"], False),  # s2,s5: 大きいほど良い
        (["Drop3D"], False),  # s3: 大きいほど良い（下落率）
        (["RSI4"], True),  # s4: 小さいほど良い
        (["Return6D"], False),  # s6: 大きいほど良い
        (["ATR50"], False),  # s7: 参考
    ]
    # system 固有優先順位
    if name == "system4":
        key_order = [(["RSI4"], True), (["ATR40"], True)] + key_order
    elif name == "system2":
        key_order = [(["ADX7"], False), (["RSI3"], False)] + key_order
    elif name == "system5":
        key_order = [(["ADX7"], False), (["ATR10"], True)] + key_order
    elif name == "system6":
        key_order = [(["Return6D"], False), (["ATR10"], True)] + key_order

    for keys, asc in key_order:
        for k in keys:
            if k in candidate:
                v = candidate.get(k)
                if v is None:
                    return k, None, asc
                if isinstance(v, (int, float, str)):
                    try:
                        return k, float(v), asc
                    except Exception:
                        return k, None, asc
                else:
                    return k, None, asc
    # 見つからない場合
    return None, None, False


def _label_for_score_key(key: str | None) -> str:
    """スコアキーの日本語ラベルを返す（既知のもののみ簡潔表示）。"""
    if key is None:
        return "スコア"
    k = str(key).upper()
    mapping = {
        "ROC200": "ROC200",
        "ADX7": "ADX",
        "RSI4": "RSI4",
        "RSI3": "RSI3",
        "DROP3D": "3日下落率",
        "RETURN6D": "過去6日騰落率",
        "ATR10": "ATR10",
        "ATR20": "ATR20",
        "ATR40": "ATR40",
        "ATR50": "ATR50",
    }
    return mapping.get(k, k)


def _pick_atr_col(df: pd.DataFrame) -> str | None:
    for col in ("ATR20", "ATR10", "ATR40", "ATR50", "ATR14"):
        if col in df.columns:
            return col
    return None


def _compute_entry_stop(
    strategy, df: pd.DataFrame, candidate: dict, side: str
) -> tuple[float, float] | None:
    # strategy 独自の compute_entry があれば優先
    try:
        _fn = strategy.compute_entry  # type: ignore[attr-defined]
    except Exception:
        _fn = None
    if callable(_fn):
        try:
            res = _fn(df, candidate, 0.0)
            if res and isinstance(res, tuple) and len(res) == 2:
                entry, stop = float(res[0]), float(res[1])
                if entry > 0 and (
                    (side == "short" and stop > entry)
                    or (side == "long" and entry > stop)
                ):
                    return round(entry, 4), round(stop, 4)
        except Exception:
            pass

    # フォールバック: 当日始値 ± 3*ATR
    try:
        entry_ts = pd.Timestamp(candidate["entry_date"])
    except Exception:
        return None
    try:
        idxer = df.index.get_indexer([entry_ts])
        entry_idx = int(idxer[0]) if len(idxer) else -1
    except Exception:
        return None
    if entry_idx <= 0 or entry_idx >= len(df):
        return None
    try:
        entry = float(df.iloc[entry_idx]["Open"])
    except Exception:
        return None
    atr_col = _pick_atr_col(df)
    if not atr_col:
        return None
    try:
        atr = float(df.iloc[entry_idx - 1][atr_col])
    except Exception:
        return None
    mult = 3.0
    stop = entry - mult * atr if side == "long" else entry + mult * atr
    return round(entry, 4), round(stop, 4)


def get_today_signals_for_strategy(
    strategy,
    raw_data_dict: dict[str, pd.DataFrame],
    *,
    market_df: pd.DataFrame | None = None,
    today: pd.Timestamp | None = None,
    progress_callback: Callable[..., None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    stage_progress: (
        Callable[[int, int | None, int | None, int | None, int | None], None] | None
    ) = None,
    use_process_pool: bool = False,
    max_workers: int | None = None,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """
    各 Strategy の prepare_data / generate_candidates を流用し、
    最新営業日の候補のみを DataFrame で返す。

    戻り値カラム:
        - symbol, system, side, signal_type,
          entry_date, entry_price, stop_price,
          score_key, score
    """
    from common.utils_spy import get_latest_nyse_trading_day

    try:
        system_name = str(strategy.SYSTEM_NAME).lower()  # type: ignore[attr-defined]
    except Exception:
        system_name = ""
    side = _infer_side(system_name)
    signal_type = "sell" if side == "short" else "buy"

    # 取引日
    if today is None:
        today = get_latest_nyse_trading_day()
    if isinstance(today, pd.Timestamp):
        today = today.normalize()

    # 準備
    total_symbols = len(raw_data_dict)
    if log_callback:
        try:
            log_callback(f"🧪 フィルターチェック開始：{total_symbols} 銘柄")
        except Exception:
            pass
    # 0% -> 25%
    try:
        if stage_progress:
            # 0% ステージでは対象銘柄数を第1引数に渡す（UI 側で "対象→n" 表示に使用）
            stage_progress(0, total_symbols, None, None, None)
    except Exception:
        pass
    t0 = _t.time()
    # ルックバック最適化：必要日数が指定されていれば各DFを末尾N行にスライス
    sliced_dict = raw_data_dict
    try:
        if (
            lookback_days is not None
            and lookback_days > 0
            and isinstance(raw_data_dict, dict)
        ):
            sliced: dict[str, pd.DataFrame] = {}
            for _sym, _df in raw_data_dict.items():
                try:
                    if _df is None or getattr(_df, "empty", True):
                        continue
                    x = _df.copy()
                    if "Date" in x.columns:
                        idx = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
                        x.index = pd.Index(idx)
                    else:
                        x.index = pd.to_datetime(x.index, errors="coerce").normalize()
                    # 不正な日時は除外
                    x = x[~x.index.isna()]
                    # 末尾N営業日相当を抽出
                    x = x.tail(int(lookback_days))
                    sliced[_sym] = x
                except Exception:
                    sliced[_sym] = _df
            sliced_dict = sliced
    except Exception:
        sliced_dict = raw_data_dict

    prepared = strategy.prepare_data(
        sliced_dict,
        progress_callback=progress_callback,
        log_callback=log_callback,
        use_process_pool=use_process_pool,
        max_workers=max_workers,
        lookback_days=lookback_days,
    )
    try:
        if log_callback:
            em, es = divmod(int(max(0, _t.time() - t0)), 60)
            log_callback(f"⏱️ フィルター/前処理 完了（経過 {em}分{es}秒）")
    except Exception:
        pass
    # フィルター通過件数（前営業日を優先。無い場合は最終行）。
    try:
        # 前営業日（当日エントリーのシグナルは前日の終値で判定）
        prev_day = pd.Timestamp(today) - pd.Timedelta(days=1)

        def _last_filter_on_date(x: pd.DataFrame) -> bool:
            try:
                if getattr(x, "empty", True) or "filter" not in x.columns:
                    return False
                # Date列があれば優先、無ければindexで比較
                if "Date" in x.columns:
                    dt = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
                    sel = x.loc[dt == prev_day, "filter"]
                else:
                    idx = pd.to_datetime(x.index, errors="coerce").normalize()
                    sel = x.loc[idx == prev_day, "filter"]
                if len(sel) > 0:
                    v = sel.iloc[-1]
                    return bool(False if pd.isna(v) else bool(v))
                # フォールバック: 最終行
                v = pd.Series(x["filter"]).tail(1).iloc[0]
                return bool(False if pd.isna(v) else bool(v))
            except Exception:
                return False

        filter_pass = sum(int(_last_filter_on_date(df)) for df in prepared.values())
    except Exception:
        filter_pass = 0
    if log_callback:
        try:
            log_callback(f"🧪 フィルターチェック完了：{filter_pass} 銘柄")
        except Exception:
            pass
    try:
        if stage_progress:
            stage_progress(25, filter_pass, None, None, None)
    except Exception:
        pass

    # 候補生成（market_df を必要とする実装に配慮）
    gen_fn = strategy.generate_candidates  # type: ignore[attr-defined]
    params = inspect.signature(gen_fn).parameters
    if log_callback:
        try:
            log_callback(f"🧩 セットアップチェック開始：{filter_pass} 銘柄")
        except Exception:
            pass
    t1 = _t.time()
    if "market_df" in params and market_df is not None:
        candidates_by_date, _ = gen_fn(
            prepared,
            market_df=market_df,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )
    else:
        candidates_by_date, _ = gen_fn(
            prepared,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )
    try:
        if log_callback:
            em, es = divmod(int(max(0, _t.time() - t1)), 60)
            log_callback(f"⏱️ セットアップ/候補抽出 完了（経過 {em}分{es}秒）")
    except Exception:
        pass

    # セットアップ通過件数（前営業日を優先。無ければ最終行）
    try:
        prev_day = pd.Timestamp(today) - pd.Timedelta(days=1)

        def _last_setup_on_date(x: pd.DataFrame) -> bool:
            try:
                if getattr(x, "empty", True) or "setup" not in x.columns:
                    return False
                if "Date" in x.columns:
                    dt = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
                    sel = x.loc[dt == prev_day, "setup"]
                else:
                    idx = pd.to_datetime(x.index, errors="coerce").normalize()
                    sel = x.loc[idx == prev_day, "setup"]
                if len(sel) > 0:
                    v = sel.iloc[-1]
                    return bool(False if pd.isna(v) else bool(v))
                v = pd.Series(x["setup"]).tail(1).iloc[0]
                return bool(False if pd.isna(v) else bool(v))
            except Exception:
                return False

        setup_pass = sum(int(_last_setup_on_date(df)) for df in prepared.values())
    except Exception:
        setup_pass = 0

    # system別セットアップ内訳（フィルタ通過→条件ごとの通過数）
    try:
        if log_callback:
            prev_day = pd.Timestamp(today) - pd.Timedelta(days=1)

            def _last_row(x: pd.DataFrame) -> pd.Series | None:
                try:
                    if "Date" in x.columns:
                        dt = pd.to_datetime(x["Date"], errors="coerce").dt.normalize()
                        rows = x.loc[dt == prev_day]
                    else:
                        idx = pd.to_datetime(x.index, errors="coerce").normalize()
                        rows = x.loc[idx == prev_day]
                    if len(rows) == 0:
                        rows = x.tail(1)
                    if len(rows) == 0:
                        return None
                    return rows.iloc[-1]
                except Exception:
                    return None

            name = system_name.lower()
            if name == "system1":
                sma_pass = 0
                for df in prepared.values():
                    row = _last_row(df)
                    if row is None:
                        continue
                    try:
                        filt = bool(row.get("filter"))
                        sma = float(row.get("SMA25", 0)) > float(row.get("SMA50", 0))
                        if filt and sma:
                            sma_pass += 1
                    except Exception:
                        continue
                msg = (
                    "🧩 system1セットアップ内訳: "
                    f"フィルタ通過={filter_pass}, SMA25>SMA50: {sma_pass}"
                )
                log_callback(msg)
            elif name == "system2":
                rsi_pass = 0
                two_up_pass = 0
                for df in prepared.values():
                    row = _last_row(df)
                    if row is None:
                        continue
                    try:
                        if not bool(row.get("filter")):
                            continue
                        if float(row.get("RSI3", 0)) > 90:
                            rsi_pass += 1
                            if bool(row.get("TwoDayUp")):
                                two_up_pass += 1
                    except Exception:
                        continue
                msg = (
                    "🧩 system2セットアップ内訳: "
                    f"フィルタ通過={filter_pass}, RSI3>90: {rsi_pass}, "
                    f"TwoDayUp: {two_up_pass}"
                )
                log_callback(msg)
            elif name == "system3":
                close_pass = 0
                drop_pass = 0
                for df in prepared.values():
                    row = _last_row(df)
                    if row is None:
                        continue
                    try:
                        if not bool(row.get("filter")):
                            continue
                        if float(row.get("Close", 0)) > float(row.get("SMA150", 0)):
                            close_pass += 1
                            if float(row.get("Drop3D", 0)) >= 0.125:
                                drop_pass += 1
                    except Exception:
                        continue
                msg = (
                    "🧩 system3セットアップ内訳: "
                    f"フィルタ通過={filter_pass}, Close>SMA150: {close_pass}, "
                    f"3日下落率>=12.5%: {drop_pass}"
                )
                log_callback(msg)
            elif name == "system4":
                above_sma = 0
                for df in prepared.values():
                    row = _last_row(df)
                    if row is None:
                        continue
                    try:
                        filt = bool(row.get("filter"))
                        over = float(row.get("Close", 0)) > float(row.get("SMA200", 0))
                        if filt and over:
                            above_sma += 1
                    except Exception:
                        continue
                msg = (
                    "🧩 system4セットアップ内訳: "
                    f"フィルタ通過={filter_pass}, Close>SMA200: {above_sma}"
                )
                log_callback(msg)
            elif name == "system5":
                price_pass = 0
                adx_pass = 0
                rsi_pass = 0
                for df in prepared.values():
                    row = _last_row(df)
                    if row is None:
                        continue
                    try:
                        if not bool(row.get("filter")):
                            continue
                        close_over = float(row.get("Close", 0)) > (
                            float(row.get("SMA100", 0)) + float(row.get("ATR10", 0))
                        )
                        if close_over:
                            price_pass += 1
                            if float(row.get("ADX7", 0)) > 55:
                                adx_pass += 1
                                if float(row.get("RSI3", 100)) < 50:
                                    rsi_pass += 1
                    except Exception:
                        continue
                msg = (
                    "🧩 system5セットアップ内訳: "
                    f"フィルタ通過={filter_pass}, Close>SMA100+ATR10: {price_pass}, "
                    f"ADX7>55: {adx_pass}, RSI3<50: {rsi_pass}"
                )
                log_callback(msg)
            elif name == "system6":
                ret_pass = 0
                up_pass = 0
                for df in prepared.values():
                    row = _last_row(df)
                    if row is None:
                        continue
                    try:
                        if not bool(row.get("filter")):
                            continue
                        if float(row.get("Return6D", 0)) > 0.20:
                            ret_pass += 1
                            if bool(row.get("UpTwoDays")):
                                up_pass += 1
                    except Exception:
                        continue
                msg = (
                    "🧩 system6セットアップ内訳: "
                    f"フィルタ通過={filter_pass}, Return6D>20%: {ret_pass}, "
                    f"UpTwoDays: {up_pass}"
                )
                log_callback(msg)
    except Exception:
        pass
    try:
        if stage_progress:
            stage_progress(50, filter_pass, setup_pass, None, None)
    except Exception:
        pass
    # トレード候補件数（当日のみ）
    try:
        total_candidates_today = len((candidates_by_date or {}).get(today, []) or [])
    except Exception:
        total_candidates_today = 0
    try:
        if stage_progress:
            stage_progress(75, filter_pass, setup_pass, total_candidates_today, None)
    except Exception:
        pass
    if log_callback:
        try:
            log_callback(f"🧩 セットアップチェック完了：{setup_pass} 銘柄")
            log_callback(f"🧮 トレード候補選定開始：{setup_pass} 銘柄")
            log_callback(
                f"🧮 トレード候補選定完了（当日）：{total_candidates_today} 銘柄"
            )
        except Exception:
            pass

    if not candidates_by_date:
        return pd.DataFrame(
            columns=[
                "symbol",
                "system",
                "side",
                "signal_type",
                "entry_date",
                "entry_price",
                "stop_price",
                "score_key",
                "score",
            ]
        )

    # 当日分のみ抽出
    today_candidates: list[dict] = candidates_by_date.get(today, [])  # type: ignore
    if not today_candidates:
        return pd.DataFrame(
            columns=[
                "symbol",
                "system",
                "side",
                "signal_type",
                "entry_date",
                "entry_price",
                "stop_price",
                "score_key",
                "score",
            ]
        )
    rows: list[TodaySignal] = []
    for c in today_candidates:
        sym = c.get("symbol")
        if not sym or sym not in prepared:
            continue
        df = prepared[sym]
        comp = _compute_entry_stop(strategy, df, c, side)
        if not comp:
            continue
        entry, stop = comp
        skey, sval, _asc = _score_from_candidate(system_name, c)

        # System1 は ROC200 を必ずスコアに採用できるよう堅牢化
        try:
            if (system_name == "system1") and (
                skey is None or str(skey).upper() != "ROC200"
            ):
                skey = "ROC200"
        except Exception:
            pass

        # signal 日（通常は entry_date の前営業日を想定）
        signal_date_ts: pd.Timestamp | None = None
        try:
            # candidate["Date"] があれば優先
            if "Date" in c and c.get("Date") is not None:
                date_arg: Any = c.get("Date")
                tmp = pd.to_datetime(date_arg, errors="coerce")
                if not pd.isna(tmp):
                    signal_date_ts = pd.Timestamp(tmp).normalize()
        except Exception:
            # フォールバックは後段の entry_date 補完に任せる
            pass
        if signal_date_ts is None:
            try:
                ed_arg: Any = c.get("entry_date")
                ed = pd.to_datetime(ed_arg, errors="coerce")
                if isinstance(ed, pd.Timestamp) and not pd.isna(ed):
                    signal_date_ts = ed.normalize() - pd.Timedelta(days=1)
            except Exception:
                signal_date_ts = None

        # 欠損スコアの補完（まず値、次に順位）
        rank_val: int | None = None
        total_for_rank: int = 0
        if skey is not None:
            # 1) 欠損なら prepared から同日値を補完
            if sval is None or (isinstance(sval, float) and pd.isna(sval)):
                try:
                    if signal_date_ts is not None:
                        row = prepared[sym][
                            pd.to_datetime(
                                prepared[sym]["Date"]
                                if "Date" in prepared[sym].columns
                                else prepared[sym].index
                            )
                            .normalize()
                            .eq(signal_date_ts)
                        ]
                        if not row.empty and skey in row.columns:
                            _v = row.iloc[0][skey]
                            if _v is not None and not pd.isna(_v):
                                sval = float(_v)
                except Exception:
                    pass
            # System1 用のフォールバック（前日が見つからない場合は直近値）
            if (system_name == "system1") and (
                sval is None or (isinstance(sval, float) and pd.isna(sval))
            ):
                try:
                    if skey in prepared[sym].columns:
                        _v = pd.Series(prepared[sym][skey]).dropna().tail(1).iloc[0]
                        sval = float(_v)
                except Exception:
                    pass

            # 2) 値がまだ欠損なら、同日全銘柄の順位を算出してスコアに設定
            try:
                if signal_date_ts is not None:
                    vals: list[tuple[str, float]] = []
                    for psym, pdf in prepared.items():
                        try:
                            if "Date" in pdf.columns:
                                row = pdf[
                                    pd.to_datetime(pdf["Date"]).dt.normalize()
                                    == signal_date_ts
                                ]
                            else:
                                row = pdf[
                                    pd.to_datetime(pdf.index).normalize()
                                    == signal_date_ts
                                ]
                            if not row.empty and skey in row.columns:
                                v = row.iloc[0][skey]
                                if v is not None and not pd.isna(v):
                                    vals.append((psym, float(v)))
                        except Exception:
                            continue
                    total_for_rank = len(vals)
                    if total_for_rank:
                        # 並び順: system の昇降順推定に合わせる（ROC200 などは降順）
                        reverse = not _asc
                        # 値が同一のときはシンボルで安定ソート
                        vals_sorted = sorted(
                            vals, key=lambda t: (t[1], t[0]), reverse=reverse
                        )
                        # 自銘柄の順位を決定
                        symbols_sorted = [s for s, _ in vals_sorted]
                        if sym in symbols_sorted:
                            rank_val = symbols_sorted.index(sym) + 1
                        # スコアが欠損なら順位をそのままスコアに採用
                        if sval is None or (isinstance(sval, float) and pd.isna(sval)):
                            if rank_val is not None:
                                sval = float(rank_val)
            except Exception:
                pass

        # 選定理由（順位を最優先、なければ簡潔かつシステム固有の文言）
        reason_parts: list[str] = []
        # System1 は日本語で「ROC200がn位のため」に統一（順位が取れない場合のみ汎用文言）
        if system_name == "system1":
            if rank_val is not None:
                reason_parts = [f"ROC200が{rank_val}位のため"]
            else:
                reason_parts = ["ROC200が上位のため"]
        elif system_name == "system2":
            if rank_val is not None and skey is not None:
                reason_parts = [f"{_label_for_score_key(skey)}が{rank_val}位のため"]
            else:
                reason_parts = ["モメンタムが強く過熱のため"]
        elif system_name == "system3":
            if rank_val is not None and skey is not None:
                reason_parts = [f"{_label_for_score_key(skey)}が{rank_val}位のため"]
            else:
                reason_parts = ["ボラティリティが高く条件一致のため"]
        elif system_name == "system4":
            if rank_val is not None:
                reason_parts = [f"RSI4が{rank_val}位（低水準）のため"]
            else:
                reason_parts = ["SPY上昇局面の押し目候補のため"]
        elif system_name == "system5":
            if rank_val is not None and skey is not None:
                reason_parts = [f"{_label_for_score_key(skey)}が{rank_val}位のため"]
            else:
                reason_parts = ["ADXが強く、反発期待のため"]
        elif system_name == "system6":
            if rank_val is not None:
                reason_parts = [f"過去6日騰落率が{rank_val}位のため"]
            else:
                reason_parts = ["短期下落トレンド（ショート）条件一致のため"]
        else:
            if skey is not None and rank_val is not None:
                if rank_val <= 10:
                    reason_parts = [f"{_label_for_score_key(skey)}が{rank_val}位のため"]
                else:
                    reason_parts = [f"rank={rank_val}/{total_for_rank}"]
            elif skey is not None:
                # 値は原則非表示（冗長回避）。必要最小限だけ示す。
                try:
                    if sval is not None and not (
                        isinstance(sval, float) and pd.isna(sval)
                    ):
                        reason_parts.append("スコア条件を満たしたため")
                except Exception:
                    reason_parts.append("スコア条件を満たしたため")

        # fallback generic info
        if not reason_parts:
            reason_parts.append("条件一致のため")

        reason_text = "; ".join(reason_parts)

        try:
            _ed_raw: Any = c.get("entry_date")
            _ed = pd.Timestamp(_ed_raw) if _ed_raw is not None else None
            if _ed is None or pd.isna(_ed):
                # entry_date が欠損する候補は無効
                continue
            entry_date_norm = pd.Timestamp(_ed).normalize()
        except Exception:
            continue

        rows.append(
            TodaySignal(
                symbol=str(sym),
                system=system_name,
                side=side,
                signal_type=signal_type,
                entry_date=entry_date_norm,
                entry_price=float(entry),
                stop_price=float(stop),
                score_key=skey,
                # スコアは値があれば値、無ければ順位（上記で補完済み）
                score=(
                    None
                    if sval is None or (isinstance(sval, float) and pd.isna(sval))
                    else float(sval)
                ),
                reason=reason_text,
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "system",
                "side",
                "signal_type",
                "entry_date",
                "entry_price",
                "stop_price",
                "score_key",
                "score",
            ]
        )

    out = pd.DataFrame([r.__dict__ for r in rows])
    try:
        if stage_progress:
            stage_progress(
                100, filter_pass, setup_pass, total_candidates_today, len(rows)
            )
    except Exception:
        pass
    return out


__all__ = [
    "get_today_signals_for_strategy",
    "LONG_SYSTEMS",
    "SHORT_SYSTEMS",
]
