# -*- coding: utf-8 -*-
"""
共通UIコンポーネント（UTF-8・日本語対応）。
既存の公開API（関数名・戻り値）は維持しつつ、各フェーズ（データ取得/インジ計算/候補抽出/バックテスト）で
UIManager（任意）に進捗とログを出力できるようにしている。
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, Tuple, overload

import matplotlib as mpl
from matplotlib import font_manager as _font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import streamlit as st

from common.utils import safe_filename, get_cached_data

try:
    # 設定からUIフラグを参照（失敗時はデフォルト動作にフォールバック）
    from config.settings import get_settings

    _APP_SETTINGS = get_settings(create_dirs=True)
except Exception:
    _APP_SETTINGS = None
from scripts.tickers_loader import get_all_tickers
from common.cache_manager import load_base_cache, base_cache_path
from common.holding_tracker import (
    generate_holding_matrix,
    display_holding_heatmap,
    download_holding_csv,
)
from core.system1 import generate_roc200_ranking_system1
import common.i18n as i18n

# 互換用エイリアス（既存コードの tr(...) 呼び出しを維持）
tr = i18n.tr
import matplotlib as mpl
import logging
from matplotlib import font_manager as _font_manager


# ------------------------------
# Type overloads for static checkers
# ------------------------------
@overload
def fetch_data(
    symbols: Iterable[str], max_workers: int = 8, ui_manager: object | None = None
) -> Dict[str, pd.DataFrame]: ...


@overload
def prepare_backtest_data(
    strategy: Any,
    symbols: Iterable[str],
    system_name: str = "SystemX",
    spy_df: pd.DataFrame | None = None,
    ui_manager: object | None = None,
    use_process_pool: bool = False,
    **kwargs: Any,
) -> tuple[dict[str, pd.DataFrame] | None, Any | None, pd.DataFrame | None]: ...


@overload
def run_backtest_app(
    strategy: Any,
    system_name: str = "SystemX",
    limit_symbols: int | None = None,
    system_title: str | None = None,
    spy_df: pd.DataFrame | None = None,
    ui_manager: object | None = None,
    **kwargs: Any,
) -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    dict[str, pd.DataFrame] | None,
    float,
    Any | None,
]: ...


@overload
def show_signal_trade_summary(
    source_df: pd.DataFrame | Dict[str, pd.DataFrame] | None,
    trades_df: pd.DataFrame | None,
    system_name: str,
    display_name: str | None = None,
) -> pd.DataFrame: ...


@overload
def save_signal_and_trade_logs(
    signal_counts_df: pd.DataFrame | None,
    results: pd.DataFrame | list[dict[str, Any]] | None,
    system_name: str,
    capital: float,
) -> None: ...


@overload
def save_prepared_data_cache(
    data_dict: Dict[str, pd.DataFrame], system_name: str = "SystemX"
) -> None: ...


@overload
def show_results(
    results_df: pd.DataFrame,
    capital: float,
    system_name: str = "SystemX",
    *,
    key_context: str = "main",
) -> None: ...


# 日本語表示のためのフォントフォールバック（Windows向け優先）
def _set_japanese_font_fallback() -> None:
    """日本語フォントをインストール済みのものだけに設定して警告を回避する。"""
    try:
        preferred = [
            "Noto Sans JP",
            "IPAexGothic",
            "Yu Gothic",
            "Meiryo",
            "MS Gothic",
            "Yu Gothic UI",
            "MS PGothic",
            "Hiragino Sans",
            "Hiragino Kaku Gothic ProN",
            "TakaoGothic",
            "DejaVu Sans",
        ]
        available = {f.name for f in _font_manager.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        if not chosen:
            chosen = ["DejaVu Sans"]
        mpl.rcParams["font.family"] = chosen
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


_set_japanese_font_fallback()

# matplotlib.font_manager の冗長な INFO を抑制
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# ------------------------------
# Small utilities
# ------------------------------
def clean_date_column(df: pd.DataFrame, col_name: str = "Date") -> pd.DataFrame:
    if col_name in df.columns:
        df = df.copy()
        df[col_name] = pd.to_datetime(df[col_name], errors="coerce")
        df = df.dropna(subset=[col_name])
    return df


def log_with_progress(
    i: int,
    total: int,
    start_time: float,
    *,
    prefix: str = "進捗",
    batch: int = 50,
    log_area=None,
    progress_bar=None,
    extra_msg: str | None = None,
    unit: str = "件",
) -> None:
    if i % batch == 0 or i == total:
        elapsed = time.time() - start_time
        remain = (elapsed / i) * (total - i) if i > 0 else 0
        msg = (
            f"{prefix}: {i}/{total} {unit} | 経過: {int(elapsed // 60)}分{int(elapsed % 60)}秒"
            f" / 残り目安: 約{int(remain // 60)}分{int(remain % 60)}秒"
        )
        if extra_msg:
            msg += f"\n{extra_msg}"
        try:
            if log_area is not None:
                log_area.text(msg)
        except Exception:
            pass
        try:
            if progress_bar is not None:
                progress_bar.progress(0 if total == 0 else i / total)
        except Exception:
            pass


def default_log_callback(
    processed: int, total: int, start_time: float, prefix: str = "📊 状況"
) -> str:
    elapsed = time.time() - start_time
    remain = (elapsed / processed) * (total - processed) if processed else 0
    return (
        f"{prefix}: {processed}/{total} 件 | 経過: {int(elapsed // 60)}分{int(elapsed % 60)}秒"
        f" / 残り目安: 約{int(remain // 60)}分{int(remain % 60)}秒"
    )


# ------------------------------
# Data fetch
# ------------------------------
def _mtime_or_zero(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0


@st.cache_data(show_spinner=False)
def _load_symbol_cached(
    symbol: str, *, base_path: str, base_mtime: float, raw_path: str, raw_mtime: float
) -> tuple[str, pd.DataFrame | None]:
    """ファイルの更新時刻をキーに含めてキャッシュし、ファイル更新で自動無効化。
    戻り値は (symbol, DataFrame|None)
    """
    try:
        df = load_base_cache(symbol, rebuild_if_missing=True)
        if df is not None and not df.empty:
            return symbol, df
    except Exception:
        pass
    if os.path.exists(raw_path):
        return symbol, get_cached_data(symbol)
    return symbol, None


def load_symbol(
    symbol: str, cache_dir: str = "data_cache"
) -> tuple[str, pd.DataFrame | None]:
    base_path = str(base_cache_path(symbol))
    raw_path = os.path.join(cache_dir, f"{safe_filename(symbol)}.csv")
    return _load_symbol_cached(
        symbol,
        base_path=base_path,
        base_mtime=_mtime_or_zero(base_path),
        raw_path=raw_path,
        raw_mtime=_mtime_or_zero(raw_path),
    )


@overload
def fetch_data(
    symbols: Iterable[str], max_workers: int = 8, ui_manager: object | None = None
) -> Dict[str, pd.DataFrame]: ...


def fetch_data(
    symbols, max_workers: int = 8, ui_manager=None
) -> Dict[str, pd.DataFrame]:
    data_dict: Dict[str, pd.DataFrame] = {}
    total = len(symbols)
    # UIManagerのフェーズ（fetch）があればそこへ出力
    phase = ui_manager.phase("fetch") if ui_manager else None
    if phase:
        progress_bar = phase.progress_bar
        log_area = phase.log_area
        # フェーズ配下に「no data」用の別スロットを確保（未作成なら生成）
        no_data_area = getattr(phase, "no_data_area", None)
        if no_data_area is None:
            try:
                no_data_area = getattr(phase, "container").empty()
            except Exception:
                no_data_area = st.empty()
            try:
                setattr(phase, "no_data_area", no_data_area)
            except Exception:
                pass
        try:
            phase.info(tr("fetch: start | {total} symbols", total=total))
        except Exception:
            pass
    else:
        st.info(tr("fetch: start | {total} symbols", total=total))
        progress_bar = st.progress(0)
        log_area = st.empty()
        # フェーズ未使用時は直下にno-data用スロットを用意
        no_data_area = st.empty()
    buffer, skipped, start_time = [], [], time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_symbol, sym): sym for sym in symbols}
        for i, future in enumerate(as_completed(futures), 1):
            sym, df = future.result()
            if df is not None and not df.empty:
                data_dict[sym] = df
                buffer.append(sym)
            else:
                skipped.append(sym)

            if i % 50 == 0 or i == total:
                log_with_progress(
                    i,
                    total,
                    start_time,
                    prefix="データ取得",
                    batch=50,
                    log_area=log_area,
                    progress_bar=progress_bar,
                    extra_msg=(f"銘柄: {', '.join(buffer)}" if buffer else None),
                )
                buffer.clear()

    try:
        progress_bar.empty()
    except Exception:
        pass
    if skipped:
        try:
            # use i18n message for skipped count, append symbols list
            # tr は kwargs を受けてフォーマット済み文字列を返すので .format は不要
            msg = tr("⚠️ no data: {n} symbols", n=len(skipped))
            # 長大なリストを避けるため、代表のみ（先頭10件）を表示
            _sample = list(skipped)[:10]
            msg = msg + "\n" + ", ".join(_sample)
            _rest = len(skipped) - len(_sample)
            if _rest > 0:
                msg += f"\n... (+{_rest} more)"
            # 取得ログを上書きせず、下の行に表示
            no_data_area.text(msg)
        except Exception:
            pass
    return data_dict


# ------------------------------
# Prepare + candidates
# ------------------------------
@overload
def prepare_backtest_data(
    strategy: Any,
    symbols: Iterable[str],
    system_name: str = "SystemX",
    spy_df: pd.DataFrame | None = None,
    ui_manager: object | None = None,
    **kwargs: Any,
) -> tuple[dict[str, pd.DataFrame] | None, Any | None, pd.DataFrame | None]: ...


def prepare_backtest_data(
    strategy,
    symbols,
    system_name: str = "SystemX",
    spy_df: pd.DataFrame | None = None,
    ui_manager=None,
    use_process_pool: bool = False,
    **kwargs,
):
    # 1) fetch
    if use_process_pool:
        data_dict = None
    else:
        data_dict = fetch_data(symbols, ui_manager=ui_manager)
        if not data_dict:
            st.error(tr("no valid data"))
            return None, None, None

    # 2) indicators (delegated to strategy)
    # indicators フェーズ
    ind_phase = ui_manager.phase("indicators") if ui_manager else None
    if ind_phase:
        try:
            ind_phase.info(tr("indicators: computing..."))
        except Exception:
            pass
        ind_progress = ind_phase.progress_bar
        ind_log = ind_phase.log_area
    else:
        st.info(tr("indicators: computing..."))
        ind_progress = st.progress(0)
        ind_log = st.empty()
    start_time = time.time()
    call_input = data_dict if not use_process_pool else symbols
    call_kwargs = dict(
        progress_callback=lambda done, total: ind_progress.progress(
            0 if total == 0 else done / total
        ),
        log_callback=lambda msg: ind_log.text(str(msg)),
        skip_callback=lambda msg: ind_log.text(str(msg)),
        **kwargs,
    )
    if use_process_pool:
        call_kwargs["use_process_pool"] = True

    try:
        prepared_dict = strategy.prepare_data(call_input, **call_kwargs)
    except TypeError:
        # 古い戦略実装との後方互換: skip_callback/use_process_pool 未対応の戦略に再試行
        call_kwargs.pop("skip_callback", None)
        call_kwargs.pop("use_process_pool", None)
        prepared_dict = strategy.prepare_data(call_input, **call_kwargs)
    try:
        ind_progress.empty()
    except Exception:
        pass

    # 3) candidates
    # candidates フェーズ
    cand_phase = ui_manager.phase("candidates") if ui_manager else None
    if cand_phase:
        try:
            cand_phase.info(tr("candidates: extracting..."))
        except Exception:
            pass
        cand_log = cand_phase.log_area
        cand_progress = cand_phase.progress_bar
    else:
        st.info(tr("candidates: extracting..."))
        cand_log = st.empty()
        cand_progress = st.progress(0)
    start_time = time.time()

    merged_df = None
    if system_name == "System1":
        if spy_df is None or spy_df.empty:
            st.error(tr("System1 requires SPY data for market filter"))
            return prepared_dict, None, None
        candidates_by_date, merged_df = generate_roc200_ranking_system1(
            prepared_dict,
            spy_df,
            on_progress=lambda i, total, start: log_with_progress(
                i,
                total,
                start,
                prefix="📈 ROC200ランキング",
                log_area=cand_log,
                progress_bar=cand_progress,
                unit=tr("days"),
            ),
            on_log=None,
        )
    else:
        # generic path (System2–7)
        try:
            candidates_by_date = strategy.generate_candidates(
                prepared_dict,
                progress_callback=lambda done, total: log_with_progress(
                    done,
                    total,
                    start_time,
                    prefix="candidates",
                    log_area=cand_log,
                    progress_bar=cand_progress,
                ),
                log_callback=lambda msg: cand_log.text(str(msg)),
                **kwargs,
            )
        except (TypeError, ValueError):
            # 戻り値の形 or 引数不一致（例: System4 の market_df）に対応
            if system_name == "System4" and spy_df is not None:
                ret = strategy.generate_candidates(
                    prepared_dict,
                    market_df=spy_df,
                    **kwargs,
                )
            else:
                ret = strategy.generate_candidates(
                    prepared_dict,
                    **kwargs,
                )
            if isinstance(ret, tuple) and len(ret) == 2:
                candidates_by_date, merged_df = ret
            else:
                candidates_by_date = ret
    # 正常系でも (dict, df) を返す実装があるため後段で正規化
    if isinstance(candidates_by_date, tuple) and len(candidates_by_date) == 2:
        candidates_by_date, merged_df = candidates_by_date
    try:
        cand_progress.empty()
    except Exception:
        pass

    if not candidates_by_date:
        st.warning(tr("{system_name}: no candidates"))
        return prepared_dict, None, None

    return prepared_dict, candidates_by_date, merged_df


# ------------------------------
# Backtest execution (common wrapper)
# ------------------------------
def run_backtest_with_logging(
    strategy,
    prepared_dict,
    candidates_by_date,
    capital,
    system_name: str = "SystemX",
    ui_manager=None,
):
    bt_phase = ui_manager.phase("backtest") if ui_manager else None
    if bt_phase:
        try:
            bt_phase.info(tr("backtest: running..."))
        except Exception:
            pass
        progress = bt_phase.progress_bar
        log_area = bt_phase.log_area
        # 資金推移は最新行のみ、エクスパンダーは使わず単一プレースホルダに出力
        fund_log_area = getattr(bt_phase, "fund_log_area", bt_phase.container.empty())
        try:
            setattr(bt_phase, "fund_log_area", fund_log_area)
        except Exception:
            pass
        debug_area = bt_phase.container.empty()
    else:
        st.info(tr("backtest: running..."))
        progress = st.progress(0)
        log_area = st.empty()
        fund_log_area = st.empty()
        debug_area = st.empty()
    start_time = time.time()
    debug_logs: list[str] = []

    def handle_log(msg):
        if isinstance(msg, str) and msg.startswith("💰"):
            # attempt to localize capital/active segments while preserving date
            import re

            s = str(msg)
            # Capital: 3812.31 USD -> 資金: 3812.31 USD
            s = re.sub(r"Capital:\s*([0-9\.,]+)\s*USD", r"資金: \1 USD", s)
            # Active: 0 -> 保有ポジション: 0
            s = re.sub(r"Active:\s*([0-9]+)", r"保有ポジション: \1", s)
            debug_logs.append(s)
            # 最新行のみを表示（差し替え）
            fund_log_area.text(s)
        else:
            log_area.text(str(msg))

    results_df = strategy.run_backtest(
        prepared_dict,
        candidates_by_date,
        capital,
        on_progress=lambda i, total, start: log_with_progress(
            i,
            total,
            start,
            prefix="bt",
            log_area=log_area,
            progress_bar=progress,
            unit="days",
        ),
        on_log=lambda msg: handle_log(msg),
    )

    try:
        progress.empty()
    except Exception:
        pass

    # ログをセッションへ保持（リランしても表示できるように）
    st.session_state[f"{system_name}_debug_logs"] = list(debug_logs)

    if st.session_state.get("show_debug_logs", True) and debug_logs:
        # ログはバックテスト・フェーズのコンテナ内に配置（システムごとにまとまるように）
        parent = bt_phase.container if bt_phase else st.container()
        # ユーザー要望: 取引ログはエクスパンダーで折りたたみ表示
        title = f"💰 {tr('trade logs')}"
        with parent.expander(title, expanded=False):
            # text_area の方が行間・スクロールで視認性が高い
            st.text_area(
                "Logs",
                "\n".join(debug_logs),
                height=300,
            )

    # 結果も併せてセッションに保存（UI層でも保存するが二重でも安全）
    st.session_state[f"{system_name}_results_df"] = results_df
    return results_df


# ------------------------------
# App entry for a single system tab
# ------------------------------
@overload
def run_backtest_app(
    strategy: Any,
    system_name: str = "SystemX",
    limit_symbols: int | None = None,
    system_title: str | None = None,
    spy_df: pd.DataFrame | None = None,
    ui_manager: object | None = None,
    **kwargs: Any,
) -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    dict[str, pd.DataFrame] | None,
    float,
    Any | None,
]: ...


def run_backtest_app(
    strategy,
    system_name: str = "SystemX",
    limit_symbols: int = 10,
    system_title: str | None = None,
    spy_df: pd.DataFrame | None = None,
    ui_manager=None,
    **kwargs,
):
    st.title(system_title or f"{system_name} backtest")

    # --- 前回実行結果の表示/クリア（セッション保持） ---
    key_results = f"{system_name}_results_df"
    key_prepared = f"{system_name}_prepared_dict"
    key_cands = f"{system_name}_candidates_by_date"
    key_capital = f"{system_name}_capital"
    key_capital_saved = f"{system_name}_capital_saved"
    key_merged = f"{system_name}_merged_df"
    key_debug = f"{system_name}_debug_logs"

    has_prev = any(
        k in st.session_state
        for k in [key_results, key_cands, f"{system_name}_capital_saved"]
    )
    if has_prev:
        with st.expander("前回の結果（リランでも保持）", expanded=False):
            prev_res = st.session_state.get(key_results)
            prev_cap = st.session_state.get(
                key_capital_saved, st.session_state.get(key_capital, 0)
            )
            if prev_res is not None and getattr(prev_res, "empty", False) is False:
                show_results(prev_res, prev_cap, system_name, key_context="prev")
            dbg = st.session_state.get(key_debug)
            if dbg:
                # Streamlit の制約により Expander 同声の入れ子は不可
                # 内側の expander を通常表示に変更
                st.markdown("**保存済み 取引ログ**")
                st.text("\n".join(map(str, dbg)))
            if st.button(tr("保存済み結果をクリア"), key=f"{system_name}_clear_saved"):
                for k in [
                    key_results,
                    key_prepared,
                    key_cands,
                    key_capital_saved,
                    key_capital,
                    key_merged,
                    key_debug,
                ]:
                    if k in st.session_state:
                        del st.session_state[k]
                # 型チェッカーや古い Streamlit 実装に対応するため存在を確認してから呼び出す
                if hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()

    if st.button(tr("clear streamlit cache"), key=f"{system_name}_clear_cache"):
        st.cache_data.clear()
        st.success(tr("cache cleared"))

    debug_key = f"{system_name}_show_debug_logs"
    if debug_key not in st.session_state:
        st.session_state[debug_key] = True
    st.checkbox(tr("show debug logs"), key=debug_key)

    use_auto = st.checkbox(
        tr("auto symbols (all tickers)"), value=True, key=f"{system_name}_auto"
    )
    _init_cap = int(st.session_state.get(key_capital_saved, 1000))
    capital = st.number_input(
        tr("capital (USD)"),
        min_value=1000,
        value=_init_cap,
        step=100,
        key=f"{system_name}_capital",
    )

    all_tickers = get_all_tickers()
    max_allowed = len(all_tickers)
    default_value = min(10, max_allowed)

    if system_name != "System7":
        # テスト用でも使いやすいように最小値を1に、刻み幅を1に変更
        limit_symbols = st.number_input(
            tr("symbol limit"),
            min_value=1,
            max_value=max_allowed,
            value=default_value,
            step=1,
            key=f"{system_name}_limit",
        )
        if st.checkbox(tr("use all symbols"), key=f"{system_name}_all"):
            limit_symbols = max_allowed

    symbols_input = None
    if not use_auto:
        symbols_input = st.text_input(
            tr("symbols (comma separated)"),
            "AAPL,MSFT,TSLA,NVDA,META",
            key=f"{system_name}_symbols_main",
        )

    if system_name == "System7":
        symbols = ["SPY"]
    elif use_auto:
        symbols = all_tickers[:limit_symbols]
    else:
        if not symbols_input:
            st.error(tr("please input symbols"))
            return None, None, None, None, None
        symbols = [s.strip().upper() for s in symbols_input.split(",")]

    # System1 専用: 実行ボタンの直前に通知トグルを配置
    if system_name in (
        "System1",
        "System2",
        "System3",
        "System4",
        "System5",
        "System6",
        "System7",
    ):
        _notify_key = f"{system_name}_notify_backtest"
        if _notify_key not in st.session_state:
            st.session_state[_notify_key] = True
        _label = tr("バックテスト結果を通知する（Webhook）")
        try:
            _use_toggle = hasattr(st, "toggle")
        except Exception:
            _use_toggle = False
        if _use_toggle:
            st.toggle(_label, key=_notify_key)
        else:
            st.checkbox(_label, key=_notify_key)
        try:
            import os as _os  # local alias to avoid top imports churn

            if not (
                _os.getenv("DISCORD_WEBHOOK_URL") or _os.getenv("SLACK_BOT_TOKEN")
            ):
                st.caption(tr("Webhook/Bot 設定が未設定です（.env を確認）"))
        except Exception:
            pass

    run_clicked = st.button(tr("run"), key=f"{system_name}_run")
    result_area = st.container()
    if run_clicked:
        with result_area:
            prepared_dict, candidates_by_date, merged_df = prepare_backtest_data(
                strategy,
                symbols,
                system_name=system_name,
                spy_df=spy_df,
                ui_manager=ui_manager,
                **kwargs,
            )
            if candidates_by_date is None:
                return None, None, None, None, None

            results_df = run_backtest_with_logging(
                strategy,
                prepared_dict,
                candidates_by_date,
                capital,
                system_name,
                ui_manager=ui_manager,
            )
            show_results(results_df, capital, system_name, key_context="curr")

            # セッションへ保存（リラン対策）
            st.session_state[key_results] = results_df
            st.session_state[key_prepared] = prepared_dict
            st.session_state[key_cands] = candidates_by_date
            st.session_state[key_capital_saved] = capital
            if merged_df is not None:
                st.session_state[key_merged] = merged_df

            if system_name == "System1":
                return results_df, merged_df, prepared_dict, capital, candidates_by_date
            else:
                return results_df, None, prepared_dict, capital, candidates_by_date

    return None, None, None, None, None


# ------------------------------
# Rendering helpers
# ------------------------------
def summarize_results(results_df: pd.DataFrame, capital: float):
    df = results_df.copy()

    # 日付を確実に日時型に
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    # 基本集計
    df = df.sort_values("exit_date").reset_index(drop=True)
    trades = len(df)
    total_return = float(df["pnl"].sum()) if "pnl" in df.columns else 0.0
    wins = int((df["pnl"] > 0).sum()) if "pnl" in df.columns else 0
    win_rate = (wins / trades * 100.0) if trades > 0 else 0.0

    # exit_date 基準で累積PnL を作成（グラフ用）
    df2 = df.copy()
    if "pnl" in df2.columns:
        df2["cumulative_pnl"] = df2["pnl"].cumsum()
    else:
        df2["cumulative_pnl"] = 0.0

    # 日次保有状態・エクイティ等（簡易版）
    # cumulative_pnl からドローダウンを計算
    try:
        cum = df2["cumulative_pnl"].astype(float)
        dd_series = cum - cum.cummax()
        max_dd = float(abs(dd_series.min()))
    except Exception:
        max_dd = 0.0

    summary = {
        "trades": int(trades),
        "total_return": float(total_return),
        "win_rate": float(win_rate),
        "max_dd": float(max_dd),
    }

    # 呼び出し元は (summary, df2) を期待しているため返す
    return summary, df2


@overload
def show_results(
    results_df: pd.DataFrame,
    capital: float,
    system_name: str = "SystemX",
    *,
    key_context: str = "main",
) -> None: ...


def show_results(
    results_df: pd.DataFrame,
    capital: float,
    system_name: str = "SystemX",
    *,
    key_context: str = "main",
):
    if results_df is None or results_df.empty:
        st.info(i18n.tr("no trades"))
        return

    st.success(i18n.tr("backtest finished"))
    st.subheader(i18n.tr("results"))
    st.dataframe(results_df)

    # デバッグ: 列名・型・先頭数行を表示（max drawdown が0の原因確認用、確認後は削除してください）
    # removed debug: results_df.head()
    # removed debug: results_df.columns
    # removed debug: results_df.dtypes

    # 一部環境で summarize_results が 2 引数版でラップされていることがあるため、
    # system_name 固有のデバッグフラグを一時的に共通キーへコピーしてから呼び出す
    try:
        prev_flag = st.session_state.get("show_debug_logs", None)
        # system_name 固有フラグがあれば優先して一時的にセット
        sys_flag = st.session_state.get(f"{system_name}_show_debug_logs", None)
        if sys_flag is not None:
            st.session_state["show_debug_logs"] = sys_flag
    except Exception:
        prev_flag = None

    # 互換呼び出し（2 引数版でも動作するようにする）
    summary, df2 = summarize_results(results_df, capital)
    # 最大ドローダウンを再計算して summary に反映（表示のゼロを防止）
    try:
        cum = df2["cumulative_pnl"].astype(float)
        dd_series = cum - cum.cummax()
        max_dd_val = float(abs(dd_series.min()))
        try:
            summary["max_dd"] = max_dd_val
        except Exception:
            pass
    except Exception:
        pass

    # フラグを元に戻す
    try:
        if prev_flag is None:
            if "show_debug_logs" in st.session_state:
                del st.session_state["show_debug_logs"]
        else:
            st.session_state["show_debug_logs"] = prev_flag
    except Exception:
        pass

    # Series/Dict いずれにも安全に対応し、欠損キーは 0 扱い
    if isinstance(summary, pd.Series):
        summary = summary.to_dict()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("取引数", int(summary.get("trades", 0)))
    col2.metric("合計損益", f"{float(summary.get('total_return', 0.0)):.2f}")
    col3.metric("勝率 (%)", f"{float(summary.get('win_rate', 0.0)):.2f}")
    col4.metric("最大ドローダウン", f"{float(summary.get('max_dd', 0.0)):.2f}")

    st.subheader(i18n.tr("cumulative pnl"))
    # 日本語を軸ラベルに使う際のフォントフォールバック設定（環境にあるフォントを優先して選択）
    try:
        _set_japanese_font_fallback()
    except Exception:
        pass
    plt.figure(figsize=(10, 4))
    plt.plot(df2["exit_date"], df2["cumulative_pnl"], label="CumPnL")
    # Drawdown（累積損益のピークからの下落）を赤線で重ねる
    try:
        cum = df2["cumulative_pnl"].astype(float)
        dd = cum - cum.cummax()
        plt.plot(df2["exit_date"], dd, color="red", linewidth=1.2, label="Drawdown")
    except Exception:
        pass
    plt.xlabel(i18n.tr("date"))
    plt.ylabel(i18n.tr("pnl"))
    plt.legend()
    # streamlit.pyplot には Figure を渡す（plt モジュールそのものを渡さない）
    try:
        fig = plt.gcf()
        st.pyplot(fig)
    except Exception:
        # フォールバック: 直接渡すのは避けるがエラー時は無視
        pass

    st.subheader(i18n.tr("yearly summary"))
    yearly = df2.groupby(df2["exit_date"].dt.to_period("Y"))["pnl"].sum().reset_index()
    yearly["損益"] = yearly["pnl"].round(2)
    yearly["リターン(%)"] = yearly["pnl"] / (capital if capital else 1) * 100
    yearly = yearly.rename(columns={"exit_date": "年"})
    st.dataframe(
        yearly[["年", "損益", "リターン(%)"]].style.format(
            {"損益": "{:.2f}", "リターン(%)": "{:.1f}%"}
        )
    )
    st.subheader(i18n.tr("monthly summary"))
    monthly = df2.groupby(df2["exit_date"].dt.to_period("M"))["pnl"].sum().reset_index()
    monthly["損益"] = monthly["pnl"].round(2)
    monthly["リターン(%)"] = monthly["pnl"] / (capital if capital else 1) * 100
    monthly = monthly.rename(columns={"exit_date": "月"})
    st.dataframe(
        monthly[["月", "損益", "リターン(%)"]].style.format(
            {"損益": "{:.2f}", "リターン(%)": "{:.1f}%"}
        )
    )

    st.subheader(i18n.tr("holdings heatmap (by day)"))
    progress_heatmap = st.progress(0)
    heatmap_log = st.empty()
    start_time = time.time()
    unique_dates = sorted(df2["entry_date"].dt.normalize().unique())
    total_dates = len(unique_dates)
    for i, _date in enumerate(unique_dates, 1):
        _ = df2[(df2["entry_date"] <= _date) & (df2["exit_date"] >= _date)]
        log_with_progress(
            i,
            total_dates,
            start_time,
            prefix="heatmap",
            batch=10,
            log_area=heatmap_log,
            progress_bar=progress_heatmap,
            unit="days",
        )
        time.sleep(0.005)
    heatmap_log.text(i18n.tr("drawing heatmap..."))
    holding_matrix = generate_holding_matrix(df2)
    display_holding_heatmap(
        holding_matrix, title=f"{system_name} - {i18n.tr('holdings heatmap (by day)')}"
    )
    heatmap_log.success(tr("heatmap generated"))
    # unique-key download button to avoid DuplicateElementId across tabs/systems
    csv_bytes = holding_matrix.to_csv().encode("utf-8")
    if getattr(getattr(_APP_SETTINGS, "ui", None), "show_download_buttons", True):
        st.download_button(
            label=(i18n.tr("download holdings csv")),
            data=csv_bytes,
            file_name=f"holding_status_{system_name}.csv",
            mime="text/csv",
            key=f"{system_name}_{key_context}_download_holding_csv",
        )
    try:
        progress_heatmap.empty()
    except Exception:
        pass


@overload
def show_signal_trade_summary(
    source_df: pd.DataFrame | Dict[str, pd.DataFrame] | None,
    trades_df: pd.DataFrame | None,
    system_name: str,
    display_name: str | None = None,
) -> pd.DataFrame: ...


def show_signal_trade_summary(
    source_df, trades_df, system_name: str, display_name: str | None = None
):
    if system_name == "System1" and isinstance(source_df, pd.DataFrame):
        signal_counts = source_df["symbol"].value_counts().reset_index()
        signal_counts.columns = ["symbol", "Signal_Count"]
    else:
        signal_counts = {
            sym: int(df.get("setup", pd.Series(dtype=int)).sum())
            for sym, df in (source_df or {}).items()
        }
        signal_counts = pd.DataFrame(
            signal_counts.items(), columns=["symbol", "Signal_Count"]
        )

    if trades_df is not None and not trades_df.empty:
        trade_counts = (
            trades_df.groupby("symbol").size().reset_index(name="Trade_Count")
        )
    else:
        trade_counts = pd.DataFrame(columns=["symbol", "Trade_Count"])

    summary_df = pd.merge(signal_counts, trade_counts, on="symbol", how="outer").fillna(
        0
    )
    summary_df["Signal_Count"] = summary_df["Signal_Count"].astype(int)
    summary_df["Trade_Count"] = summary_df["Trade_Count"].astype(int)

    label = f"{display_name or system_name} signal発生件数 / トレード発生件数"
    with st.expander(label, expanded=False):
        st.dataframe(summary_df.sort_values("Signal_Count", ascending=False))
    return summary_df


def display_roc200_ranking(
    ranking_df: pd.DataFrame,
    years: int = 5,
    top_n: int = 10,
    title: str = "System1 ROC200ランキング",
):
    if ranking_df is None or ranking_df.empty:
        st.info(tr("ランキングデータがありません"))
        return
    df = ranking_df.copy()
    df["Date"] = (
        pd.to_datetime(df["Date"]) if "Date" in df.columns else pd.to_datetime(df.index)
    )
    df = df.reset_index(drop=True)
    if "ROC200_Rank" not in df.columns and "ROC200" in df.columns:
        df["ROC200_Rank"] = df.groupby("Date")["ROC200"].rank(
            ascending=False, method="first"
        )
    if years:
        start_date = pd.Timestamp.now() - pd.DateOffset(years=years)
        df = df[df["Date"] >= start_date]
    if top_n:
        df = df.groupby("Date").head(top_n)
    df = df.sort_values(["Date", "ROC200_Rank"], ascending=[True, True])
    with st.expander(f"{title} (直近{years}年 / 上位{top_n}件)", expanded=False):
        st.dataframe(
            df.reset_index(drop=True)[["Date", "ROC200_Rank", "symbol"]],
            hide_index=False,
        )


# ------------------------------
# Save helpers
# ------------------------------
@overload
def save_signal_and_trade_logs(
    signal_counts_df: pd.DataFrame | None,
    results: pd.DataFrame | list[dict[str, Any]] | None,
    system_name: str,
    capital: float,
) -> None: ...


def save_signal_and_trade_logs(signal_counts_df, results, system_name, capital):
    today_str = pd.Timestamp.today().strftime("%Y-%m-%d_%H%M")
    save_dir = "results_csv"
    os.makedirs(save_dir, exist_ok=True)
    sig_dir = os.path.join(save_dir, "signals")
    os.makedirs(sig_dir, exist_ok=True)
    trade_dir = os.path.join(save_dir, "trades")
    os.makedirs(trade_dir, exist_ok=True)

    if signal_counts_df is not None and not signal_counts_df.empty:
        signal_path = os.path.join(
            sig_dir, f"{system_name}_signals_{today_str}_{int(capital)}.csv"
        )
        signal_counts_df.to_csv(signal_path, index=False)
        st.write(tr("シグナルを保存しました: {signal_path}", signal_path=signal_path))
        # 即時ダウンロード
        # システム別CSVダウンロードボタンを削除（⑤の要求）
        # st.download_button(
        #     label=f"{system_name} シグナルCSVをダウンロード",
        #     data=signal_counts_df.to_csv(index=False).encode("utf-8"),
        #     file_name=f"{system_name}_signals_{today_str}_{int(capital)}.csv",
        #     mime="text/csv",
        #     key=f"{system_name}_download_signals_csv",
        # )

    trades_df = pd.DataFrame(results) if isinstance(results, list) else results
    if trades_df is not None and not trades_df.empty:
        # 画面内プレビュー（呼び出し元でエクスパンダー内にいる想定）
        try:
            preferred_cols = [
                "entry_date",
                "exit_date",
                "symbol",
                "action",
                "price",
                "qty",
                "pnl",
            ]
            cols = [c for c in preferred_cols if c in trades_df.columns]
            st.dataframe(trades_df[cols] if cols else trades_df)
        except Exception:
            pass
        trade_path = os.path.join(
            trade_dir, f"{system_name}_trades_{today_str}_{int(capital)}.csv"
        )
        trades_df.to_csv(trade_path, index=False)
        st.write(tr("トレードを保存しました: {trade_path}", trade_path=trade_path))
        # システム別CSVダウンロードボタンを削除（⑤の要求）
        # st.download_button(
        #     label=f"{system_name} トレードCSVをダウンロード",
        #     data=trades_df.to_csv(index=False).encode("utf-8"),
        #     file_name=f"{system_name}_trades_{today_str}_{int(capital)}.csv",
        #     mime="text/csv",
        #     key=f"{system_name}_download_trades_csv",
        # )


@overload
def save_prepared_data_cache(
    data_dict: Dict[str, pd.DataFrame], system_name: str = "SystemX"
) -> None: ...


def save_prepared_data_cache(
    data_dict: Dict[str, pd.DataFrame], system_name: str = "SystemX"
):
    st.info(tr("{system_name} の日次データを保存中...", system_name=system_name))
    if not data_dict:
        st.warning(tr("保存するデータがありません"))
        return
    total = len(data_dict)
    progress_bar = st.progress(0)
    for i, (sym, df) in enumerate(data_dict.items(), 1):
        path = os.path.join("data_cache", f"{safe_filename(sym)}.csv")
        try:
            df.to_csv(path)
        except Exception:
            # 書き込み失敗してもループ継続
            pass
        progress_bar.progress(0 if total == 0 else i / total)
    st.write(tr("{total}件のファイルを保存しました", total=total))
    try:
        progress_bar.empty()
    except Exception:
        pass


@overload
def save_prepared_data_cache(
    data_dict: Dict[str, pd.DataFrame], system_name: str = "SystemX"
) -> None: ...


def save_prepared_data_cache(data_dict, system_name: str = "SystemX"):
    st.info(tr("{system_name} の日次データを保存中...", system_name=system_name))
    if not data_dict:
        st.warning(tr("保存するデータがありません"))
        return
    total = len(data_dict)
    progress_bar = st.progress(0)
    for i, (sym, df) in enumerate(data_dict.items(), 1):
        path = os.path.join("data_cache", f"{safe_filename(sym)}.csv")
        df.to_csv(path)
        progress_bar.progress(0 if total == 0 else i / total)
    st.write(tr("{total}件のファイルを保存しました", total=total))
    try:
        progress_bar.empty()
    except Exception:
        pass
