"""
共通UIコンポーネント（UTF-8・日本語対応）。
既存の公開API（関数名・戻り値）は維持しつつ、各フェーズ（データ取得/インジ計算/候補抽出/バックテスト）で
UIManager（任意）に進捗とログを出力できるようにしている。
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib import font_manager as _font_manager

from common.cache_format import round_dataframe
from common.utils import get_cached_data, safe_filename
from config.settings import get_settings

try:
    # 設定からUIフラグを参照（失敗時はデフォルト動作にフォールバック）
    from config.settings import get_settings

    _APP_SETTINGS = get_settings(create_dirs=True)
except Exception:
    _APP_SETTINGS = None
import common.i18n as i18n
from common.cache_manager import base_cache_path, load_base_cache
from common.holding_tracker import display_holding_heatmap, generate_holding_matrix
from core.system1 import generate_roc200_ranking_system1
from scripts.tickers_loader import get_all_tickers

# 互換用エイリアス（既存コードの tr(...) 呼び出しを維持）
tr = i18n.tr


# ------------------------------
# Type overloads for static checkers
# ------------------------------
# overloads removed - keep concrete implementations only


# overloads removed - keep concrete implementations only


# overloads removed - keep concrete implementations only


# overloads removed - keep concrete implementations only


# overloads removed - keep concrete implementations only


# overloads removed - keep concrete implementations only


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
        df = load_base_cache(
            symbol, rebuild_if_missing=True, prefer_precomputed_indicators=True
        )
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


def fetch_data(
    symbols, max_workers: int = 8, ui_manager=None
) -> dict[str, pd.DataFrame]:
    data_dict: dict[str, pd.DataFrame] = {}
    total = len(symbols)
    # UIManagerのフェーズ（fetch）があればそこへ出力
    phase = ui_manager.phase("fetch") if ui_manager else None
    if phase:
        progress_bar = phase.progress_bar
        log_area = phase.log_area
        # フェーズ配下に「no data」用の別スロットを確保（未作成なら生成）
        no_data_area = phase.no_data_area if hasattr(phase, "no_data_area") else None
        if no_data_area is None:
            try:
                no_data_area = phase.container.empty()
            except Exception:
                no_data_area = st.empty()
            try:
                phase.no_data_area = no_data_area
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
        # cast to Any to satisfy narrow type checkers used in the repo
        call_kwargs["use_process_pool"] = cast(Any, True)

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
        fund_log_area = (
            bt_phase.fund_log_area
            if hasattr(bt_phase, "fund_log_area")
            else bt_phase.container.empty()
        )
        try:
            bt_phase.fund_log_area = fund_log_area
        except Exception:
            pass
    else:
        st.info(tr("backtest: running..."))
        progress = st.progress(0)
        log_area = st.empty()
        fund_log_area = st.empty()
    # debug_area is not used directly here; keep UI placeholder via st.empty() when needed
    _ = st.empty()
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
                rerun = getattr(st, "experimental_rerun", None)
                if callable(rerun):
                    try:
                        rerun()
                    except Exception:
                        pass

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

    # 通常株のみフィルタリングオプション
    use_common_stocks_only = st.checkbox(
        tr("普通株のみ（約6,200銘柄、ETF・優先株除外）"),
        value=False,
        key=f"{system_name}_common_only",
    )

    _init_cap = int(st.session_state.get(key_capital_saved, 1000))
    capital = st.number_input(
        tr("capital (USD)"),
        min_value=1000,
        value=_init_cap,
        step=100,
        key=f"{system_name}_capital",
    )

    # ティッカーリスト取得（フィルタリングオプション考慮）
    if use_common_stocks_only:
        try:
            from scripts.tickers_loader import get_common_stocks_only

            all_tickers = get_common_stocks_only()
            st.info(f"通常株フィルタ適用: {len(all_tickers)}銘柄")
        except ImportError as e:
            st.error(f"通常株フィルタリング機能のインポートに失敗: {e}")
            all_tickers = get_all_tickers()
        except Exception as e:
            st.warning(f"通常株フィルタリング失敗: {e}")
            st.info("フォールバック: 全銘柄を使用します")
            all_tickers = get_all_tickers()
    else:
        all_tickers = get_all_tickers()

    max_allowed = len(all_tickers)
    
    # System6用のデフォルト値を特別に設定
    if system_name == "System6":
        default_value = min(500, max_allowed)   # System6は500がデフォルト（保守的）
    else:
        default_value = min(10, max_allowed)    # 他のシステムは10がデフォルト

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

            if not (_os.getenv("DISCORD_WEBHOOK_URL") or _os.getenv("SLACK_BOT_TOKEN")):
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
    try:
        settings = get_settings(create_dirs=True)
        round_dec = getattr(settings.cache, "round_decimals", None)
    except Exception:
        round_dec = None
    try:
        hm_out = round_dataframe(holding_matrix, round_dec)
    except Exception:
        hm_out = holding_matrix
    csv_bytes = hm_out.to_csv().encode("utf-8")
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


def extract_zero_reason_from_logs(logs: list[str] | None) -> str | None:
    """ログ配列から候補0件の理由を抽出して返す（見つからなければ None）。

    対応パターン:
    - "候補0件理由: ..."
    - "セットアップ不成立: ..."
    """
    if not logs:
        return None
    import re as _re

    for ln in reversed(list(logs)):
        if not ln:
            continue
        m = _re.search(r"候補0件理由[:：]\s*(.+)$", ln)
        if m:
            return m.group(1).strip()
        m2 = _re.search(r"セットアップ不成立[:：]\s*(.+)$", ln)
        if m2:
            return m2.group(1).strip()
    return None


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
        try:
            settings = get_settings(create_dirs=True)
            round_dec = getattr(settings.cache, "round_decimals", None)
        except Exception:
            round_dec = None
        try:
            out_df = round_dataframe(signal_counts_df, round_dec)
        except Exception:
            out_df = signal_counts_df
        out_df.to_csv(signal_path, index=False)
        st.write(tr("シグナルを保存しました: {signal_path}", signal_path=signal_path))
        # 即時ダウンロード
        st.download_button(
            label=f"{system_name} シグナルCSVをダウンロード",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{system_name}_signals_{today_str}_{int(capital)}.csv",
            mime="text/csv",
            key=f"{system_name}_download_signals_csv",
        )

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
        try:
            try:
                settings = get_settings(create_dirs=True)
                round_dec = getattr(settings.cache, "round_decimals", None)
            except Exception:
                round_dec = None
            try:
                out_trades = round_dataframe(trades_df, round_dec)
            except Exception:
                out_trades = trades_df
            out_trades.to_csv(trade_path, index=False)
            st.write(tr("トレードを保存しました: {trade_path}", trade_path=trade_path))
            # 即時ダウンロード
            st.download_button(
                label=f"{system_name} トレードCSVをダウンロード",
                data=out_trades.to_csv(index=False).encode("utf-8"),
                file_name=f"{system_name}_trades_{today_str}_{int(capital)}.csv",
                mime="text/csv",
                key=f"{system_name}_download_trades_csv",
            )
        except Exception:
            # 書き込み/ダウンロード失敗しても処理を継続
            pass


def save_prepared_data_cache(
    data_dict: dict[str, pd.DataFrame], system_name: str = "SystemX"
):
    """Save prepared per-symbol CSVs under `data_cache/` (Streamlit UI helper).

    This implementation attempts to round numeric columns according to
    `settings.cache.round_decimals` before writing. Failures fall back to
    writing the unrounded DataFrame.
    """
    st.info(tr("{system_name} の日次データを保存中...", system_name=system_name))
    if not data_dict:
        st.warning(tr("保存するデータがありません"))
        return
    total = len(data_dict)
    progress_bar = st.progress(0)
    for i, (sym, df) in enumerate(data_dict.items(), 1):
        path = os.path.join("data_cache", f"{safe_filename(sym)}.csv")
        try:
            try:
                settings = get_settings(create_dirs=True)
                round_dec = getattr(settings.cache, "round_decimals", None)
            except Exception:
                round_dec = None
            try:
                out_df = round_dataframe(df, round_dec)
            except Exception:
                out_df = df
            try:
                out_df.to_csv(path)
            except Exception:
                df.to_csv(path)
        except Exception:
            # Ignore failures and continue
            pass
        progress_bar.progress(0 if total == 0 else i / total)
    st.write(tr("{total}件のファイルを保存しました", total=total))
    try:
        progress_bar.empty()
    except Exception:
        pass


def display_cache_health_dashboard() -> None:
    """
    rolling cacheの健全性を表示するダッシュボードコンポーネント。
    """
    st.subheader("🩺 Cache Health Dashboard")

    from common.cache_manager import CacheManager
    from config.settings import get_settings

    try:
        settings = get_settings(create_dirs=True)
        cache_manager = CacheManager(settings)

        # 健全性サマリー取得
        health_summary = cache_manager.get_rolling_health_summary()

        # メタファイル状況
        st.write("### 📋 メタファイル状況")
        col1, col2 = st.columns(2)

        with col1:
            meta_status = "✅ 存在" if health_summary["meta_exists"] else "❌ 不在"
            st.metric("メタファイル", meta_status)

        with col2:
            st.metric("Rolling Files", f"{health_summary['rolling_files_count']}個")

        # SPY アンカー状況
        st.write("### ⚓ SPY アンカー状況")
        anchor_status = health_summary["anchor_symbol_status"]
        col1, col2, col3 = st.columns(3)

        with col1:
            anchor_exists = "✅ 存在" if anchor_status["exists"] else "❌ 不在"
            st.metric("SPY存在", anchor_exists)

        with col2:
            st.metric("データ行数", f"{anchor_status['rows']:,}")

        with col3:
            target_status = "✅ 十分" if anchor_status["meets_target"] else "⚠️ 不足"
            st.metric("目標達成", target_status)

        # 目標データ長
        st.write("### 🎯 目標設定")
        st.metric("目標データ長", f"{health_summary['target_length']}日")

        # メタファイル内容詳細
        if health_summary["meta_exists"] and health_summary["meta_content"]:
            st.write("### 📄 メタファイル詳細")
            st.json(health_summary["meta_content"])

        # アクションボタン
        st.write("### ⚡ アクション")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Rolling Cache 分析実行"):
                with st.spinner("分析中..."):
                    analysis_result = cache_manager.analyze_rolling_gaps()
                    _display_cache_analysis_results(analysis_result)

        with col2:
            if st.button("🧹 Rolling Cache Prune実行"):
                with st.spinner("Prune実行中..."):
                    prune_result = cache_manager.prune_rolling_if_needed()
                    st.success(
                        f"✅ Prune完了: {prune_result['pruned_files']}ファイル処理"
                    )

    except Exception as e:
        st.error(f"Cache health dashboard エラー: {str(e)}")
        logging.error(f"Cache health dashboard error: {e}")


def _display_cache_analysis_results(analysis_result: dict) -> None:
    """Cache分析結果を表示する内部ヘルパー関数。"""
    st.write("### 📊 Rolling Cache 分析結果")

    # サマリーメトリクス
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("総シンボル数", analysis_result["total_symbols"])

    with col2:
        st.metric("整備済み", analysis_result["available_in_rolling"])

    with col3:
        st.metric("未整備", analysis_result["missing_from_rolling"])

    with col4:
        coverage = analysis_result["coverage_percentage"]
        st.metric("カバレッジ", f"{coverage:.1f}%")

    # カバレッジ状況の視覚化
    if coverage >= 90:
        st.success("🎉 Rolling cache整備状況は良好です")
    elif coverage >= 70:
        st.warning("⚠️ Rolling cache整備率の改善を推奨します")
    else:
        st.error("🚨 Rolling cache整備が不十分です")

    # 未整備シンボルの表示
    missing_symbols = analysis_result.get("missing_symbols", [])
    if missing_symbols:
        st.write("### ❌ 未整備シンボル")

        if len(missing_symbols) <= 20:
            # 20個以下なら全て表示
            st.write(", ".join(missing_symbols))
        else:
            # 多い場合は展開可能にする
            with st.expander(f"未整備シンボル一覧 ({len(missing_symbols)}個)"):
                # 10個ずつ区切って表示
                for i in range(0, len(missing_symbols), 10):
                    chunk = missing_symbols[i : i + 10]
                    st.write(", ".join(chunk))


def display_system_cache_coverage() -> None:
    """
    システム別のcache coverage状況を表示するコンポーネント。
    """
    st.subheader("🎯 System別 Cache Coverage")

    from common.cache_manager import CacheManager
    from common.system_groups import analyze_system_symbols_coverage
    from config.settings import get_settings
    from scripts.tickers_loader import get_all_tickers

    try:
        settings = get_settings(create_dirs=True)
        cache_manager = CacheManager(settings)

        # 全ティッカーから各システム用のシンボルマップを構築
        # 実装では各システムに固有のフィルタリングロジックが必要だが、
        # ここでは簡略化して全シンボルを使用
        all_tickers = get_all_tickers()
        system_symbols_map = {}
        for system_num in range(1, 8):
            # 実際の実装では、各システム固有のフィルタリング条件を適用
            system_symbols_map[f"system{system_num}"] = all_tickers[:500]  # 簡略化

        # 全体のcache分析
        overall_analysis = cache_manager.analyze_rolling_gaps()

        # システム別カバレッジ分析
        coverage_analysis = analyze_system_symbols_coverage(
            system_symbols_map, overall_analysis
        )

        # グループ別サマリー表示
        st.write("### 📈 グループ別サマリー")
        group_data = coverage_analysis["by_group"]

        for group_name in ["long", "short"]:
            if group_name in group_data:
                group_stats = group_data[group_name]
                col1, col2, col3, col4 = st.columns(4)

                group_display = (
                    "Long Systems" if group_name == "long" else "Short Systems"
                )
                st.write(f"**{group_display}**")

                with col1:
                    st.metric("総シンボル", group_stats["total_symbols"])

                with col2:
                    st.metric("整備済み", group_stats["available"])

                with col3:
                    st.metric("未整備", group_stats["missing"])

                with col4:
                    coverage = group_stats["coverage_percentage"]
                    status = group_stats["status"]
                    st.metric("状況", f"{status} {coverage:.1f}%")

        # システム別詳細
        st.write("### 🔍 システム別詳細")
        system_data = coverage_analysis["by_system"]

        # データフレーム形式で表示
        df_data = []
        for system_name in [f"system{i}" for i in range(1, 8)]:
            if system_name in system_data:
                stats = system_data[system_name]
                df_data.append(
                    {
                        "システム": system_name.upper(),
                        "総シンボル": stats["total_symbols"],
                        "整備済み": stats["available"],
                        "未整備": stats["missing"],
                        "カバレッジ": f"{stats['coverage_percentage']:.1f}%",
                        "状況": stats["status"],
                    }
                )

        if df_data:
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)

        # 詳細分析用の展開セクション
        with st.expander("📋 詳細分析結果"):
            st.json(coverage_analysis)

    except Exception as e:
        st.error(f"System cache coverage エラー: {str(e)}")
        logging.error(f"System cache coverage error: {e}")


def display_cache_recommendations(analysis_result: dict) -> None:
    """
    Cache分析結果に基づく推奨アクションを表示する。
    """
    from common.system_groups import format_cache_coverage_report

    # 分析結果をフォーマット
    report = format_cache_coverage_report(
        analysis_result["total_symbols"],
        analysis_result["available_in_rolling"],
        analysis_result["missing_from_rolling"],
        analysis_result["coverage_percentage"],
        analysis_result.get("missing_symbols", []),
    )

    # ステータス表示
    st.write(f"### {report['status']} 総合評価")
    st.write(f"**優先度**: {report['priority']}")

    # サマリー情報
    summary = report["summary"]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("総数", summary["total"])
    with col2:
        st.metric("整備済み", summary["available"])
    with col3:
        st.metric("未整備", summary["missing"])
    with col4:
        st.metric("カバレッジ", summary["coverage"])

    # 推奨アクション
    st.write("### 💡 推奨アクション")
    for recommendation in report["recommendations"]:
        st.write(f"- {recommendation}")

    # 未整備シンボルプレビュー
    if report["missing_symbols_preview"]:
        st.write("### 🔍 未整備シンボル（プレビュー）")
        for symbol in report["missing_symbols_preview"]:
            st.write(f"- {symbol}")
