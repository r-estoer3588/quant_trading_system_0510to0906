from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st

# プロジェクトルート（apps/systems/ から2階層上）をパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import common.ui_patch  # noqa: F401
from common.i18n import language_selector, load_translations_from_dir, tr
from common.logging_utils import log_with_progress
from common.notifier import Notifier, get_notifiers_from_env, now_jst_str
from common.performance_summary import summarize as summarize_perf
from common.price_chart import save_price_chart
from common.ui_components import (
    run_backtest_app,
    save_signal_and_trade_logs,
    show_signal_trade_summary,
)
from common.ui_manager import UIManager
from strategies.system6_strategy import System6Strategy

# 翻訳辞書ロード + 言語選択
load_translations_from_dir(Path(__file__).parent / "translations")
if not st.session_state.get("_integrated_ui", False):
    language_selector()

SYSTEM_NAME = "System6"
DISPLAY_NAME = "システム6"

strategy: System6Strategy = System6Strategy()
notifiers: list[Notifier] = get_notifiers_from_env()


def run_system6_historical_analysis(data_dict: dict) -> pd.DataFrame | None:
    """System6セットアップ条件の過去発生状況を分析する"""
    try:
        analysis_data = []
        for symbol, df in list(data_dict.items())[:20]:  # 最初の20銘柄のみ分析
            if df is None or df.empty or len(df) < 50:
                continue

            # System6の条件をチェック
            filter_ok = (df["Close"] >= 5.0) & (
                df.get("dollarvolume50", df.get("DollarVolume50", 0)) > 10_000_000
            )
            setup_ok = (
                filter_ok
                & (df.get("return_6d", df.get("Return_6D", 0)) > 0.20)
                & df.get("UpTwoDays", df.get("uptwodays", False))
            )

            filter_days = filter_ok.sum() if hasattr(filter_ok, "sum") else 0
            setup_days = setup_ok.sum() if hasattr(setup_ok, "sum") else 0
            total_days = len(df)

            if filter_days > 0:
                setup_rate = (setup_days / filter_days) * 100
                analysis_data.append(
                    {
                        "Symbol": symbol,
                        "Total Days": total_days,
                        "Filter Pass": filter_days,
                        "Setup Pass": setup_days,
                        "Setup Rate (%)": round(setup_rate, 2),
                    }
                )

        if analysis_data:
            analysis_df = pd.DataFrame(analysis_data)
            return analysis_df.head(10)
        return None
    except Exception:
        return None


def display_return6d_ranking(
    candidates_by_date,
    years: int = 5,
    top_n: int = 100,
) -> None:
    if not candidates_by_date:
        st.warning(tr("return_6dランキングデータがありません"))
        return
    rows: list[dict[str, Any]] = []
    total = len(candidates_by_date)
    progress = st.progress(0)
    log_area = st.empty()

    def _progress_update(v: float) -> None:
        progress.progress(v)

    start = time.time()
    for i, (date, cands) in enumerate(candidates_by_date.items(), 1):
        for c in cands:
            rows.append(
                {
                    "Date": date,
                    "symbol": c.get("symbol"),
                    "return_6d": c.get("return_6d"),
                }
            )
        log_with_progress(
            i,
            total,
            start,
            progress_func=_progress_update,
            unit=tr("days"),
        )
    progress.empty()
    log_area.write(tr("return_6dランキング完了"))
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])  # type: ignore[arg-type]
    start_date = pd.Timestamp.now() - pd.DateOffset(years=years)
    df = df[df["Date"] >= start_date]
    df["return_6d_Rank"] = df.groupby("Date")["return_6d"].rank(
        ascending=False,
        method="first",
    )
    df = df.sort_values(["Date", "return_6d_Rank"], ascending=[True, True])
    df = df.groupby("Date").head(top_n)

    # return_6d を % 表示用に変換（内部データは 0.x のまま保持）
    df["return_6d_pct"] = (df["return_6d"] * 100).round(2)

    title = tr(
        "{display_name} return_6d ランキング（直近{years}年 / 上位{top_n}銘柄）",
        display_name=DISPLAY_NAME,
        years=years,
        top_n=top_n,
    )
    with st.expander(title, expanded=False):
        display_df = df.reset_index(drop=True)[
            ["Date", "return_6d_Rank", "symbol", "return_6d_pct"]
        ]
        display_df = display_df.rename(columns={"return_6d_pct": "return_6d (%)"})
        st.dataframe(display_df, hide_index=False)


def run_tab(ui_manager: UIManager | None = None) -> None:
    st.header(
        tr(
            "{display_name} バックテスト（return_6d ランキング）",
            display_name=DISPLAY_NAME,
        )
    )

    # System6の銘柄数についての説明
    st.info(
        tr(
            "💡 **System6 銘柄選択のガイド**\n\n"
            "• **普通株（約6,200銘柄）**: 一般的な普通株式のみ\n"
            "• **制限数**: System6では100-1,000銘柄程度が実用的\n\n"
            "**推奨設定**: 銘柄制限を100-500程度に調整"
        )
    )

    # System6の条件詳細を追加
    with st.expander("🎯 System6の条件詳細", expanded=False):
        st.markdown(
            tr(
                "**System6はショート戦略で、極端な相場状況専用です**\n\n"
                "**フィルター条件（基本要件）:**\n"
                "• 価格 ≥ $5.00\n"
                "• 50日平均ドルボリューム ≥ 1,000万ドル\n\n"
                "**セットアップ条件（非常に厳しい）:**\n"
                "• **6日間リターン ≥ 20%**（最も厳しい条件）\n"
                "• **連続2日上昇**（UpTwoDays = True）\n\n"
                "**統計例:**\n"
                "通常の相場では、フィルター通過銘柄の1%未満がセットアップ条件を満たします。\n"
                "急激な相場変動時にのみトレード機会が発生する設計です。"
            )
        )

    ui_base: UIManager = (
        ui_manager.system(SYSTEM_NAME) if ui_manager else UIManager().system(SYSTEM_NAME)
    )
    # フェーズは作成するが、主に詳細表示用に使用
    _ = ui_base.phase("fetch", title=tr("データ取得"))
    _ = ui_base.phase("indicators", title=tr("インジケーター・フィルター準備"))
    _ = ui_base.phase("candidates", title=tr("System6条件チェック・候補抽出"))

    # フェーズ別の詳細表示エリアを作成
    fetch_detail = st.empty()
    ind_detail = st.empty()
    cand_detail = st.empty()

    # 通知トグルは共通UI(run_backtest_app)内に配置して順序を統一
    notify_key = f"{SYSTEM_NAME}_notify_backtest"
    run_start = time.time()
    _rb = cast(
        tuple[
            pd.DataFrame | None,
            pd.DataFrame | None,
            dict[str, pd.DataFrame] | None,
            float,
            object | None,
        ],
        run_backtest_app(
            strategy,
            system_name=SYSTEM_NAME,
            ui_manager=ui_base,
        ),
    )
    elapsed = time.time() - run_start
    results_df, _, data_dict, capital, candidates_by_date = _rb

    # 詳細な完了メッセージを表示
    if data_dict:
        fetch_detail.success(f"✅ データ取得完了: {len(data_dict)}銘柄のデータを読み込み")
        ind_detail.info(
            "⚡ プリコンピューテッド指標を使用: ATR10, DollarVolume50, Return_6D, UpTwoDays"
        )

        # 候補選定の詳細統計を表示
        if candidates_by_date and isinstance(candidates_by_date, dict):
            total_candidates = sum(len(cands) for cands in candidates_by_date.values())
            trading_days = len(candidates_by_date)
            cand_detail.success(
                f"✅ 候補抽出完了: {total_candidates}件の候補を{trading_days}日間で発見"
            )
        else:
            cand_detail.warning(
                f"⚠️ 候補なし: {len(data_dict)}銘柄中、System6の厳しい条件を満たす銘柄は見つかりませんでした"
            )
    else:
        fetch_detail.error("❌ データ取得に失敗")
        ind_detail.error("❌ インジケーター準備をスキップ")
        cand_detail.error("❌ 候補選定をスキップ")
    if results_df is not None and candidates_by_date is not None:
        display_return6d_ranking(candidates_by_date)
        summary_df = show_signal_trade_summary(
            data_dict,
            results_df,
            SYSTEM_NAME,
            display_name=DISPLAY_NAME,
        )
        with st.expander(tr("取引ログ・保存ファイル"), expanded=False):
            save_signal_and_trade_logs(
                summary_df,
                results_df,
                SYSTEM_NAME,
                capital,
            )
        # Prepared data cache save removed (deprecated feature)
        summary, df2 = summarize_perf(results_df, capital)
        try:
            _max_dd = float(df2["drawdown"].min())
        except Exception:
            _max_dd = float(getattr(summary, "max_drawdown", 0.0))
        try:
            _dd_pct = float((df2["drawdown"] / (float(capital) + df2["cum_max"])).min() * 100)
        except Exception:
            _dd_pct = 0.0
        stats: dict[str, Any] = {
            "総リターン": f"{summary.total_return:.2f}",
            "最大DD": f"{_max_dd:.2f} ({_dd_pct:.2f}%)",
            "Sharpe": f"{summary.sharpe:.2f}",
            "実施日時": now_jst_str(),
            "銘柄数": len(data_dict) if data_dict else 0,
            "開始資金": int(capital),
            "処理時間": f"{elapsed:.2f}s",
        }
        # メトリクスは共通 show_results で統一表示
        pass
        # 年別サマリー（%表記）
        try:
            equity = float(capital) + df2["cumulative_pnl"].astype(float)
            equity.index = pd.to_datetime(df2["exit_date"])  # type: ignore
            daily_eq = equity.resample("D").last().ffill()
            ys = daily_eq.resample("YE").first()
            ye = daily_eq.resample("YE").last()
            yearly_df = pd.DataFrame(
                {
                    "year": pd.to_datetime(ye.index).year,
                    "pnl": (ye - ys).round(2).values,
                    "return_pct": ((ye / ys - 1) * 100).values,
                }
            )
            # 年次サマリーは共通の show_results で統一表示
            _ = yearly_df
        except Exception:
            pass
        ranking: list[str] = (
            [str(s) for s in results_df["symbol"].head(10)]
            if "symbol" in results_df.columns
            else []
        )
        period = ""
        if "entry_date" in results_df.columns and "exit_date" in results_df.columns:
            start = pd.to_datetime(results_df["entry_date"]).min()
            end = pd.to_datetime(results_df["exit_date"]).max()
            period = f"{start:%Y-%m-%d}〜{end:%Y-%m-%d}"
        chart_url = None
        if not results_df.empty and "symbol" in results_df.columns:
            try:
                top_sym = results_df.sort_values("pnl", ascending=False)["symbol"].iloc[0]
                _, chart_url = save_price_chart(str(top_sym), trades=results_df)
            except Exception:
                chart_url = None
        if st.session_state.get(notify_key, False):
            sent = False
            for n in notifiers:
                try:
                    _mention: str | None = (
                        "channel" if getattr(n, "platform", None) == "slack" else None
                    )
                    if hasattr(n, "send_backtest_ex"):
                        n.send_backtest_ex(
                            SYSTEM_NAME.lower(),
                            period,
                            stats,
                            ranking,
                            image_url=chart_url,
                            mention=_mention,
                        )
                    else:
                        n.send_backtest(
                            SYSTEM_NAME.lower(),
                            period,
                            stats,
                            ranking,
                        )
                    sent = True
                except Exception:
                    continue
            if sent:
                st.success(tr("通知を送信しました"))
            else:
                st.warning(tr("通知の送信に失敗しました"))
    else:
        # System6特有の説明メッセージ（候補なしの場合）
        cand_detail.warning("⚠️ トレード候補が見つかりませんでした")

        # 詳細な分析情報を表示
        analysis_container = st.container()
        with analysis_container:
            st.info(
                tr(
                    "🔍 **候補なしの理由分析**\n\n"
                    "System6は極端な相場状況でのみ機能する戦略です。\n"
                    "以下の厳しい条件をすべて満たす必要があります:\n\n"
                    "1. **価格フィルター**: 株価 ≥ $5.00\n"
                    "2. **流動性フィルター**: 50日平均ドルボリューム ≥ 1,000万ドル\n"
                    "3. **モメンタムフィルター**: 6日間リターン ≥ 20% 🔥\n"
                    "4. **連続上昇フィルター**: 連続2日上昇\n\n"
                    "**通常の相場**: フィルター通過銘柄の1%未満がセットアップ条件達成\n"
                    "**急変相場**: 10-20%の銘柄がセットアップ条件達成の可能性"
                )
            )

            # データの詳細情報を表示
            if data_dict:
                st.success(f"✅ {len(data_dict)}銘柄のデータ準備は完了しています")

                # System6の過去の発生状況ボタンを追加
                if st.button(
                    "📊 System6セットアップ条件の過去発生状況を確認", key="system6_historical_check"
                ):
                    with st.spinner("過去データを分析中..."):
                        # セットアップ発生頻度の簡易分析を実行
                        analysis_results = run_system6_historical_analysis(data_dict)
                        if analysis_results is not None and not analysis_results.empty:
                            st.write("**過去のセットアップ発生状況:**")
                            st.dataframe(analysis_results)
            else:
                st.error("❌ データの取得に失敗しました")

        # フォールバック表示（セッション保存から復元）
        prev_res = st.session_state.get(f"{SYSTEM_NAME}_results_df")
        prev_cands = st.session_state.get(f"{SYSTEM_NAME}_candidates_by_date")
        prev_data = st.session_state.get(f"{SYSTEM_NAME}_prepared_dict")
        prev_cap = st.session_state.get(f"{SYSTEM_NAME}_capital_saved")
        if prev_res is not None and prev_cands is not None:
            display_return6d_ranking(prev_cands)
            _ = show_signal_trade_summary(
                prev_data,
                prev_res,
                SYSTEM_NAME,
                display_name=DISPLAY_NAME,
            )
            try:
                from common.ui_components import show_results

                show_results(
                    prev_res,
                    prev_cap or 0.0,
                    SYSTEM_NAME,
                    key_context="prev",
                )
            except Exception:
                pass


if __name__ == "__main__":
    import sys

    if "streamlit" not in sys.argv[0]:
        run_tab()
