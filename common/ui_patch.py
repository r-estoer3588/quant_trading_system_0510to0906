"""
UI互換パッチローダー。
common.ui_components の関数を共通実装へ委譲するために動的差し替えする。
アプリ起動時に `import common.ui_patch` するだけで有効。
"""

from __future__ import annotations

import logging

try:
    import pandas as pd
    import streamlit as st

    from common.logging_utils import log_with_progress as _core_log_with_progress
    from common.performance_summary import summarize as _summarize_perf
    import common.ui_components as _ui
    from config.settings import get_settings
except Exception:  # pragma: no cover
    _core_log_with_progress = None
    _summarize_perf = None
    _ui = None
    pd = None


def _patched_log_with_progress(
    i,
    total,
    start_time,
    prefix="処理",
    batch=50,
    log_area=None,
    progress_bar=None,
    extra_msg=None,
    unit="件",
    **kwargs,
):
    if _core_log_with_progress is None:
        # 旧実装にフォールバック（安全側）
        import time as _t

        if i % batch == 0 or i == total:
            elapsed = _t.time() - start_time
            remain = (elapsed / i) * (total - i) if i > 0 else 0
            msg = (
                f"{prefix}: {i}/{total} {unit} 完了"
                f"| 経過: {int(elapsed // 60)}分{int(elapsed % 60)}秒"
                f"/ 残り: 約{int(remain // 60)}分{int(remain % 60)}秒"
            )
            if extra_msg:
                msg += f"\n{extra_msg}"
            if log_area:
                log_area.text(msg)
            if progress_bar:
                progress_bar.progress(i / total if total else 0)
        return

    _core_log_with_progress(
        i,
        total,
        start_time,
        prefix=prefix,
        batch=batch,
        log_func=(lambda m: log_area.text(m)) if log_area else None,
        progress_func=(lambda v: progress_bar.progress(v)) if progress_bar else None,
        extra_msg=extra_msg,
        unit=unit,
        **{k: v for k, v in kwargs.items() if k in {"silent"}},
    )
    # 追加: コンソール(標準出力)にも常に進捗を出す
    try:
        if i % batch == 0 or i == total:
            elapsed = __import__("time").time() - start_time
            remain = (elapsed / i) * (total - i) if i > 0 else 0
            msg = (
                f"{prefix}: {i}/{total} {unit} 完了 | "
                f"経過: {int(elapsed // 60)}分{int(elapsed % 60)}秒 / "
                f"残り: 約{int(remain // 60)}分{int(remain % 60)}秒"
            )
            if extra_msg:
                msg += f"\n{extra_msg}"
            logging.getLogger().info(msg)
    except Exception:
        pass


def _patched_summarize_results(results_df, capital):
    if _summarize_perf is None or results_df is None or results_df.empty:
        return {}, results_df
    s, df2 = _summarize_perf(results_df, capital)
    return s.to_dict(), df2


if _ui is not None:  # 動的差し替え（存在チェックで安全化）
    try:
        if hasattr(_ui, "log_with_progress"):
            _ui.log_with_progress = _patched_log_with_progress  # type: ignore[attr-defined]
        else:
            # 後方互換: 存在しない場合も追加（旧バージョン想定）
            _ui.log_with_progress = _patched_log_with_progress  # type: ignore[attr-defined]
        if hasattr(_ui, "summarize_results"):
            _ui.summarize_results = _patched_summarize_results  # type: ignore[attr-defined]
        else:
            _ui.summarize_results = _patched_summarize_results  # type: ignore[attr-defined]
    except Exception:
        pass

# ダウンロードボタンの一括無効化（自動保存がある場合に隠す）
try:
    _settings = get_settings(create_dirs=True) if "get_settings" in globals() else None
    if _settings is not None and "st" in globals() and st is not None:
        _ui_cfg = getattr(_settings, "ui", None)

        # 元の download_button を退避
        if not hasattr(st, "_orig_download_button") and hasattr(st, "download_button"):
            st._orig_download_button = st.download_button

        def _patched_download_button(*args, **kwargs):  # noqa: D401
            """一部 CSV ダウンロードを非表示にし、その他は設定に従う。"""
            fname = kwargs.get("file_name")
            if fname is None and len(args) >= 3:
                fname = args[2]
            try:
                # シグナル/トレードの CSV は常に非表示（自動保存のため）
                if isinstance(fname, str) and (
                    "_signals_" in fname or "_trades_" in fname
                ):
                    return False
            except Exception:
                pass

            # それ以外は設定のフラグに従う
            if not getattr(_ui_cfg, "show_download_buttons", True):
                return False
            try:
                return st._orig_download_button(*args, **kwargs)
            except Exception:
                return False

    try:
        if st is not None:
            st.download_button = _patched_download_button  # type: ignore[attr-defined]
    except Exception:
        pass
except Exception:
    # 失敗時は従来動作のまま
    pass

# show_results を上部統一レイアウトに差し替え
try:
    import pandas as _pd
    import streamlit as _st

    from common.i18n import tr as _tr
    import common.ui_components as _ui_mod

    _plt = None  # 遅延インポート: 実際にグラフ描画が必要になるまで import しない

    def _show_results_patched(
        results_df, capital, system_name: str = "SystemX", *, key_context: str = "main"
    ):
        if results_df is None or getattr(results_df, "empty", True):
            _st.info(_tr("no trades"))
            return

        # 概要と拡張DF
        try:
            summary, df2 = _summarize_perf(results_df, float(capital))
            d = summary.to_dict()
        except Exception:
            d, df2 = {"trades": 0, "total_return": 0.0, "win_rate": 0.0}, results_df

        # 最大DD（負値）とピーク資産比の%を計算
        try:
            dd_value = float(df2["drawdown"].min())
            dd_pct = float(
                (df2["drawdown"] / (float(capital) + df2["cum_max"])).min() * 100
            )
        except Exception:
            dd_value, dd_pct = 0.0, 0.0

        # 上部メトリクス（統一）
        c1, c2, c3, c4 = _st.columns(4)
        c1.metric(_tr("trades"), d.get("trades"))
        c2.metric(_tr("total pnl"), f"{d.get('total_return', 0):.2f}")
        c3.metric(_tr("win rate (%)"), f"{d.get('win_rate', 0):.2f}")
        c4.metric(_tr("max drawdown"), f"{dd_value:.2f}", f"{dd_pct:.2f}%")

        # 結果テーブル
        _st.subheader(_tr("results"))
        _st.dataframe(results_df)

        # 累積PnL + Drawdown プロット
        try:
            if "cumulative_pnl" in df2.columns:
                if _plt is None:
                    try:
                        import matplotlib.pyplot as _plt  # type: ignore
                    except Exception:  # pragma: no cover
                        _plt = None
                if _plt is not None:
                    fig = _plt.figure(figsize=(10, 4))
                    ax = fig.add_subplot(111)
                    ax.plot(df2["exit_date"], df2["cumulative_pnl"], label="CumPnL")
                    if "cum_max" in df2.columns:
                        _dd = df2["cumulative_pnl"] - df2["cum_max"]
                        ax.plot(
                            df2["exit_date"],
                            _dd,
                            color="red",
                            linewidth=1.2,
                            label="Drawdown",
                        )
                    ax.legend(loc="upper left")
                    ax.grid(alpha=0.3)
                    _st.pyplot(fig)
        except Exception:
            pass

        # 年次サマリー（%表記）
        try:
            equity = float(capital) + df2["cumulative_pnl"].astype(float)
            equity.index = _pd.to_datetime(df2["exit_date"])
            daily = equity.resample("D").last().ffill()
            ys = daily.resample("YE").first()
            ye = daily.resample("YE").last()
            yearly_df = _pd.DataFrame(
                {
                    "年": ye.index.year,
                    "損益": (ye - ys).round(2).values,
                    "リターン(%)": ((ye / ys - 1) * 100).values,
                }
            )
            _st.subheader(_tr("yearly summary"))
            _st.dataframe(
                yearly_df.style.format({"損益": "{:.2f}", "リターン(%)": "{:.1f}%"})
            )
            # 月次サマリー
            ms = daily.resample("ME").first()
            me = daily.resample("ME").last()
            monthly_df = _pd.DataFrame(
                {
                    "月": me.index.strftime("%Y-%m"),
                    "損益": (me - ms).round(2).values,
                    "リターン(%)": ((me / ms - 1) * 100).values,
                }
            )
            _st.subheader(_tr("monthly summary"))
            _st.dataframe(
                monthly_df.style.format({"損益": "{:.2f}", "リターン(%)": "{:.1f}%"})
            )
        except Exception:
            pass

        # 保有ヒートマップ（従来ヘルパー）
        try:
            matrix = _ui_mod.generate_holding_matrix(df2)
            _ui_mod.display_holding_heatmap(
                matrix, title=f"{system_name} - " + _tr("holdings heatmap (by day)")
            )
        except Exception:
            pass

    _ui_mod.show_results = _show_results_patched
except Exception:
    # 差し替え不能時は従来表示
    pass
