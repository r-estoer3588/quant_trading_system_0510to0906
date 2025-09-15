from __future__ import annotations

import logging
from pathlib import Path
import platform
import time
from typing import Any

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Streamlit のコンテキスト外で実行された場合は警告を抑止して終了する
if get_script_run_ctx(suppress_warning=True) is None:
    if __name__ == "__main__":
        print("このスクリプトはStreamlitで実行してください: `streamlit run app_today_signals.py`")
    raise SystemExit

try:
    # Streamlit の実行コンテキスト有無を判定（スレッド外からの UI 呼び出しを防ぐ）
    from streamlit.runtime.scriptrunner import get_script_run_ctx as _st_get_ctx  # type: ignore

    def _has_st_ctx() -> bool:
        try:
            return _st_get_ctx() is not None
        except Exception:
            return False

except Exception:

    def _has_st_ctx() -> bool:  # type: ignore
        return False


# Streamlit checkbox の重複ID対策（key未指定時に自動で一意キーを付与）
try:
    if not hasattr(st, "_orig_checkbox"):
        st._orig_checkbox = st.checkbox  # type: ignore[attr-defined]

        def _unique_checkbox(label, *args, **kwargs):
            if "key" not in kwargs:
                base = f"auto_cb_{abs(hash(str(label))) % 10**8}"
                count_key = f"_{base}_cnt"
                try:
                    cnt = int(st.session_state.get(count_key, 0)) + 1
                except Exception:
                    cnt = 1
                st.session_state[count_key] = cnt
                kwargs["key"] = f"{base}_{cnt}"
            return st._orig_checkbox(  # type: ignore[attr-defined]
                label,
                *args,
                **kwargs,
            )

        st.checkbox = _unique_checkbox  # type: ignore[attr-defined]
except Exception:
    # 失敗しても従来動作のまま進める
    pass

from common import broker_alpaca as ba
from common import universe as univ
from common.alpaca_order import submit_orders_df
from common.data_loader import load_price
from common.notifier import create_notifier
from common.position_age import load_entry_dates
from common.profit_protection import evaluate_positions
from config.settings import get_settings
import scripts.run_all_systems_today as _run_today_mod
from scripts.run_all_systems_today import compute_today_signals

st.set_page_config(page_title="本日のシグナル", layout="wide")
st.title("📈 本日のシグナル（全システム）")

settings = get_settings(create_dirs=True)
notifier = create_notifier(platform="slack", fallback=True)


def _get_today_logger() -> logging.Logger:
    """本日のシグナル実行用ロガー（ファイル: logs/today_signals.log）。

    Streamlit の再実行でもハンドラが重複しないように、既存ハンドラを確認してから追加します。
    """
    logger = logging.getLogger("today_signals")
    logger.setLevel(logging.INFO)
    # ルートロガーへの伝播を止める（重複防止）
    try:
        logger.propagate = False
    except Exception:
        pass
    # ルートロガーへの伝播を止め、コンソール二重出力を防止
    try:
        logger.propagate = False
    except Exception:
        pass
    try:
        log_dir = Path(settings.LOGS_DIR)
    except Exception:
        log_dir = Path("logs")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    log_path = log_dir / "today_signals.log"

    # 既に同じファイルに出力するハンドラがあれば追加しない
    has_handler = False
    for h in list(logger.handlers):
        try:
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(
                log_path
            ):
                has_handler = True
                break
        except Exception:
            continue
    if not has_handler:
        try:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:
            # ログ設定失敗時もUI処理は継続
            pass
    return logger


with st.sidebar:
    st.header("ユニバース")
    universe = univ.load_universe_file()
    if not universe:
        universe = univ.build_universe_from_cache(limit=None)
        univ.save_universe_file(universe)
    all_syms = universe

    # 任意の件数でユニバースを制限するテスト用オプション
    limit_max = max(1, len(all_syms))
    test_limit = st.number_input(
        "銘柄数 (0は全銘柄)",
        min_value=0,
        max_value=limit_max,
        value=0,
        step=1,
    )
    syms = all_syms[: int(test_limit)] if test_limit else all_syms

    st.write(f"銘柄数: {len(syms)}")
    st.write(", ".join(syms[:10]) + (" ..." if len(syms) > 10 else ""))

    st.header("資産")
    # Alpacaから取得した資産のみを使う
    if "today_cap_long" not in st.session_state:
        st.session_state["today_cap_long"] = 0.0
    if "today_cap_short" not in st.session_state:
        st.session_state["today_cap_short"] = 0.0
    # 口座情報の一時保存領域
    st.session_state.setdefault("alpaca_acct_type", None)
    st.session_state.setdefault("alpaca_buying_power", None)
    st.session_state.setdefault("alpaca_cash", None)
    st.session_state.setdefault("alpaca_multiplier", None)
    st.session_state.setdefault("alpaca_shorting_enabled", None)
    st.session_state.setdefault("alpaca_status", None)

    # Alpacaから取得してフォームに反映
    if st.button("🔍 Alpacaから資産取得してフォームに反映"):
        try:
            client = ba.get_client(paper=True)
            acct = client.get_account()
            # 口座情報を保存（表示用）
            try:
                st.session_state["alpaca_acct_type"] = getattr(acct, "account_type", None)
                st.session_state["alpaca_multiplier"] = getattr(acct, "multiplier", None)
                st.session_state["alpaca_shorting_enabled"] = getattr(
                    acct, "shorting_enabled", None
                )
                st.session_state["alpaca_status"] = getattr(acct, "status", None)
            except Exception:
                pass
            bp_raw = getattr(acct, "buying_power", None)
            if bp_raw is None:
                bp_raw = getattr(acct, "cash", None)
            if bp_raw is not None:
                bp = float(bp_raw)
                st.session_state["alpaca_buying_power"] = bp
                try:
                    st.session_state["alpaca_cash"] = float(getattr(acct, "cash", None) or 0.0)
                except Exception:
                    pass
                st.session_state["today_cap_long"] = round(bp / 2.0, 2)
                st.session_state["today_cap_short"] = round(bp / 2.0, 2)
                st.success(
                    f"long資産/short資産を{st.session_state['today_cap_long']}ずつに設定"
                    f"（buying_powerの半分={bp}）"
                )
            else:
                st.warning("Alpaca口座情報: buying_power/cashが取得できません")
        except Exception as e:
            st.error(f"Alpaca資産取得エラー: {e}")

    # 口座情報（表示のみの更新ボタン）
    if st.button("ℹ️ Alpaca口座情報を更新（表示のみ）"):
        try:
            client = ba.get_client(paper=True)
            acct = client.get_account()
            st.session_state["alpaca_acct_type"] = getattr(acct, "account_type", None)
            st.session_state["alpaca_buying_power"] = float(
                getattr(acct, "buying_power", getattr(acct, "cash", 0.0)) or 0.0
            )
            st.session_state["alpaca_cash"] = float(getattr(acct, "cash", 0.0))
            st.session_state["alpaca_multiplier"] = getattr(acct, "multiplier", None)
            st.session_state["alpaca_shorting_enabled"] = getattr(acct, "shorting_enabled", None)
            st.session_state["alpaca_status"] = getattr(acct, "status", None)
            st.success("口座情報を更新しました（表示のみ）")
        except Exception as e:
            st.error(f"口座情報の更新に失敗: {e}")

    # 口座情報の表示（タイプ推定 + Buying Power）
    acct_type_raw = st.session_state.get("alpaca_acct_type")
    multiplier = st.session_state.get("alpaca_multiplier")
    try:
        mult_f = float(multiplier) if multiplier is not None else None
    except Exception:
        mult_f = None
    derived_type = (
        "Margin"
        if (mult_f is not None and mult_f > 1.0)
        else ("Cash" if mult_f is not None else "不明")
    )
    bp_val = st.session_state.get("alpaca_buying_power")
    bp_txt = f"${bp_val:,.2f}" if isinstance(bp_val, (int, float)) else "未取得"
    st.caption("Alpaca口座情報")
    st.write(f"アカウント種別（推定）: {derived_type}  |  Buying Power: {bp_txt}")
    if acct_type_raw is not None or mult_f is not None:
        st.caption(
            f"詳細: account_type={acct_type_raw}, "
            f"multiplier={mult_f if mult_f is not None else '-'}"
        )

    # 資産入力フォーム
    st.session_state["today_cap_long"] = st.number_input(
        "long資産 (USD)",
        min_value=0.0,
        step=100.0,
        value=float(st.session_state["today_cap_long"]),
        key="today_cap_long_input",
    )
    st.session_state["today_cap_short"] = st.number_input(
        "short資産 (USD)",
        min_value=0.0,
        step=100.0,
        value=float(st.session_state["today_cap_short"]),
        key="today_cap_short_input",
    )

    st.header("CSV保存")
    save_csv = st.checkbox("CSVをsignals_dirに保存", value=False)

    # 既定で並列実行をON（Windowsでも有効化）
    is_windows = platform.system().lower().startswith("win")
    run_parallel_default = True
    run_parallel = st.checkbox("並列実行（システム横断）", value=run_parallel_default)

    # 並列実行の詳細設定は削除（初期デフォルト挙動に戻す）
    st.header("Alpaca自動発注")
    paper_mode = st.checkbox("ペーパートレードを使用", value=True)
    retries = st.number_input("リトライ回数", min_value=0, max_value=5, value=2)
    delay = st.number_input("遅延（秒）", min_value=0.0, step=0.5, value=0.5)
    poll_status = st.checkbox("注文状況を10秒ポーリング", value=False)
    do_trade = st.checkbox("Alpacaで自動発注", value=False)
    update_bp_after = st.checkbox("注文後に余力を自動更新", value=True)

    # 注文状況を10秒ポーリングとは？
    # → Alpacaに注文を送信した後、注文IDのステータス（filled, canceled等）を10秒間、
    #    1秒ごとに取得・表示する機能です。
    # これにより、注文が約定したかどうかをリアルタイムで確認できます。

    # キャッシュクリアボタン
    if st.button("キャッシュクリア"):
        st.cache_data.clear()
        st.success("キャッシュをクリアしました")

    if st.button("全注文キャンセル"):
        try:
            client = ba.get_client(paper=paper_mode)
            ba.cancel_all_orders(client)
            st.success("すべての未約定注文をキャンセルしました")
        except Exception as e:
            st.error(f"注文キャンセルエラー: {e}")

    # 表示制御は固定（チェックボックスは廃止）
    st.session_state["ui_vis"] = {
        "overall_progress": True,
        "per_system_progress": True,
        "data_load_progress_lines": True,
        "previous_results": True,
        "system_details": True,
    }

st.subheader("保有ポジションと利益保護判定")
if st.button("🔍 Alpacaから保有ポジション取得"):
    try:
        client = ba.get_client(paper=paper_mode)
        positions = client.get_all_positions()
        st.session_state["positions_df"] = evaluate_positions(positions)
        st.success("ポジションを取得しました")
    except Exception as e:
        st.error(f"ポジション取得エラー: {e}")

if "positions_df" in st.session_state:
    df_pos = st.session_state["positions_df"]
    if df_pos.empty:
        st.info("保有ポジションはありません。")
    else:
        st.dataframe(df_pos, use_container_width=True)

if st.button("▶ 本日のシグナル実行", type="primary"):
    # 実行ごとに per-system ログ表示（text_area）の状態をクリア
    try:
        for i in range(1, 8):
            st.session_state.pop(f"logs_system{i}", None)
    except Exception:
        pass
    # 指標ごとに必要な日数（＋10%余裕）を定義
    indicator_days = {
        "ROC200": int(200 * 1.1),
        "SMA25": int(25 * 1.1),
        "ATR20": int(20 * 1.1),
        "ADX7": int(7 * 1.1),
        "RETURN6": int(6 * 1.1),
        "Drop3D": int(3 * 1.1),
        "Return6D": int(6 * 1.1),
        # 必要に応じて追加
    }

    # 必要な最大日数を算出（全システムで使う指標の最大値）
    max_days = max(indicator_days.values())

    # 開始時刻を記録
    start_time = time.time()
    # CLI バナーはエンジン側で出すため、UI側では出さない
    # 進捗表示用の領域（1行上書き）
    # 大きめ表示のフェーズタイトル
    phase_title_area = st.empty()
    progress_area = st.empty()
    # プログレスバー（表示設定に応じて更新可）
    prog = st.progress(0)
    prog_txt = st.empty()
    # システムごとのプログレスバー（設定でオフなら作成しない）
    ui_vis = st.session_state.get("ui_vis", {})
    if ui_vis.get("per_system_progress", True):
        sys_cols = st.columns(7)
        sys_labels = [f"System{i}" for i in range(1, 8)]
        for i, col in enumerate(sys_cols, start=1):
            col.caption(sys_labels[i - 1])
        sys_bars = {f"system{i}": sys_cols[i - 1].progress(0) for i in range(1, 8)}
        sys_stage_txt = {f"system{i}": sys_cols[i - 1].empty() for i in range(1, 8)}
        # 追加: メトリクス表示用（各システム1行のプレースホルダ）
        sys_metrics_txt = {}
        for i in range(1, 8):
            n = f"system{i}"
            with sys_cols[i - 1]:
                sys_metrics_txt[n] = st.empty()
                # 初期表示（短縮名 + ハイフン）
                sys_metrics_txt[n].text("Tgt -  FILpass -  STUpass -  TRDlist -  Entry -  Exit -")
        # 追加: システム別の補足表示（System2のフィルタ内訳など）
        sys_extra_txt = {f"system{i}": sys_cols[i - 1].empty() for i in range(1, 8)}
        sys_states = {k: 0 for k in sys_bars.keys()}
    else:
        sys_bars = {}
        sys_stage_txt = {}
        sys_metrics_txt = {}
        sys_extra_txt = {}
        sys_states = {}
    # 追加: 全ログを蓄積（system別タブで表示）
    log_lines: list[str] = []
    # 追加: per-system メトリクス保持（filter/setup/cand/entry/exit）
    stage_counts: dict[str, dict[str, int | None]] = {
        f"system{i}": {
            "target": None,
            "filter": None,
            "setup": None,
            "cand": None,
            "entry": None,
            "exit": None,
        }
        for i in range(1, 8)
    }

    def _ui_log(msg: str) -> None:
        try:
            elapsed = max(0, time.time() - start_time)
            m, s = divmod(int(elapsed), 60)
            # 日付と時刻を含めてUIに表示
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{now} | {m}分{s}秒] {msg}"
            log_lines.append(line)
            # 冗長ログをUIでは抑制（ファイルには別途書き出し）
            if _has_st_ctx():
                try:
                    _msg = str(msg)
                    ui_vis2 = st.session_state.get("ui_vis", {})
                    show_overall = bool(ui_vis2.get("overall_progress", True))
                    allow_data_load = bool(ui_vis2.get("data_load_progress_lines", True))
                    # データロード進捗（📦/🧮）はホワイトリストで扱う
                    is_data_load_line = (
                        _msg.startswith("📦 基礎データロード進捗")
                        or _msg.startswith("🧮 指標データロード進捗")
                        or _msg.startswith("📦 基礎データロード完了")
                        or _msg.startswith("🧮 指標データロード完了")
                        or _msg.startswith("🧮 共有指標 前計算")
                    )
                    # 共有指標 前計算ログも表示対象に含める（UIでの可視化を要望）
                    # 不要ログ（UI表示では抑制したいもの）
                    skip_keywords = (
                        "進捗",
                        "インジケーター",
                        "indicator",
                        "indicators",
                        "指標計算",
                        "バッチ時間",
                        "batch time",
                        "next batch size",
                        "候補抽出",
                        "候補日数",
                        "銘柄:",
                        "📊 インジケーター計算",
                        "📊 候補抽出",
                        "⏱️ バッチ時間",
                    )
                    should_show = False
                    if show_overall:
                        if is_data_load_line and allow_data_load:
                            should_show = True
                        elif not any(k in _msg for k in skip_keywords):
                            should_show = True
                    if should_show:
                        progress_area.text(line)
                except Exception:
                    try:
                        progress_area.text(line)
                    except Exception:
                        pass
            # ファイルにもINFOで書き出す
            try:
                _get_today_logger().info(str(msg))
            except Exception:
                pass
        except Exception:
            # 表示に失敗しても処理は継続
            pass

    # フェーズ表示の状態管理（可変辞書でスコープ問題を回避）
    phase_state = {"percent": 0, "label": "対象読み込み"}

    def _map_overall_phase(tag: str) -> str:
        try:
            t = (tag or "").lower()
        except Exception:
            t = ""
        # 全体フェーズの日本語ラベル
        if t in {
            "init",
            "対象読み込み:start",
            "load_basic:start",
            "load_basic",
            "load_indicators",
            "spx",
            "spy",
        }:
            return "対象読み込み"
        if t in {"filter", "フィルター"}:
            return "フィルター"
        if t in {"run_strategies", "setup"} or t.startswith("system"):
            return "セットアップ"
        if t in {"strategies_done", "trade候補", "トレード候補選定"}:
            return "トレード候補選定"
        if t in {"finalize", "done", "エントリー"}:
            return "エントリー"
        # 既定
        return phase_state.get("label", "対象読み込み")

    def _render_phase_title(percent: int, phase_label: str) -> None:
        try:
            # 大きめの文字で表示（H2相当）
            phase_title_area.markdown(f"## 進捗 {percent}%: {phase_label}フェーズ")
        except Exception:
            pass

    def _set_phase_label(phase_label: str) -> None:
        try:
            phase_state["label"] = str(phase_label)
            _render_phase_title(
                int(phase_state.get("percent", 0)),
                phase_state.get("label", "対象読み込み"),
            )
        except Exception:
            pass

    def _ui_progress(done: int, total: int, name: str) -> None:
        try:
            if not _has_st_ctx():
                return
            ui_vis2 = st.session_state.get("ui_vis", {})
            if not bool(ui_vis2.get("overall_progress", True)):
                return
            total = max(1, int(total))
            ratio = min(max(int(done), 0), total) / total
            percent = int(ratio * 100)
            prog.progress(percent)
            # 現在の全体フェーズを更新
            phase_lbl = _map_overall_phase(name)
            # 画面下のテキストも日本語フェーズで統一
            if phase_lbl:
                prog_txt.text(f"進捗 {percent}%: {phase_lbl}")
            # 大きなタイトルも更新
            phase_state["percent"] = percent
            phase_state["label"] = phase_lbl
            _render_phase_title(phase_state["percent"], phase_state["label"])
        except Exception:
            pass

    def _per_system_progress(name: str, phase: str) -> None:
        try:
            if not _has_st_ctx():
                return
            ui_vis2 = st.session_state.get("ui_vis", {})
            if not bool(ui_vis2.get("per_system_progress", True)):
                return
            n = str(name).lower()
            bar = sys_bars.get(n)
            if not bar:
                return
            if phase == "start":
                sys_states[n] = 50
                bar.progress(50)
                sys_stage_txt[n].text("run 50%")
            elif phase == "done":
                sys_states[n] = 100
                bar.progress(100)
                sys_stage_txt[n].text("done 100%")
        except Exception:
            pass

    # 段階進捗（0/25/50/75/100）
    def _per_system_stage(
        name: str,
        v: int,
        filter_cnt: int | None = None,
        setup_cnt: int | None = None,
        cand_cnt: int | None = None,
        final_cnt: int | None = None,
    ) -> None:
        try:
            if not _has_st_ctx():
                return
            ui_vis2 = st.session_state.get("ui_vis", {})
            if not bool(ui_vis2.get("per_system_progress", True)):
                return
            n = str(name).lower()
            bar = sys_bars.get(n)
            if not bar:
                return
            vv = max(0, min(100, int(v)))
            bar.progress(vv)
            sys_states[n] = vv
            # クイックフェーズ表示（常に1行で収まる短縮表記）
            sys_stage_txt[n].text(f"run {vv}%" if vv < 100 else "done 100%")
            # 全体フェーズの見出しを、各システムの段階にあわせて上書き（日本語）
            try:
                if vv <= 0:
                    _set_phase_label("対象準備")
                elif vv < 10:
                    _set_phase_label("対象読み込み")
                elif vv < 30:
                    _set_phase_label("フィルター")
                elif vv < 60:
                    _set_phase_label("セットアップ")
                elif vv < 90:
                    _set_phase_label("トレード候補選定")
                else:
                    _set_phase_label("エントリー")
            except Exception:
                pass
            # メトリクス保持
            sc = stage_counts.setdefault(n, {})
            if filter_cnt is not None:
                if vv == 0:
                    sc["target"] = int(filter_cnt)
                else:
                    sc["filter"] = int(filter_cnt)
            if setup_cnt is not None:
                sc["setup"] = int(setup_cnt)
            if cand_cnt is not None:
                sc["cand"] = int(cand_cnt)
            if final_cnt is not None:
                sc["entry"] = int(final_cnt)
            # 逐次メトリクスを行ごとに個別更新（欠損は「-」）
            try:
                # 名称(半角スペース)銘柄数 で1行表示。最大5桁でも収まる短縮名を使用。
                # 6行表示: 各メトリクスを独立行で表示
                def _v(x):
                    return "-" if (x is None) else str(x)

                lines = [
                    f"Tgt {_v(sc.get('target'))}",
                    f"FILpass {_v(sc.get('filter'))}",
                    f"STUpass {_v(sc.get('setup'))}",
                    f"TRDlist {_v(sc.get('cand'))}",
                    f"Entry {_v(sc.get('entry'))}",
                    f"Exit {_v(sc.get('exit'))}",
                ]
                sys_metrics_txt[n].text("\n".join(lines))
            except Exception:
                # 表示に失敗しても処理は継続
                pass
        except Exception:
            pass

    # 追加: per-system の Exit 件数を UI に即時反映する受け口
    def _per_system_exit(name: str, count: int) -> None:
        try:
            if not _has_st_ctx():
                return
            ui_vis2 = st.session_state.get("ui_vis", {})
            if not bool(ui_vis2.get("per_system_progress", True)):
                return
            n = str(name).lower()
            sc = stage_counts.setdefault(n, {})
            sc["exit"] = int(count)
            # 既存の行を更新
            if n in sys_metrics_txt:
                tgt_txt = "-"
                try:
                    tgt_txt = (
                        str(sc.get("target"))
                        if sc.get("target") is not None
                        else str(sc.get("filter")) if sc.get("setup") is None else "-"
                    )
                except Exception:
                    pass

                def _v2(x):
                    return "-" if (x is None) else str(x)

                lines = [
                    f"Tgt {_v2(tgt_txt)}",
                    f"FILpass {_v2(sc.get('filter'))}",
                    f"STUpass {_v2(sc.get('setup'))}",
                    f"TRDlist {_v2(sc.get('cand'))}",
                    f"Entry {_v2(sc.get('entry'))}",
                    f"Exit {_v2(sc.get('exit'))}",
                ]
                sys_metrics_txt[n].text("\n".join(lines))
        except Exception:
            pass

    # ノート欄は不要のため削除（受け口は未登録）

    # ボタン押下直後の開始ログをUIにも出力（ファイルにも出力されます）
    _ui_log("▶ 本日のシグナル: シグナル検出処理開始")

    # ステージ進捗の受け口を先に登録（スレッドから参照されるため）
    try:
        # orchestrator 側のモジュールグローバルに直接差し込む
        _run_today_mod._PER_SYSTEM_STAGE = _per_system_stage  # type: ignore[attr-defined]
        _run_today_mod._PER_SYSTEM_EXIT = _per_system_exit  # type: ignore[attr-defined]
    except Exception:
        pass

    # ここでは何もしない（サイドバーで設定済みの環境変数を利用）

    # シグナル計算時に必要な日数分だけデータを渡すようにcompute_today_signalsへ
    with st.spinner("実行中... (経過時間表示あり)"):
        final_df, per_system = compute_today_signals(
            syms,
            capital_long=float(st.session_state["today_cap_long"]),
            capital_short=float(st.session_state["today_cap_short"]),
            save_csv=save_csv,
            log_callback=_ui_log,
            progress_callback=_ui_progress,
            per_system_progress=_per_system_progress,
            # 事前ロードは行わず、内部ローダに任せる
            parallel=bool(run_parallel),
        )

    # DataFrameのインデックスをリセットして疑似インデックスを排除
    final_df = final_df.reset_index(drop=True)
    per_system = {name: df.reset_index(drop=True) for name, df in per_system.items()}

    # 追加: メトリクス行を filter/setup/cand/entry/exit の5種で表示
    try:
        if st.session_state.get("ui_vis", {}).get("per_system_progress", True):
            for i in range(1, 8):
                key = f"system{i}"
                sc = stage_counts.get(key, {})
                # Fallback 補完
                if sc.get("cand") is None:
                    df_sys = per_system.get(key)
                    sc["cand"] = 0 if df_sys is None or df_sys.empty else int(len(df_sys))
                # エントリー数は最終結果から算出
                try:
                    if not final_df.empty and "system" in final_df.columns:
                        sc["entry"] = int((final_df["system"].str.lower() == key).sum())
                    elif sc.get("entry") is None:
                        sc["entry"] = 0
                except Exception:
                    pass
                # 先頭に対象件数（初回0%時のfilter_cntを流用）を表示
                target_txt = "-"
                try:
                    # 0% 通知時に一時的に filter に総数が入る場合がある
                    if sc.get("target") is not None:
                        target_txt = str(sc.get("target"))
                    elif sc.get("filter") is not None and sc.get("setup") is None:
                        target_txt = str(sc.get("filter"))
                except Exception:
                    pass
                # 1行の短縮名表示
                labels = [
                    ("Tgt", target_txt),
                    ("FILpass", sc.get("filter", "-")),
                    ("STUpass", sc.get("setup", "-")),
                    ("TRDlist", sc.get("cand", "-")),
                    ("Entry", sc.get("entry", "-")),
                    ("Exit", sc.get("exit", "-")),
                ]
                parts = [f"{nm} {('-' if v is None else v)}" for nm, v in labels]
                line = "  ".join(map(str, parts))
                if key in sys_metrics_txt:
                    sys_metrics_txt[key].text(line)
    except Exception:
        pass

    # 表示順を system1→system7 で統一し、最終結果も同順に並べ替え
    system_order = [f"system{i}" for i in range(1, 8)]
    if not final_df.empty and "system" in final_df.columns:
        try:
            tmp = final_df.copy()
            tmp["_system_no"] = (
                tmp["system"].astype(str).str.extract(r"(\d+)").fillna(0).astype(int)
            )
            sort_cols = [c for c in ["side", "_system_no"] if c in tmp.columns]
            tmp = tmp.sort_values(sort_cols, kind="stable").drop(
                columns=["_system_no"], errors="ignore"
            )
            final_df = tmp.reset_index(drop=True)
        except Exception:
            pass

    # 項番（1始まり）を付与
    if final_df is not None and not final_df.empty:
        try:
            final_df.insert(0, "no", range(1, len(final_df) + 1))
        except Exception:
            pass

    # 処理終了時に総経過時間を表示（分+秒）
    total_elapsed = max(0, time.time() - start_time)
    m, s = divmod(int(total_elapsed), 60)
    # 追加表示: 分・秒表示の総経過時間（重複表示の場合は本行を採用）
    st.info(f"総経過時間: {m}分{s}秒")
    # ファイルにも終了ログ（件数付き）
    try:
        final_n = 0 if final_df is None or final_df.empty else int(len(final_df))
        per_counts = []
        try:
            for name, df in per_system.items():
                per_counts.append(f"{name}={0 if df is None or df.empty else len(df)}")
        except Exception:
            per_counts = []
        detail = f" | システム別: {', '.join(per_counts)}" if per_counts else ""
        _get_today_logger().info(
            f"✅ 本日のシグナル: シグナル検出処理終了 (経過 {m}分{s}秒, "
            f"最終候補 {final_n} 件){detail}"
        )
    except Exception:
        pass

    # 実行ログは廃止。代わりにここに system 別ログタブを表示
    per_system_logs: dict[str, list[str]] = {f"system{i}": [] for i in range(1, 8)}
    # 指標計算や冗長行はタブ内でも非表示にする
    _skip_in_tabs = (
        "📊 指標計算",
        "⏱️ バッチ時間",
        "🧮 指標データ",
        "🧮 指標データロード",
        "🧮 共有指標の前計算",
        "📦 基礎データロード",
        "候補抽出",
        "インジケーター",
        "indicator",
        "indicators",
        "batch time",
        "next batch size",
    )
    for ln in log_lines:
        # タブ表示ではスキップすべきログを除外
        try:
            if any(k in ln for k in _skip_in_tabs):
                continue
        except Exception:
            pass
        ln_l = ln.lower()
        for i in range(1, 8):
            key = f"system{i}"
            tag1 = f"[system{i}]"  # 旧形式
            tag2 = f" {key}:"  # 現行の『🔎 systemX: ...』など
            tag3 = f"{key}:"  # 行頭等に現れる場合も拾う
            tag4 = f" {key}："  # 全角コロン対応
            if (tag1 in ln_l) or (tag2 in ln_l) or (tag3 in ln_l) or (tag4 in ln_l):
                per_system_logs[key].append(ln)
                break
    any_sys_logs = any(per_system_logs[k] for k in per_system_logs)
    if any_sys_logs:
        tabs = st.tabs([f"system{i}" for i in range(1, 8)])
        for i, key in enumerate([f"system{i}" for i in range(1, 8)]):
            logs = per_system_logs[key]
            if not logs:
                continue
            with tabs[i]:
                st.text_area(
                    label=f"ログ（{key}）",
                    key=f"logs_{key}",
                    value="\n".join(logs[-1000:]),
                    height=380,
                    disabled=True,
                )
                # System2 のフィルタ内訳があれば補足表示
                try:
                    if key == "system2":
                        # 直近に受け取った内訳ログを抽出
                        detail_lines = [
                            x for x in logs if ("フィルタ内訳:" in x or "filter breakdown:" in x)
                        ]
                        if detail_lines:
                            # 最後の1行だけ表示（文字列想定）
                            last_line = str(detail_lines[-1])
                            # 時刻/経過のプリフィックス以降を抽出
                            try:
                                disp = last_line.split("] ", 1)[1]
                            except Exception:
                                disp = last_line
                            st.caption(disp)
                except Exception:
                    pass

    # 通知は内部エンジン側で送信済み（重複を避けるためここでは送らない）

    # === 今日の手仕舞い候補（MOC）を推定して集計・発注オプションを提供 ===
    st.subheader("今日の手仕舞い候補（MOC）")
    exits_today_rows: list[dict[str, Any]] = []
    planned_rows: list[dict[str, Any]] = []  # for tomorrow open/close
    exit_counts: dict[str, int] = {f"system{i}": 0 for i in range(1, 8)}
    try:
        # 口座のポジションとエントリー日付のローカル記録を読み込む
        client_tmp = ba.get_client(paper=paper_mode)
        positions = list(client_tmp.get_all_positions())
        entry_map = load_entry_dates()
        # symbol->system マップ（エントリー時に保存されたもの）
        sym_map_path = Path("data/symbol_system_map.json")
        try:
            import json as _json

            symbol_system_map = (
                _json.loads(sym_map_path.read_text(encoding="utf-8"))
                if sym_map_path.exists()
                else {}
            )
        except Exception:
            symbol_system_map = {}

        # strategy クラスを遅延import
        from strategies.system1_strategy import System1Strategy
        from strategies.system2_strategy import System2Strategy
        from strategies.system3_strategy import System3Strategy
        from strategies.system4_strategy import System4Strategy
        from strategies.system5_strategy import System5Strategy
        from strategies.system6_strategy import System6Strategy

        latest_trading_day = None
        # まず SPY で最新営業日を得る（fallback で df の最終日）
        try:
            spy_df = load_price("SPY", cache_profile="rolling")
            if spy_df is not None and not spy_df.empty:
                latest_trading_day = pd.to_datetime(spy_df.index[-1]).normalize()
        except Exception:
            latest_trading_day = None

        for pos in positions:
            try:
                sym = str(getattr(pos, "symbol", "")).upper()
                if not sym:
                    continue
                qty = int(abs(float(getattr(pos, "qty", 0)) or 0))
                if qty <= 0:
                    continue
                pos_side = str(getattr(pos, "side", "")).lower()
                # system の推定（エントリー時のマップが最優先）
                system = str(symbol_system_map.get(sym, "")).lower()
                if not system:
                    if sym == "SPY" and pos_side == "short":
                        system = "system7"
                    else:
                        # 不明な場合は保守的にスキップ
                        continue
                # system7（SPYヘッジ）はここでは扱わない（別ロジック）
                if system == "system7":
                    continue
                # エントリー日付（ローカル記録）。無ければスキップ
                entry_date_str = entry_map.get(sym)
                if not entry_date_str:
                    continue
                entry_dt = pd.to_datetime(entry_date_str).normalize()
                # 価格データ
                df_price = load_price(sym, cache_profile="full")
                if df_price is None or df_price.empty:
                    continue
                # index を DatetimeIndex に揃える
                try:
                    df = df_price.copy(deep=False)
                    if "Date" in df.columns:
                        df.index = pd.Index(pd.to_datetime(df["Date"]).dt.normalize())
                    else:
                        df.index = pd.Index(pd.to_datetime(df.index).normalize())
                except Exception:
                    continue
                if latest_trading_day is None and len(df.index) > 0:
                    latest_trading_day = pd.to_datetime(df.index[-1]).normalize()
                # entry_idx を探す（見つからない場合は最も近い将来日に丸め）
                try:
                    idx = df.index
                    if entry_dt in idx:
                        arr = idx.get_indexer([entry_dt])
                    else:
                        arr = idx.get_indexer([entry_dt], method="bfill")
                    entry_idx = int(arr[0]) if len(arr) and arr[0] >= 0 else -1
                    if entry_idx < 0:
                        continue
                except Exception:
                    continue

                # システム別に entry/stop を近似再現
                stg = None
                entry_price = None
                stop_price = None
                try:
                    prev_close = float(df.iloc[int(max(0, entry_idx - 1))]["Close"])
                    if system == "system1":
                        stg = System1Strategy()
                        entry_price = float(df.iloc[int(entry_idx)]["Open"])
                        atr20 = float(df.iloc[int(max(0, entry_idx - 1))]["ATR20"])
                        stop_mult = float(stg.config.get("stop_atr_multiple", 5.0))
                        stop_price = entry_price - stop_mult * atr20
                    elif system == "system2":
                        stg = System2Strategy()
                        entry_price = float(df.iloc[int(entry_idx)]["Open"])
                        atr = float(df.iloc[int(max(0, entry_idx - 1))]["ATR10"])
                        stop_mult = float(stg.config.get("stop_atr_multiple", 3.0))
                        stop_price = entry_price + stop_mult * atr
                    elif system == "system6":
                        stg = System6Strategy()
                        ratio = float(stg.config.get("entry_price_ratio_vs_prev_close", 1.05))
                        entry_price = round(prev_close * ratio, 2)
                        atr = float(df.iloc[int(max(0, entry_idx - 1))]["ATR10"])
                        stop_mult = float(stg.config.get("stop_atr_multiple", 3.0))
                        stop_price = entry_price + stop_mult * atr
                    elif system == "system3":
                        stg = System3Strategy()
                        ratio = float(stg.config.get("entry_price_ratio_vs_prev_close", 0.93))
                        entry_price = round(prev_close * ratio, 2)
                        atr = float(df.iloc[int(max(0, entry_idx - 1))]["ATR10"])
                        stop_mult = float(stg.config.get("stop_atr_multiple", 2.5))
                        stop_price = entry_price - stop_mult * atr
                    elif system == "system4":
                        stg = System4Strategy()
                        entry_price = float(df.iloc[int(entry_idx)]["Open"])
                        atr40 = float(df.iloc[int(max(0, entry_idx - 1))]["ATR40"])
                        stop_mult = float(stg.config.get("stop_atr_multiple", 1.5))
                        stop_price = entry_price - stop_mult * atr40
                    elif system == "system5":
                        stg = System5Strategy()
                        ratio = float(stg.config.get("entry_price_ratio_vs_prev_close", 0.97))
                        entry_price = round(prev_close * ratio, 2)
                        atr = float(df.iloc[int(max(0, entry_idx - 1))]["ATR10"])
                        stop_mult = float(stg.config.get("stop_atr_multiple", 3.0))
                        stop_price = entry_price - stop_mult * atr
                        # System5 は ATR を参照するので一部内部状態も付与
                        try:
                            stg._last_entry_atr = atr  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    else:
                        continue
                except Exception:
                    continue
                if stg is None or entry_price is None or stop_price is None:
                    continue

                try:
                    exit_price, exit_date = stg.compute_exit(
                        df,
                        int(entry_idx),
                        float(entry_price),
                        float(stop_price),
                    )
                except Exception:
                    continue

                # "今日の大引けで手仕舞い" のみ対象（MOC）。System5 の Open 決済は除外。
                today_norm = pd.to_datetime(df.index[-1]).normalize()
                if latest_trading_day is not None:
                    today_norm = latest_trading_day
                is_today_exit = pd.to_datetime(exit_date).normalize() == today_norm
                if is_today_exit:
                    if system == "system5":
                        # System5 は翌日寄り決済
                        planned_rows.append(
                            {
                                "symbol": sym,
                                "qty": qty,
                                "position_side": pos_side,
                                "system": system,
                                "when": "tomorrow_open",
                            }
                        )
                    else:
                        when = "today_close"
                        exits_today_rows.append(
                            {
                                "symbol": sym,
                                "qty": qty,
                                "position_side": pos_side,
                                "system": system,
                                "when": when,
                            }
                        )
                        exit_counts[system] = exit_counts.get(system, 0) + 1
                else:
                    # 翌日寄り/引けの予約を前日に作成
                    if system == "system5":
                        planned_rows.append(
                            {
                                "symbol": sym,
                                "qty": qty,
                                "position_side": pos_side,
                                "system": system,
                                "when": "tomorrow_open",
                            }
                        )
                    elif system in {"system1", "system2", "system3", "system6"}:
                        planned_rows.append(
                            {
                                "symbol": sym,
                                "qty": qty,
                                "position_side": pos_side,
                                "system": system,
                                "when": "tomorrow_close",
                            }
                        )
            except Exception:
                continue

        # UI 表示 + stage_counts へ反映
        if exits_today_rows:
            df_ex = pd.DataFrame(exits_today_rows)
            st.dataframe(df_ex, use_container_width=True)
            # 全体フェーズ: エグジット
            try:
                _set_phase_label("エグジット")
            except Exception:
                pass
            # stage_counts を更新してメトリクスに exit を反映
            for k, v in exit_counts.items():
                if v and k in stage_counts:
                    stage_counts[k]["exit"] = int(v)
            # 既存のメトリクス表示を更新（exit 反映）
            try:
                for i2 in range(1, 8):
                    key2 = f"system{i2}"
                    sc2 = stage_counts.get(key2, {})
                    if key2 in sys_metrics_txt and sys_metrics_txt.get(key2) is not None:
                        target_txt2 = "-"
                        try:
                            if sc2.get("target") is not None:
                                target_txt2 = str(sc2.get("target"))
                            elif sc2.get("filter") is not None and sc2.get("setup") is None:
                                target_txt2 = str(sc2.get("filter"))
                        except Exception:
                            target_txt2 = "-"
                        # 行長回避のため一部を事前に文字列化
                        _f_val = sc2.get("filter")
                        _f_txt = "-" if _f_val is None else str(_f_val)

                        def _v2(x):
                            return x if isinstance(x, str) else ("-" if x is None else str(x))

                        lines2 = [
                            f"Tgt {_v2(target_txt2)}",
                            f"FILpass {_v2(_f_txt)}",
                            f"STUpass {_v2(sc2.get('setup'))}",
                            f"TRDlist {_v2(sc2.get('cand'))}",
                            f"Entry {_v2(sc2.get('entry'))}",
                            f"Exit {_v2(sc2.get('exit'))}",
                        ]
                        sys_metrics_txt[key2].text("\n".join(lines2))
            except Exception:
                pass
            # 発注ボタン（MOC）
            if st.button("本日分の手仕舞い注文（MOC）を送信"):
                from common.alpaca_order import submit_exit_orders_df

                res = submit_exit_orders_df(
                    df_ex,
                    paper=paper_mode,
                    tif="CLS",
                    retries=int(retries),
                    delay=float(max(0.0, delay)),
                    log_callback=_ui_log,
                    notify=True,
                )
                if res is not None and not res.empty:
                    st.dataframe(res, use_container_width=True)
        else:
            st.info("本日大引けでの手仕舞い候補はありません。")

        # 計画出力（翌日寄り/引け）
        if planned_rows:
            st.caption("明日発注する手仕舞い計画（保存→スケジューラが実行）")
            df_plan = pd.DataFrame(planned_rows)
            st.dataframe(df_plan, use_container_width=True)
            if st.button("計画を保存（JSONL）"):
                import json as _json

                plan_path = Path("data/planned_exits.jsonl")
                try:
                    plan_path.parent.mkdir(parents=True, exist_ok=True)
                    with plan_path.open("w", encoding="utf-8") as f:
                        for r in planned_rows:
                            f.write(_json.dumps(r, ensure_ascii=False) + "\n")
                    st.success(f"保存しました: {plan_path}")
                except Exception as e:
                    st.error(f"保存に失敗: {e}")

            st.write("")
            col_open, col_close = st.columns(2)
            with col_open:
                if st.button("⏱️ 寄り（OPG）予約を今すぐ送信", key="run_scheduler_open"):
                    try:
                        from schedulers.next_day_exits import submit_planned_exits as _run_sched

                        df_exec = _run_sched("open")
                        if df_exec is not None and not df_exec.empty:
                            st.success("寄り（OPG）分の予約送信を実行しました。結果を表示します。")
                            st.dataframe(df_exec, use_container_width=True)
                        else:
                            st.info("寄り（OPG）対象の予約はありませんでした。")
                    except Exception as e:
                        st.error(f"寄り（OPG）予約の実行に失敗: {e}")
            with col_close:
                if st.button("⏱️ 引け（CLS）予約を今すぐ送信", key="run_scheduler_close"):
                    try:
                        from schedulers.next_day_exits import submit_planned_exits as _run_sched

                        df_exec = _run_sched("close")
                        if df_exec is not None and not df_exec.empty:
                            st.success("引け（CLS）分の予約送信を実行しました。結果を表示します。")
                            st.dataframe(df_exec, use_container_width=True)
                        else:
                            st.info("引け（CLS）対象の予約はありませんでした。")
                    except Exception as e:
                        st.error(f"引け（CLS）予約の実行に失敗: {e}")
    except Exception as e:
        st.warning(f"手仕舞い候補の推定に失敗しました: {e}")

    st.subheader("最終選定銘柄")
    if final_df is None or final_df.empty:
        st.info("本日のシグナルはありません。")
    else:
        st.dataframe(final_df, use_container_width=True)
        csv = final_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "最終CSVをダウンロード",
            data=csv,
            file_name="today_signals_final.csv",
        )

        # Alpaca 自動発注（任意）
        if do_trade:
            st.divider()
            st.subheader("Alpaca自動発注結果")
            system_order_type = {
                "system1": "market",
                "system3": "market",
                "system4": "market",
                "system5": "market",
                "system2": "limit",
                "system6": "limit",
                "system7": "limit",
            }
            results_df = submit_orders_df(
                final_df,
                paper=paper_mode,
                order_type=None,
                system_order_type=system_order_type,
                tif="DAY",
                retries=int(retries),
                delay=float(max(0.0, delay)),
                log_callback=_ui_log,
                notify=True,
            )
            if results_df is not None and not results_df.empty:
                st.dataframe(results_df, use_container_width=True)
                if poll_status and any(results_df["order_id"].fillna("").astype(str)):
                    st.info("注文状況を10秒間ポーリングします...")
                    # ポーリングは新規にクライアントを作って実施
                    try:
                        client = ba.get_client(paper=paper_mode)
                    except Exception:
                        client = None
                    if client is not None:
                        order_ids = [str(oid) for oid in results_df["order_id"].tolist() if oid]
                        end = time.time() + 10
                        last: dict[str, Any] = {}
                        while time.time() < end:
                            status_map = ba.get_orders_status_map(client, order_ids)
                            if status_map != last:
                                if status_map:
                                    st.write(status_map)
                                last = status_map
                            time.sleep(1.0)
            # 注文後に余力を自動更新（buying_power/cash を取得し、長短を半々に再設定）
            if update_bp_after:
                try:
                    client2 = ba.get_client(paper=paper_mode)
                    acct = client2.get_account()
                    bp_raw = getattr(acct, "buying_power", None)
                    if bp_raw is None:
                        bp_raw = getattr(acct, "cash", None)
                    if bp_raw is not None:
                        bp = float(bp_raw)
                        st.session_state["today_cap_long"] = round(bp / 2.0, 2)
                        st.session_state["today_cap_short"] = round(bp / 2.0, 2)
                        st.success(
                            "約定反映後の余力で長短を再設定しました: "
                            f"${st.session_state['today_cap_long']} / "
                            f"${st.session_state['today_cap_short']}"
                        )
                        try:
                            _ui_log(
                                f"🔄 Alpaca口座余力を更新: buying_power={bp:.2f} "
                                f"→ long/short={bp/2:.2f}"
                            )
                        except Exception:
                            pass
                    else:
                        st.warning("Alpaca口座情報: buying_power/cashが取得できません（更新なし）")
                except Exception as e:
                    st.error(f"余力の自動更新に失敗: {e}")
    with st.expander("システム別詳細"):
        for name in system_order:
            df = per_system.get(name)
            st.markdown(f"#### {name}")
            if df is None or df.empty:
                st.write("(空) 候補は0件です。メトリクスを表示します。")
                try:
                    # 段階メトリクス（filter/setup/cand/entry/exit）を1行で表示
                    sc = stage_counts.get(name, {})
                    tgt_txt = "-"
                    try:
                        if sc.get("target") is not None:
                            tgt_txt = str(sc.get("target"))
                        elif sc.get("filter") is not None and sc.get("setup") is None:
                            tgt_txt = str(sc.get("filter"))
                    except Exception:
                        tgt_txt = "-"

                    def _v(x):
                        return "-" if x is None else str(x)

                    metrics_line = "  ".join(
                        [
                            f"Tgt {_v(tgt_txt)}",
                            f"FILpass {_v(sc.get('filter'))}",
                            f"STUpass {_v(sc.get('setup'))}",
                            f"TRDlist {_v(sc.get('cand'))}",
                            f"Entry {_v(sc.get('entry'))}",
                            f"Exit {_v(sc.get('exit'))}",
                        ]
                    )
                    st.caption(metrics_line)
                except Exception:
                    pass
                # 直近ログは表示しない（ユーザー要望）
            else:
                # show dataframe (includes reason column if available)
                st.dataframe(df, use_container_width=True)
                csv2 = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"{name}のCSVをダウンロード",
                    data=csv2,
                    file_name=f"signals_{name}.csv",
                    key=f"{name}_download_csv",
                )

    # ④ 前回結果を別出し（既に run_all_systems_today が出力しているログをサマリ化）
    prev_msgs = [line for line in log_lines if line and ("(前回結果) system" in line)]
    if prev_msgs:
        # 件数と時刻を抽出し、system番号順に並べ替え
        import re as _re

        def _parse_prev_line(ln: str):
            # [YYYY-mm-dd HH:MM:SS | x分y秒] 🧾 ✅ (前回結果) systemX: N 件
            ts = ln.split("] ")[0].strip("[")
            m = _re.search(r"\(前回結果\) (system\d+):\s*(\d+)", ln)
            sys = m.group(1) if m else "system999"
            cnt = int(m.group(2)) if m else 0
            return sys, cnt, ts, ln

        parsed = [_parse_prev_line(x) for x in prev_msgs]
        order = {f"system{i}": i for i in range(1, 8)}
        parsed.sort(key=lambda t: order.get(t[0], 999))
        lines_sorted = [f"{p[2]} | {p[0]}: {p[1]}件\n{p[3]}" for p in parsed]
        with st.expander("前回結果（system別）", expanded=False):
            st.text("\n\n".join(lines_sorted))
