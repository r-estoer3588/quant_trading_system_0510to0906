from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

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
from common.notifier import create_notifier
from common.alpaca_order import submit_orders_df
from common.profit_protection import evaluate_positions
from config.settings import get_settings
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

    run_parallel = st.checkbox("並列実行（システム横断）", value=True)
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

    st.header("表示/非表示")
    # 表示制御の既定値（初期値）
    ui_defaults = {
        "overall_progress": True,
        "per_system_progress": True,
        "data_load_progress_lines": True,  # 📦/🧮 の進捗行
        "execution_log": True,  # 全体の実行ログエクスパンダー
        "per_system_logs": True,  # システム別 実行ログエクスパンダー
        "previous_results": True,  # 前回結果（system別）
        "system_details": True,  # システム別詳細テーブル
    }
    # セッションに保持（初回のみ）
    if "ui_vis" not in st.session_state:
        st.session_state["ui_vis"] = ui_defaults.copy()

    ui_vis = st.session_state["ui_vis"]
    # チェックボックスで更新
    ui_vis["overall_progress"] = st.checkbox(
        "全体進捗バー", value=ui_vis.get("overall_progress", True), key="ui_overall_progress"
    )
    ui_vis["per_system_progress"] = st.checkbox(
        "システム別進捗バー",
        value=ui_vis.get("per_system_progress", True),
        key="ui_per_system_progress",
    )
    ui_vis["data_load_progress_lines"] = st.checkbox(
        "データロード進捗行（📦/🧮）",
        value=ui_vis.get("data_load_progress_lines", True),
        key="ui_data_load_progress",
    )
    ui_vis["execution_log"] = st.checkbox(
        "実行ログ（全体）", value=ui_vis.get("execution_log", True), key="ui_exec_log"
    )
    ui_vis["per_system_logs"] = st.checkbox(
        "システム別 実行ログ", value=ui_vis.get("per_system_logs", True), key="ui_per_system_logs"
    )
    ui_vis["previous_results"] = st.checkbox(
        "前回結果（system別）", value=ui_vis.get("previous_results", True), key="ui_prev_results"
    )
    ui_vis["system_details"] = st.checkbox(
        "システム別詳細（表）", value=ui_vis.get("system_details", True), key="ui_system_details"
    )
    # 保存
    st.session_state["ui_vis"] = ui_vis

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
    # 進捗表示用の領域（1行上書き）
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
        # 追加: メトリクス表示用の行（stageの下の行）
        sys_metrics_txt = {f"system{i}": sys_cols[i - 1].empty() for i in range(1, 8)}
        sys_states = {k: 0 for k in sys_bars.keys()}
    else:
        sys_bars = {}
        sys_stage_txt = {}
        sys_metrics_txt = {}
        sys_states = {}
    # 追加: 全ログを蓄積（UIで折り畳み表示用）
    log_lines: list[str] = []

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
                    )
                    # 不要ログ（UI表示では抑制したいもの）
                    skip_keywords = (
                        "進捗",
                        "インジケーター",
                        "indicator",
                        "indicators",
                        "指標計算",
                        "共有指標",
                        "バッチ時間",
                        "batch time",
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

    def _ui_progress(done: int, total: int, name: str) -> None:
        try:
            if not _has_st_ctx():
                return
            ui_vis2 = st.session_state.get("ui_vis", {})
            if not bool(ui_vis2.get("overall_progress", True)):
                return
            total = max(1, int(total))
            ratio = min(max(int(done), 0), total) / total
            prog.progress(int(ratio * 100))
            if name:
                prog_txt.text(f"進捗 {int(ratio*100)}%: {name}")
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
                sys_stage_txt[n].text("running… (50%)")
            elif phase == "done":
                sys_states[n] = 100
                bar.progress(100)
                sys_stage_txt[n].text("done (100%)")
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
            phase = (
                "filter"
                if vv < 25
                else "setup" if vv < 50 else "candidates" if vv < 75 else "final"
            )
            parts = []
            if filter_cnt is not None:
                parts.append(f"F:{filter_cnt}")
            if setup_cnt is not None:
                parts.append(f"S:{setup_cnt}")
            if cand_cnt is not None:
                parts.append(f"C:{cand_cnt}")
            if final_cnt is not None:
                parts.append(f"Final:{final_cnt}")
            summary = " | ".join(parts) if parts else "…"
            sys_stage_txt[n].text(f"{phase} {summary}")
        except Exception:
            pass

    # ボタン押下直後の開始ログをUIにも出力（ファイルにも出力されます）
    _ui_log("▶ 本日のシグナル: シグナル検出処理開始")

    # ステージ進捗の受け口を先に登録（スレッドから参照されるため）
    try:
        globals()["_PER_SYSTEM_STAGE"] = _per_system_stage
    except Exception:
        pass

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

    # 追加: 「done (100%)」の下に systemごとのメトリクスを表示
    try:
        ui_vis2 = st.session_state.get("ui_vis", {})
        if ui_vis2.get("per_system_progress", True):
            import re as _re

            metrics_map: dict[str, tuple[int, int]] = {}
            # ログから最新のメトリクス概要行を探す
            lines_rev = list(reversed(log_lines))
            target_line = None
            for ln in lines_rev:
                if "📊 メトリクス概要:" in ln:
                    target_line = ln
                    break
            if target_line:
                # 例: system1: pre=159, cand=0, system2: pre=76, cand=0, ...
                for m in _re.finditer(r"(system\d+):\s*pre=(\d+),\s*cand=(\d+)", target_line):
                    sys_name = m.group(1).lower()
                    pre = int(m.group(2))
                    cand = int(m.group(3))
                    metrics_map[sys_name] = (pre, cand)
            # Fallback: per_system の件数から cand を、pre は不明なら '-' 表示
            for i in range(1, 8):
                key = f"system{i}"
                pre, cand = metrics_map.get(key, (None, None)) if metrics_map else (None, None)
                if cand is None:
                    df_sys = per_system.get(key)
                    cand = 0 if df_sys is None or df_sys.empty else int(len(df_sys))
                pre_str = str(pre) if pre is not None else "-"
                try:
                    # 表示: pre/cand を done の下の行に
                    txt = f"pre={pre_str}, cand={cand}"
                    if key in sys_metrics_txt:
                        sys_metrics_txt[key].text(txt)
                except Exception:
                    pass
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

    # 追加: 実行ログをUIに折り畳み表示（CSVダウンロード付き）
    if st.session_state.get("ui_vis", {}).get("execution_log", True):
        with st.expander("実行ログ", expanded=False):
            try:
                st.code("\n".join(log_lines))
                log_csv = "\n".join(log_lines).encode("utf-8")
                st.download_button(
                    "実行ログCSVをダウンロード",
                    data=log_csv,
                    file_name="today_run_logs.csv",
                    mime="text/csv",
                    key="today_logs_csv",
                )
            except Exception:
                pass

    for name in system_order:
        df = per_system.get(name)
        syms2 = df["symbol"].tolist() if df is not None and not df.empty else []
        if syms2:
            notifier.send_signals(name, syms2)

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
    if st.session_state.get("ui_vis", {}).get("system_details", True):
        with st.expander("システム別詳細"):
            for name in system_order:
                df = per_system.get(name)
                st.markdown(f"#### {name}")
                if df is None or df.empty:
                    st.write("(空)")
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
    if prev_msgs and st.session_state.get("ui_vis", {}).get("previous_results", True):
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

    # ③ systemごとの実行ログ（[systemX] で始まる行）
    per_system_logs: dict[str, list[str]] = {f"system{i}": [] for i in range(1, 8)}
    for ln in log_lines:
        for i in range(1, 8):
            tag = f"[system{i}] "
            if ln.find(tag) != -1:
                per_system_logs[f"system{i}"].append(ln)
                break
    any_sys_logs = any(per_system_logs[k] for k in per_system_logs)
    if any_sys_logs and st.session_state.get("ui_vis", {}).get("per_system_logs", True):
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
