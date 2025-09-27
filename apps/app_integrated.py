from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import streamlit as st

# プロジェクトルート（apps/ から1階層上）をパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.i18n import language_selector, load_translations_from_dir, tr
from common.logging_utils import setup_logging
import common.ui_patch  # noqa: F401
from common.ui_tabs import (
    render_batch_tab,
    render_cache_health_tab,
    render_integrated_tab,
    render_metrics_tab,
    render_positions_tab,
)
from common.utils_spy import get_spy_data_cached
from config.settings import get_settings

# Must be the first Streamlit command on the page
st.set_page_config(page_title="Trading Systems 1-7 (Integrated)", layout="wide")

# Mark that we are running inside the integrated UI to avoid duplicate widgets
st.session_state["_integrated_ui"] = True


# expose Notifier symbol for tests (module-level)
try:  # noqa: WPS501
    from common.notifier import Notifier, create_notifier  # type: ignore
except Exception:  # pragma: no cover

    class Notifier:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            pass


# Load external translations once at startup
load_translations_from_dir(Path(__file__).parent / "translations")


def render_digest_log(log_file_path: Path, container: Any) -> None:
    """
    progress_today.jsonl からリアルタイムで進捗を表示する。

    Args:
        log_file_path: progress_today.jsonl へのパス
        container: streamlit の container（st.empty() など）
    """
    try:
        if not log_file_path.exists():
            container.info(tr("No progress log available"))
            return

        # JSONLファイルを読み込み
        lines = []
        try:
            with open(log_file_path, encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            container.error(f"Error reading progress log: {e}")
            return

        if not lines:
            container.info(tr("Progress log is empty"))
            return

        # 最新の数行を表示用にパース
        recent_events = []
        for line in lines[-10:]:  # 最新10行
            try:
                event = json.loads(line)
                recent_events.append(event)
            except json.JSONDecodeError:
                continue

        if not recent_events:
            container.info(tr("No valid progress events"))
            return

        # 表示用のマークダウンを構築
        display_lines = []

        # 最新イベント（強調表示）
        latest_event = recent_events[-1]
        timestamp = latest_event.get("timestamp", "").split("T")[-1].split(".")[0]  # HH:MM:SS
        event_type = latest_event.get("event_type", "unknown")
        level = latest_event.get("level", "info")
        data = latest_event.get("data", {})

        # レベルに応じたアイコン
        level_icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(level, "📝")

        # 最新イベントの表示
        display_lines.append("### 🔄 Latest Progress")
        display_lines.append(f"{level_icon} **{event_type}** ({timestamp})")

        # データの主要情報を表示
        if data:
            key_info = []
            if "system" in data:
                key_info.append(f"System: **{data['system']}**")
            if "processed" in data and "total" in data:
                percentage = data.get("percentage", 0)
                key_info.append(
                    f"Progress: **{data['processed']}/{data['total']} ({percentage}%)**"
                )
            if "phase" in data:
                key_info.append(f"Phase: **{data['phase']}**")
            if "status" in data:
                key_info.append(f"Status: **{data['status']}**")

            if key_info:
                display_lines.append(" | ".join(key_info))

        # 最近のイベント履歴（簡略化）
        if len(recent_events) > 1:
            display_lines.append("### 📋 Recent Events")
            for event in recent_events[-5:-1]:  # 最新除く直近4件
                timestamp = event.get("timestamp", "").split("T")[-1].split(".")[0]
                event_type = event.get("event_type", "unknown")
                level = event.get("level", "info")
                level_icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(level, "📝")
                display_lines.append(f"- {level_icon} {timestamp} {event_type}")

        # 結合してcontainerに表示
        container.markdown("\n".join(display_lines))

    except Exception as e:
        container.error(f"Failed to render progress log: {e}")


def main() -> None:
    settings = get_settings(create_dirs=True)
    logger = setup_logging(settings)
    logger.info("app_integrated start")
    # Auto-detect Slack/Discord from environment
    # Notifier 縺ｯ驕・ｻｶ繧､繝ｳ繝昴・繝・

    try:
        # Slack が失敗した場合のみ Discord にフォールバック
        notifier = create_notifier(platform="slack", fallback=True)  # type: ignore
    except Exception:
        notifier = Notifier(platform="auto")  # type: ignore

    # Show language selector exactly once
    language_selector()

    st.title(tr("Trading Systems Integrated UI"))
    with st.expander(tr("settings"), expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("RESULTS_DIR:", str(settings.RESULTS_DIR))
            st.write("LOGS_DIR:", str(settings.LOGS_DIR))
        with col2:
            st.write("DATA_CACHE_DIR:", str(settings.DATA_CACHE_DIR))
            st.write("THREADS:", settings.THREADS_DEFAULT)
        with col3:
            st.write("DEFAULT CAPITAL:", settings.ui.default_capital)
            st.write("LOG LEVEL:", settings.logging.level)

    tabs = st.tabs(
        [tr("Integrated"), tr("Batch"), tr("Metrics"), tr("Positions"), "🩺 Cache Health"]
        + [f"System{i}" for i in range(1, 8)]
    )

    with tabs[0]:
        render_integrated_tab(settings, notifier)

    with tabs[1]:
        render_batch_tab(settings, logger, notifier)

    with tabs[2]:
        render_metrics_tab(settings)
    with tabs[3]:
        render_positions_tab(settings, notifier)

    with tabs[4]:
        render_cache_health_tab(settings)

    system_tabs = tabs[5:]
    for sys_idx, tab in enumerate(system_tabs, start=1):
        sys_name = f"System{sys_idx}"
        with tab:
            logger.info("%s tab start", sys_name)
            try:
                app_mod = __import__(f"app_system{sys_idx}")
                if sys_idx == 1:
                    spy_df = get_spy_data_cached()
                    app_mod.run_tab(spy_df=spy_df)
                else:
                    app_mod.run_tab()
            except Exception as e:  # noqa: BLE001
                logger.exception("%s tab error", sys_name)
                st.exception(e)
            finally:
                logger.info("%s tab done", sys_name)


if __name__ == "__main__":
    main()
