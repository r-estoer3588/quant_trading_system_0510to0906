from __future__ import annotations

from pathlib import Path

import streamlit as st

from common.i18n import language_selector, load_translations_from_dir, tr
from common.logging_utils import setup_logging
import common.ui_patch  # noqa: F401
from common.ui_tabs import render_batch_tab, render_integrated_tab
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

    tabs = st.tabs([tr("Integrated"), tr("Batch")] + [f"System{i}" for i in range(1, 8)])

    with tabs[0]:
        render_integrated_tab(settings, notifier)

    with tabs[1]:
        render_batch_tab(settings, logger, notifier)

    system_tabs = tabs[2:]
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
