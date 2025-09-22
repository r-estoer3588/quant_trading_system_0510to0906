import os
import time

import pandas as pd
import streamlit as st

from common.cache_manager import round_dataframe
import common.i18n as i18n
from common.utils import safe_filename
from config.settings import get_settings


def save_prepared_data_cache(
    data_dict: dict[str, pd.DataFrame],
    system_name: str = "SystemX",
    base_dir: str = "data_cache",
    batch: int = 50,
) -> None:
    """
    Save prepared per-symbol DataFrames under `data_cache/` with Streamlit progress.

    差分保存: 既存CSVの最終日が新規DataFrameの最終日以上ならスキップします。

    - data_dict: symbol -> prepared DataFrame (index=Date or has Date column)
    - system_name: e.g., "System1".."System7"
    - base_dir: cache directory
    - batch: progress/log update cadence
    """
    if not data_dict:
        st.warning(i18n.tr("⚠ 保存対象のデータがありません"))
        return

    dest_dir = base_dir
    os.makedirs(dest_dir, exist_ok=True)

    st.info(
        i18n.tr(
            "💾 {system_name} 加工済データのキャッシュ保存を開始します...",
            system_name=system_name,
        )
    )
    total = len(data_dict)
    progress_bar = st.progress(0)
    log_area = st.empty()
    start_time = time.time()

    def _latest_date_from_df(x: pd.DataFrame):
        try:
            if "Date" in x.columns:
                d = pd.to_datetime(x["Date"], errors="coerce").dropna()
                if len(d):
                    return d.max().normalize()
        except Exception:
            pass
        try:
            idx = pd.to_datetime(x.index, errors="coerce").dropna()
            if len(idx):
                return idx.max().normalize()
        except Exception:
            pass
        return None

    def _latest_date_in_csv(path: str):
        if not os.path.exists(path):
            return None
        try:
            try:
                s = pd.read_csv(path, usecols=["Date"], parse_dates=["Date"])  # type: ignore[arg-type]
                col = "Date"
            except Exception:
                s = pd.read_csv(path, usecols=["date"], parse_dates=["date"])  # type: ignore[arg-type]
                s = s.rename(columns={"date": "Date"})
                col = "Date"
            if s is None or s.empty or col not in s.columns:
                return None
            d = pd.to_datetime(s[col], errors="coerce").dropna()
            if len(d) == 0:
                return None
            return d.max().normalize()
        except Exception:
            return None

    saved = 0
    skipped = 0
    buffer: list[str] = []
    for i, (sym, df) in enumerate(data_dict.items(), 1):
        path = os.path.join(dest_dir, f"{safe_filename(sym)}.csv")

        try:
            new_latest = _latest_date_from_df(df)
            old_latest = _latest_date_in_csv(path)
            should_skip = (
                old_latest is not None
                and new_latest is not None
                and old_latest >= new_latest
            )
        except Exception:
            should_skip = False

        if should_skip:
            skipped += 1
        else:
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
                out_df.to_csv(path)
                saved += 1
            except Exception as e:
                log_area.error(f"❌ {sym}: 保存に失敗しました - {e}")

        buffer.append(sym)
        progress_bar.progress(i / total)

        if i % batch == 0 or i == total:
            elapsed = time.time() - start_time
            remain = (elapsed / i) * (total - i) if i > 0 else 0
            em, es = divmod(int(elapsed), 60)
            rm, rs = divmod(int(remain), 60)
            extra = f"銘柄: {', '.join(buffer)}" if buffer else None
            msg = (
                f"📦 保存: {saved}/{i} 件 完了 | スキップ: {skipped}"
                f" | 経過: {em}分{es}秒 / 残り: 約 {rm}分{rs}秒"
            )
            if extra:
                msg += f"\n{extra}"
            log_area.text(msg)
            buffer.clear()

    progress_bar.empty()
    st.success(
        i18n.tr(
            (
                "✅ {system_name} キャッシュ保存完了: 保存 {saved} 件 / "
                "スキップ {skipped} 件 (合計 {total} 件) → {dest_dir}"
            ),
            system_name=system_name,
            saved=saved,
            skipped=skipped,
            total=total,
            dest_dir=dest_dir,
        )
    )
