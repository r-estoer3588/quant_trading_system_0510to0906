import os
import time
import streamlit as st
import pandas as pd
from typing import Dict
from common.utils import safe_filename
import common.i18n as i18n


def save_prepared_data_cache(
    data_dict: Dict[str, pd.DataFrame],
    system_name: str = "SystemX",
    base_dir: str = "data_cache",
    batch: int = 50,
) -> None:
    """
    Save prepared per-symbol DataFrames to `data_cache/<system_name>/` with progress UI.

    - data_dict: mapping of symbol -> prepared DataFrame (index=Date or has Date column)
    - system_name: e.g., "System1".."System7"
    - base_dir: top-level cache directory
    - batch: progress/log update cadence
    """
    if not data_dict:
        st.warning(i18n.tr("⚠️ 保存対象のデータがありません"))
        return

    # Save flat under base_dir (no per-system subfolder)
    dest_dir = base_dir
    os.makedirs(dest_dir, exist_ok=True)

    st.info(i18n.tr("💾 {system_name} 加工済データのキャッシュ保存を開始します...", system_name=system_name))
    total = len(data_dict)
    progress_bar = st.progress(0)
    log_area = st.empty()
    start_time = time.time()

    buffer = []
    for i, (sym, df) in enumerate(data_dict.items(), 1):
        path = os.path.join(dest_dir, f"{safe_filename(sym)}.csv")
        try:
            df.to_csv(path)
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
                f"📦 保存: {i}/{total} 件 完了 | 経過: {em}分{es}秒 / 残り: 約 {rm}分{rs}秒"
            )
            if extra:
                msg += f"\n{extra}"
            log_area.text(msg)
            buffer.clear()

    progress_bar.empty()
    st.success(i18n.tr("✅ {system_name} キャッシュ保存完了: {dest_dir} ({total} 件)", system_name=system_name, dest_dir=dest_dir, total=total))
