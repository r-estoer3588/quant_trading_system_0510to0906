"""
UI 要素の薄いラッパ。

- UIManager: ルート→システム→フェーズの階層を管理
  - 各フェーズ内に `info`/`log`/`progress` 用のスロットを用意
  - 既存コードが `log_area.text(...)` / `progress_bar.progress(v)` を
    直接使えるよう互換 API を提供

System2 で発生していた「info の表示がログより後ろに出る」問題を解消するため、
info 用スロットを先に確保し、`info()` はそのスロットを更新する設計にしている。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import streamlit as st


@dataclass
class PhaseContext:
    """フェーズ単位の UI ハンドル。"""

    container: "st.delta_generator.DeltaGenerator"
    title: Optional[str] = None

    def __post_init__(self) -> None:
        if self.title:
            try:
                self.container.caption(self.title)
            except Exception:
                pass
        # 表示順序を保証するために先に確保: info → log → progress
        self._info = self.container.empty()
        self._log = self.container.empty()
        self._progress = self.container.progress(0)

    @property
    def log_area(self):
        return self._log

    @property
    def progress_bar(self):
        return self._progress

    def info(self, msg: str) -> None:
        try:
            # 事前確保した info スロットを更新（位置が下がらないように）
            self._info.info(msg)
        except Exception:
            pass


class UIManager:
    """UI 階層を管理するシンプルなマネージャ。"""

    def __init__(self, *, root: "st.delta_generator.DeltaGenerator" | None = None):
        self._root = root or st.container()
        self._systems: Dict[str, UIManager] = {}
        self._phases: Dict[str, PhaseContext] = {}

    # --- 階層管理 ---
    def system(self, name: str, *, title: Optional[str] = None) -> "UIManager":
        if name not in self._systems:
            c = self._root.container()
            if title:
                try:
                    c.subheader(title)
                except Exception:
                    pass
            self._systems[name] = UIManager(root=c)
        return self._systems[name]

    def phase(self, name: str, *, title: Optional[str] = None) -> PhaseContext:
        if name not in self._phases:
            c = self._root.container()
            self._phases[name] = PhaseContext(container=c, title=title)
        return self._phases[name]

    # --- 互換 API（既存コード向け） ---
    def get_log_area(self, name: str = "log"):
        return self.phase(name).log_area

    def get_progress_bar(self, name: str = "progress"):
        return self.phase(name).progress_bar

    # 外部で `with ui.container:` と使えるよう公開
    @property
    def container(self):
        return self._root

