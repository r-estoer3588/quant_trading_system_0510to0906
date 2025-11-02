from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, cast

import pandas as pd
import streamlit as st

from common.stage_metrics import (
    DEFAULT_SYSTEM_ORDER,
    GLOBAL_STAGE_METRICS,
    StageMetricsStore,
    StageSnapshot,
)
from config.settings import get_settings

try:
    from config.environment import get_env_config
except Exception:
    get_env_config = None


def read_progress_state() -> dict[str, Any]:
    """Read the latest progress state snapshot from logs/progress_state.json.

    Returns empty dict if file doesn't exist or can't be parsed.
    This is faster than parsing JSONL for the latest event.
    """
    try:
        settings = get_settings(create_dirs=False)
        logs_dir = Path(getattr(settings, "LOGS_DIR", "logs"))
    except Exception:
        logs_dir = Path("logs")
    state_path = logs_dir / "progress_state.json"
    try:
        if not state_path.exists():
            return {}
        content = state_path.read_text(encoding="utf-8", errors="ignore")
        payload = json.loads(content)
        if isinstance(payload, dict):
            return payload
        return {}
    except Exception:
        return {}


def read_progress_events(limit: int = 50) -> list[dict[str, Any]]:
    """Return the latest progress events stored in ``progress_today.jsonl``."""
    try:
        settings = get_settings(create_dirs=False)
        logs_dir = Path(getattr(settings, "LOGS_DIR", "logs"))
    except Exception:
        logs_dir = Path("logs")
    jsonl_path = logs_dir / "progress_today.jsonl"
    try:
        if not jsonl_path.exists():
            return []
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        events: list[dict[str, Any]] = []
        for line in lines[-max(1, int(limit)) :]:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                events.append(obj)
        return events
    except Exception:
        return []


class ProgressUI:
    """全体進捗とログ表示を管理するヘルパー。"""

    def __init__(self, ui_vis: dict[str, Any]):
        self.show_overall = bool(ui_vis.get("overall_progress", True))
        self.show_data_load = bool(ui_vis.get("data_load_progress_lines", True))
        self.phase_title_area = st.empty()
        self.progress_area = st.empty()
        self.progress_bar = st.progress(0) if self.show_overall else None
        # progress_textは削除（タイトルで表示するため）
        self.phase_state: dict[str, Any] = {"percent": 0, "label": "対象読み込み"}
        self._render_title()

    def set_label(self, label: str) -> None:
        if not self.show_overall:
            return
        self.phase_state["label"] = label
        self._render_title()

    def update(self, done: int, total: int, tag: str) -> None:
        if not self.show_overall or self.progress_bar is None:
            return
        total = max(1, int(total))
        ratio = min(max(int(done), 0), total) / total
        percent = int(ratio * 100)
        self.phase_state["percent"] = percent
        mapped = self._map_phase(tag)
        if mapped:
            self.phase_state["label"] = mapped
        try:
            self.progress_bar.progress(percent)
        except Exception:
            pass
        # プログレスバー下のテキストは削除（タイトルで表示するため）
        self._render_title()

    def update_label_for_stage(self, stage_value: int) -> None:
        if not self.show_overall:
            return
        if stage_value <= 0:
            label = "対象準備"
        elif stage_value < 10:
            label = "対象読み込み"
        elif stage_value < 30:
            label = "フィルター"
        elif stage_value < 60:
            label = "セットアップ"
        elif stage_value < 90:
            label = "トレード候補選定"
        else:
            label = "エントリー"
        self.set_label(label)

    def _render_title(self) -> None:
        if not self.show_overall:
            return
        try:
            percent = int(self.phase_state.get("percent", 0))
            label = str(self.phase_state.get("label", "対象読み込み"))
            self.phase_title_area.markdown(f"## 進捗 {percent}%: {label}フェーズ")
        except Exception:
            pass

    @staticmethod
    def _map_phase(tag: str) -> str:
        try:
            t = (tag or "").lower()
        except Exception:
            t = ""
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
        # Finalize (allocation -> entry) remains エントリー
        if t in {"finalize", "エントリー"}:
            return "エントリー"
        # Exit / completion: show hand-off/close phase (手仕舞い)
        if t in {"exit", "done", "system_complete"}:
            return "エグジットフェーズ"
        return "対象読み込み"


class StageTracker:
    """システム別の進捗と件数メトリクスを管理する。"""

    def __init__(
        self,
        ui_vis: dict[str, Any],
        progress_ui: ProgressUI,
        *,
        progress_event_reader: Callable[[int], list[dict[str, Any]]] | None = None,
        has_streamlit_ctx: Callable[[], bool] | None = None,
    ) -> None:
        self.progress_ui = progress_ui
        self._read_progress_events = progress_event_reader or read_progress_events
        self._has_st_ctx = has_streamlit_ctx or (lambda: True)
        self.show_ui = (
            bool(ui_vis.get("per_system_progress", True)) and self._has_st_ctx()
        )
        self.bars: dict[str, Any] = {}
        self.stage_txt: dict[str, Any] = {}
        self.metrics_txt: dict[str, Any] = {}
        self.states: dict[str, int] = {}
        self.metrics_store = StageMetricsStore(DEFAULT_SYSTEM_ORDER)
        self.stage_counts = self.metrics_store.stage_counts
        # 最後に受け取ったステージ情報のデデュープ用タイムスタンプ
        self._last_event: dict[str, tuple[int, int, int, int, int, float]] = {}
        self.universe_total: int | None = None
        self.universe_target: int | None = None
        # JSONL進捗イベント同期用: 最後に読み込んだ候補数
        self._jsonl_candidates: dict[str, int] = {}
        if self.show_ui:
            sys_cols = st.columns(7)
            sys_labels = [f"System{i}" for i in range(1, 8)]
            for i, col in enumerate(sys_cols, start=1):
                key = f"system{i}"
                try:
                    col.caption(sys_labels[i - 1])
                    self.bars[key] = col.progress(0)
                    self.stage_txt[key] = col.empty()
                    self.metrics_txt[key] = col.empty()
                    self._render_metrics(key)
                except Exception:
                    self.show_ui = False
                    break
        self._initialize_from_store()

    def update_progress(self, name: str, phase: str) -> None:
        if not self.show_ui:
            return
        # 最新の候補数などを先に同期してから進捗を評価
        try:
            self._sync_from_jsonl_if_needed()
        except Exception:
            pass
        key = str(name).lower()
        progress_bar = self.bars.get(key)
        if progress_bar is None:
            return

        # フェーズに応じた適切な値を設定
        if phase == "start":
            # 開始時は強制的に0にリセット
            value = 0
            self.states[key] = 0
        elif phase == "done":
            # 完了時は強制的に100に設定
            value = 100
            self.states[key] = 100
        else:
            # その他のフェーズでは実際の進捗値を取得
            try:
                snapshot = GLOBAL_STAGE_METRICS.get_snapshot(key)
                if snapshot is not None:
                    value = snapshot.progress
                else:
                    value = self.states.get(key, 0)
            except Exception:
                value = self.states.get(key, 0)

            # 値を0-100に制限し、後退を防止
            try:
                value = int(value)
            except Exception:
                value = self.states.get(key, 0)
            value = max(0, min(100, value))
            prev = int(self.states.get(key, 0))
            if value < prev:
                value = prev
            else:
                self.states[key] = value

        try:
            progress_bar.progress(value)
            self.stage_txt[key].text(f"run {value}%" if value < 100 else "done 100%")
        except Exception:
            pass

        # JSONL進捗イベントと同期: 候補数を更新
        self._sync_from_jsonl_if_needed()

    def _sync_from_jsonl_if_needed(self) -> None:
        """最新の候補数を取得してメトリクスを更新する。

        優先順位:
        1. progress_state.json（最新イベントのスナップショット、最速）
        2. progress_today.jsonl（従来の JSONL フォールバック）
        """
        # まず state.json から最新の system_complete を試す
        try:
            state = read_progress_state()
            last_event = state.get("last_event")
            if (
                isinstance(last_event, dict)
                and last_event.get("event_type") == "system_complete"
            ):
                data = last_event.get("data", {})
                sys_name = data.get("system", "").lower()
                candidates = data.get("candidates")
                if sys_name and candidates is not None:
                    prev = self._jsonl_candidates.get(sys_name)
                    if prev != candidates:
                        self._jsonl_candidates[sys_name] = candidates
                        counts = self._ensure_counts(sys_name)
                        counts["cand"] = int(candidates)
                        counts["entry"] = int(candidates)
                        self._render_metrics(sys_name)
                    return  # state から取得成功したので終了
        except Exception:
            pass  # state 失敗時は JSONL にフォールバック

        # フォールバック: JSONL から取得（従来ロジック）
        try:
            events = self._read_progress_events(100)
        except Exception:
            events = []
        if not events:
            return
        for event in reversed(events):  # 新しい順に処理
            if event.get("event_type") != "system_complete":
                continue
            data = event.get("data", {})
            sys_name = data.get("system", "").lower()
            candidates = data.get("candidates")
            if sys_name and candidates is not None:
                # 候補数が変化していたら更新
                prev = self._jsonl_candidates.get(sys_name)
                if prev != candidates:
                    self._jsonl_candidates[sys_name] = candidates
                    counts = self._ensure_counts(sys_name)
                    counts["cand"] = int(candidates)
                    counts["entry"] = int(candidates)  # Entry も同じ値で更新
                    self._render_metrics(sys_name)

    def _sync_final_counts_from_jsonl(self) -> None:
        """最終候補数を取得してメトリクスを更新する。

        優先順位:
        1. progress_state.json から全システムの候補数を収集
        2. progress_today.jsonl から収集（フォールバック）
        """
        system_candidates: dict[str, int] = {}

        # まず state.json から収集を試みる
        try:
            state = read_progress_state()
            last_event = state.get("last_event")
            if (
                isinstance(last_event, dict)
                and last_event.get("event_type") == "system_complete"
            ):
                data = last_event.get("data", {})
                sys_name = data.get("system", "").lower()
                candidates = data.get("candidates")
                if sys_name and candidates is not None:
                    system_candidates[sys_name] = int(candidates)
        except Exception:
            pass

        # state から取得できなかった場合、または追加情報が必要な場合は JSONL も読む
        if not system_candidates:
            try:
                events = self._read_progress_events(50)
            except Exception:
                events = []
            if events:
                # system_complete イベントから各システムの候補数を取得
                for event in events:
                    if event.get("event_type") != "system_complete":
                        continue
                    data = event.get("data", {})
                    sys_name = data.get("system", "").lower()
                    candidates = data.get("candidates")
                    if sys_name and candidates is not None:
                        system_candidates[sys_name] = int(candidates)

        # 各システムのメトリクスを更新
        for sys_name, cand_count in system_candidates.items():
            counts = self._ensure_counts(sys_name)
            counts["cand"] = cand_count
            # Entry も候補数と同じに設定（配分前）
            if counts.get("entry") is None or counts["entry"] == 0:
                counts["entry"] = cand_count
            self._render_metrics(sys_name)

    def _initialize_from_store(self) -> None:
        try:
            stored_target = GLOBAL_STAGE_METRICS.get_universe_target()
            if stored_target is not None:
                self.universe_target = int(stored_target)
        except Exception:
            pass
        try:
            snapshots = GLOBAL_STAGE_METRICS.all_snapshots()
        except Exception:
            snapshots = {}
        for sys_name, snapshot in snapshots.items():
            try:
                self._apply_snapshot(sys_name, snapshot)
            except Exception:
                continue

    def _apply_snapshot(self, name: str, snapshot: StageSnapshot) -> None:
        key = str(name).lower()
        counts = self._ensure_counts(key)

        # ターゲット数の設定（優先度順で設定）
        if snapshot.target is not None:
            try:
                target_val = int(snapshot.target)
                counts["target"] = target_val
                self.universe_total = target_val
            except Exception:
                pass
        elif snapshot.filter_pass is not None and counts.get("target") is None:
            try:
                fallback_target = int(snapshot.filter_pass)
                counts["target"] = fallback_target
                if self.universe_total is None:
                    self.universe_total = fallback_target
            except Exception:
                pass

        # 進捗データの設定
        if snapshot.filter_pass is not None:
            try:
                counts["filter"] = int(snapshot.filter_pass)
                # フィルター通過数が設定されている場合、ターゲットがなければフィルター数をターゲットとして使用
                if counts.get("target") is None:
                    counts["target"] = int(snapshot.filter_pass)
                    if self.universe_total is None:
                        self.universe_total = int(snapshot.filter_pass)
            except Exception:
                pass
        if snapshot.setup_pass is not None:
            try:
                counts["setup"] = int(snapshot.setup_pass)
            except Exception:
                pass
        if snapshot.candidate_count is not None:
            try:
                counts["cand"] = self._clamp_trdlist(snapshot.candidate_count)
            except Exception:
                pass
        if snapshot.entry_count is not None:
            try:
                counts["entry"] = int(snapshot.entry_count)
            except Exception:
                pass
        if snapshot.exit_count is not None:
            try:
                counts["exit"] = int(snapshot.exit_count)
            except Exception:
                pass
        self._update_bar(key, snapshot.progress)
        self.progress_ui.update_label_for_stage(snapshot.progress)
        self._render_metrics(key)

    def update_stage(
        self,
        name: str,
        value: int,
        filter_cnt: int | None = None,
        setup_cnt: int | None = None,
        cand_cnt: int | None = None,
        final_cnt: int | None = None,
    ) -> None:
        key = str(name).lower()
        # 短時間内に同一内容の更新が来ると UI がフラッタリングするため、
        # 同一システム・同一値・同一カウントの更新は 0.5 秒以内は無視する。
        try:
            last = self._last_event.get(key)
            cur_sig = (
                value,
                int(filter_cnt) if filter_cnt is not None else -1,
                int(setup_cnt) if setup_cnt is not None else -1,
                int(cand_cnt) if cand_cnt is not None else -1,
                int(final_cnt) if final_cnt is not None else -1,
                time.time(),
            )
            if last is not None:
                same = last[0:5] == cur_sig[0:5]
                recent = (cur_sig[5] - last[5]) < 0.5
                if same and recent:
                    return
            self._last_event[key] = cur_sig
        except Exception:
            pass
        snapshot: StageSnapshot | None
        try:
            snapshot = GLOBAL_STAGE_METRICS.record_stage(
                key,
                value,
                filter_cnt,
                setup_cnt,
                cand_cnt,
                final_cnt,
                emit_event=False,
            )
        except Exception:
            snapshot = None
        if snapshot is not None:
            self._apply_snapshot(key, snapshot)
            return
        counts = self._ensure_counts(key)
        if filter_cnt is not None:
            try:
                filter_val = int(filter_cnt)
            except Exception:
                filter_val = None
            if filter_val is not None:
                if value == 0:
                    counts["target"] = filter_val
                    self.universe_total = filter_val
                else:
                    counts["filter"] = filter_val
                    if counts.get("target") is None:
                        counts["target"] = (
                            self.universe_total
                            if self.universe_total is not None
                            else filter_val
                        )
                        if self.universe_total is None:
                            self.universe_total = filter_val
        if setup_cnt is not None:
            counts["setup"] = int(setup_cnt)
        if cand_cnt is not None:
            counts["cand"] = self._clamp_trdlist(cand_cnt)
        if final_cnt is not None:
            counts["entry"] = int(final_cnt)
        self._update_bar(key, value)
        self.progress_ui.update_label_for_stage(value)
        self._render_metrics(key)

    def set_universe_target(self, tgt: int | None) -> None:
        """全体ユニバース（Tgt）を設定。UI に即時反映する。"""
        try:
            if tgt is None:
                self.universe_target = None
                self.universe_total = None
            else:
                self.universe_target = int(tgt)
                self.universe_total = int(tgt)
            GLOBAL_STAGE_METRICS.set_universe_target(self.universe_target)
        except Exception:
            self.universe_target = None
            self.universe_total = None
            try:
                GLOBAL_STAGE_METRICS.set_universe_target(None)
            except Exception:
                pass
        # 全 system の表示を更新
        self.refresh_all()

    def update_exit(self, name: str, count: int) -> None:
        key = str(name).lower()
        snapshot: StageSnapshot | None
        try:
            snapshot = GLOBAL_STAGE_METRICS.record_exit(key, count, emit_event=False)
        except Exception:
            snapshot = None
        if snapshot is not None:
            self._apply_snapshot(key, snapshot)
            return
        counts = self._ensure_counts(key)
        counts["exit"] = int(count)
        self._render_metrics(key)

    def finalize_counts(
        self, final_df: pd.DataFrame, per_system: dict[str, pd.DataFrame]
    ) -> None:  # noqa: E501
        """最終化：残った候補/エントリー数を補完し、全バーを100%にする。"""
        # まずJSONL進捗イベントから最終候補数を同期
        try:
            self._sync_final_counts_from_jsonl()
        except Exception:
            pass

        # AllocationSummary が dict で同梱されている場合、slot_candidates を候補数のフォールバックに使用する
        alloc_slot_candidates: dict[str, int] | None = None
        alloc_final_counts: dict[str, int] | None = None
        system_diagnostics_map: dict[str, dict] | None = None

        try:
            if isinstance(per_system, dict):
                alloc_dict = per_system.get("__allocation_summary__")
            else:
                alloc_dict = None
            if isinstance(alloc_dict, dict):
                # slot_candidates 取得
                cand_map = alloc_dict.get("slot_candidates")
                if isinstance(cand_map, dict):
                    # 正規化: keyは小文字system名に統一し、値はint化
                    alloc_slot_candidates = {}
                    for k, v in cand_map.items():
                        try:
                            key = str(k).strip().lower()
                            val = int(v) if v is not None else 0
                            alloc_slot_candidates[key] = max(0, val)
                        except Exception:
                            continue
                # final_counts 取得（エントリー数の最終確定値）
                final_map = alloc_dict.get("final_counts")
                if isinstance(final_map, dict):
                    alloc_final_counts = {}
                    for k, v in final_map.items():
                        try:
                            key = str(k).strip().lower()
                            val = int(v) if v is not None else 0
                            alloc_final_counts[key] = max(0, val)
                        except Exception:
                            continue
                # system_diagnostics 取得（setup_predicate_count用）
                diag_map = alloc_dict.get("system_diagnostics")
                if isinstance(diag_map, dict):
                    system_diagnostics_map = {}
                    for k, v in diag_map.items():
                        try:
                            key = str(k).strip().lower()
                            if isinstance(v, dict):
                                system_diagnostics_map[key] = v
                        except Exception:
                            continue
        except Exception:
            alloc_slot_candidates = None
            alloc_final_counts = None
            system_diagnostics_map = None
        for name, counts in self.stage_counts.items():
            snapshot: StageSnapshot | None
            try:
                snapshot = GLOBAL_STAGE_METRICS.get_snapshot(name)
            except Exception:
                snapshot = None
            if snapshot is not None:
                if counts.get("target") is None and snapshot.target is not None:
                    try:
                        counts["target"] = int(snapshot.target)
                        if self.universe_total is None:
                            self.universe_total = int(snapshot.target)
                    except Exception:
                        pass
                if counts.get("filter") is None and snapshot.filter_pass is not None:
                    try:
                        counts["filter"] = int(snapshot.filter_pass)
                    except Exception:
                        pass
                if counts.get("setup") is None and snapshot.setup_pass is not None:
                    try:
                        counts["setup"] = int(snapshot.setup_pass)
                    except Exception:
                        pass
                if counts.get("cand") is None and snapshot.candidate_count is not None:
                    try:
                        counts["cand"] = self._clamp_trdlist(snapshot.candidate_count)
                    except Exception:
                        pass
                if counts.get("entry") is None and snapshot.entry_count is not None:
                    try:
                        counts["entry"] = int(snapshot.entry_count)
                    except Exception:
                        pass
                if counts.get("exit") is None and snapshot.exit_count is not None:
                    try:
                        counts["exit"] = int(snapshot.exit_count)
                    except Exception:
                        pass
        try:
            system_series = (
                final_df["system"].astype(str).str.strip().str.lower()
                if "system" in final_df.columns
                else pd.Series(dtype=str)
            )
        except Exception:
            system_series = pd.Series(dtype=str)
        for name, counts in self.stage_counts.items():
            # diagnosticsからsetup_predicate_countを取得して設定
            if counts.get("setup") is None and system_diagnostics_map:
                try:
                    diag = system_diagnostics_map.get(name)
                    if isinstance(diag, dict):
                        setup_count = diag.get("setup_predicate_count")
                        if isinstance(setup_count, (int, float)) and setup_count >= 0:
                            counts["setup"] = int(setup_count)
                except Exception:
                    pass

            # diagnostics から ranked_top_n_count（ランキング段階で選ばれた top-N 件）を
            # 優先的に TRDlist 表示へ反映する。
            if system_diagnostics_map:
                try:
                    diag = system_diagnostics_map.get(name)
                    if isinstance(diag, dict):
                        r_topn = diag.get("ranked_top_n_count")
                        if isinstance(r_topn, (int, float)) and int(r_topn) > 0:
                            counts["cand"] = self._clamp_trdlist(int(r_topn))
                except Exception:
                    pass

            # cand が未設定 もしくは 0 の場合は AllocationSummary / per_system でフォールバック
            cand_val = counts.get("cand")
            if cand_val is None or int(cand_val or 0) <= 0:
                # 1) AllocationSummary の slot_candidates からフォールバック
                used = False
                try:
                    if (
                        alloc_slot_candidates is not None
                        and name in alloc_slot_candidates
                    ):
                        counts["cand"] = self._clamp_trdlist(
                            alloc_slot_candidates.get(name)
                        )
                        used = True
                except Exception:
                    used = False
                # 2) per_system の DataFrame 件数にフォールバック
                if not used:
                    df_sys = per_system.get(name)
                    if (
                        df_sys is None
                        or not isinstance(df_sys, pd.DataFrame)
                        or df_sys.empty
                    ):
                        counts["cand"] = 0
                    else:
                        counts["cand"] = self._clamp_trdlist(len(df_sys))
            if counts.get("entry") is None and not system_series.empty:
                try:
                    counts["entry"] = int((system_series == name).sum())
                except Exception:
                    counts["entry"] = 0
            # AllocationSummary.final_counts からの上書き（優先）
            if alloc_final_counts and name in alloc_final_counts:
                try:
                    counts["entry"] = int(alloc_final_counts[name])
                except Exception:
                    pass
            if counts.get("target") is None:
                if self.universe_total is not None:
                    counts["target"] = self.universe_total
                elif counts.get("filter") is not None and counts.get("setup") is None:
                    counts["target"] = counts.get("filter")
            try:
                GLOBAL_STAGE_METRICS.record_stage(
                    name,
                    int(
                        self.states.get(
                            name, 100 if counts.get("entry") is not None else 0
                        )
                    ),
                    counts.get("filter"),
                    counts.get("setup"),
                    counts.get("cand"),
                    counts.get("entry"),
                    emit_event=False,
                )
            except Exception:
                pass
        self.refresh_all()
        # Export a snapshot of the display metrics to results_csv for Playwright
        # or other automated diffing tools. This is intentionally best-effort
        # and must not raise on failure.
        try:
            self._export_metrics_snapshot()
        except Exception:
            pass

    def apply_exit_counts(self, exit_counts: dict[str, int]) -> None:
        any_applied = bool(exit_counts)
        for name, cnt in exit_counts.items():
            if cnt is None:
                continue
            snapshot: StageSnapshot | None
            try:
                snapshot = GLOBAL_STAGE_METRICS.record_exit(name, cnt, emit_event=False)
            except Exception:
                snapshot = None
            if snapshot is not None:
                self._apply_snapshot(name, snapshot)
            else:
                try:
                    self._ensure_counts(name)["exit"] = int(cnt)
                except Exception:
                    pass
        self.refresh_all()
        if any_applied:
            try:
                if self.progress_ui is not None:
                    self.progress_ui.update(8, 8, "exit")
            except Exception:
                pass

    def refresh_all(self) -> None:
        try:
            self._sync_final_counts_from_jsonl()
        except Exception:
            pass
        for name in self.metrics_store.systems():
            self._render_metrics(name)

    def _update_bar(self, key: str, value: int) -> None:
        if not self.show_ui:
            return
        progress_bar = self.bars.get(key)
        if progress_bar is None:
            return
        vv = max(0, min(100, int(value)))
        prev = int(self.states.get(key, 0))
        if vv < 100:
            try:
                snap = GLOBAL_STAGE_METRICS.get_snapshot(key)
            except Exception:
                snap = None
            if snap is not None:
                if snap.entry_count is not None:
                    vv = 100
                elif snap.candidate_count is not None and vv < 75:
                    vv = 75
                elif (
                    snap.setup_pass is not None
                    and snap.filter_pass is not None
                    and vv < 50
                ):
                    vv = 50
                elif snap.filter_pass is not None and vv < 25:
                    vv = 25
        vv = max(prev, vv)
        self.states[key] = vv
        try:
            progress_bar.progress(vv)
            self.stage_txt[key].text(f"run {vv}%" if vv < 100 else "done 100%")
        except Exception:
            pass

    def _render_metrics(self, key: str) -> None:
        placeholder = self.metrics_txt.get(key)
        if placeholder is None:
            return
        display = self.metrics_store.get_display_metrics(key)
        target_value = (
            self.universe_target
            if self.universe_target is not None
            else display.get("target")
        )
        try:
            lines = [
                f"Tgt {self._format_value(target_value)}",
                f"FILpass {self._format_value(display.get('filter'))}",
                f"STUpass {self._format_value(display.get('setup'))}",
                f"TRDlist {self._format_trdlist(display.get('cand'))}",
                f"Entry {self._format_value(display.get('entry'))}",
                f"Exit {self._format_value(display.get('exit'))}",
            ]
            placeholder.text("\n".join(lines))
        except Exception:
            pass

    def get_display_metrics(self, name: str) -> dict[str, int | None]:
        key = str(name).lower()
        result = self.metrics_store.get_display_metrics(key)
        return cast(dict[str, int | None], result)

    def _ensure_counts(self, name: str) -> dict[str, int | None]:
        result = self.metrics_store.ensure_display_metrics(name)
        return cast(dict[str, int | None], result)

    @staticmethod
    def _format_value(value: Any) -> str:
        result: str = "-" if value is None else str(value)
        return result

    @staticmethod
    def _clamp_trdlist(value: Any) -> int | None:
        result = StageMetricsStore.clamp_trdlist(value)
        return cast(int | None, result)

    def _format_trdlist(self, value: Any) -> str:
        if value is None:
            return "-"
        try:
            clamped_val = self._clamp_trdlist(value)
            return str(clamped_val) if clamped_val is not None else "-"
        except Exception:
            return "-"

    def _export_metrics_snapshot(self) -> None:
        try:
            settings2 = get_settings(create_dirs=True)
            results_dir = Path(
                getattr(settings2.outputs, "results_csv_dir", "results_csv")
            )
        except Exception:
            results_dir = Path("results_csv")
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fp = results_dir / f"ui_metrics_{ts}.json"
            payload: dict[str, Any] = {}
            for name in self.metrics_store.systems():
                try:
                    payload[name] = self.get_display_metrics(name)
                except Exception:
                    payload[name] = None
            with fp.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass
