"""
JSONL イベント・スクリーンショット・診断スナップショットの同期分析ツール（3点同期）

やること:
1) スクショのタイムスタンプと JSONL の system_* イベントを照合（既存）
2) JSONL の候補数（system_complete.candidates）と 診断 JSON の ranked_top_n_count を突き合わせ
3) 代表スクショ（start/complete 近傍）を添付して、3点の一貫性をチェックし可視化

Usage:
    python tools/sync_analysis.py
出力:
    - screenshots/progress_tracking/sync_analysis.json
    - screenshots/progress_tracking/ANALYSIS_REPORT.md（3点同期セクションを含む）
    - screenshots/progress_tracking/sync_summary.json
    - screenshots/progress_tracking/tri_sync_report.json（3点同期詳細）
"""

from datetime import datetime, timedelta
import json
from pathlib import Path
import re
from typing import Any, Iterable, cast


def load_jsonl_events(jsonl_path: Path) -> list[dict]:
    """JSONLファイルからイベントをロード"""
    events: list[dict] = []
    if not jsonl_path.exists():
        return events

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


def parse_screenshot_timestamp(filename: str) -> datetime | None:
    """
    スクショファイル名からdatetimeオブジェクトを生成
    例: progress_20251013_065209_856.png -> 2025-10-13 06:52:09
    """
    match = re.match(r"progress_(\d{8})_(\d{6})_\d+\.png", filename)
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        dt_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    return None


def find_nearest_event(screenshot_time: datetime, events: list[dict]) -> dict | None:
    """スクショの時刻に最も近いJSONLイベントを検索"""
    min_delta = None
    nearest = None

    for event in events:
        try:
            # JSONLのタイムスタンプ形式: "2025/10/13 6:52:07" または ISO形式
            ts_str = event.get("timestamp", "")
            if "/" in ts_str:
                # "2025/10/13 6:52:07" 形式
                event_time = datetime.strptime(ts_str, "%Y/%m/%d %H:%M:%S")
            else:
                # ISO形式
                event_time = datetime.fromisoformat(ts_str)

            delta = abs((screenshot_time - event_time).total_seconds())
            if min_delta is None or delta < min_delta:
                min_delta = delta
                nearest = event
        except Exception:
            continue

    return nearest


def analyze_sync():
    """メイン分析処理"""
    # JSONLイベントロード
    jsonl_path = Path("logs/progress_today.jsonl")
    events: list[dict] = load_jsonl_events(jsonl_path)

    print(f"📋 JSONLイベント: {len(events)} 件")

    # system_start/complete イベントのみ抽出
    system_events = [
        e for e in events if e.get("event_type") in ["system_start", "system_complete"]
    ]
    print(f"   うちsystem関連: {len(system_events)} 件\n")

    # スクショ一覧取得
    screenshot_dir = Path("screenshots/progress_tracking")
    screenshots = sorted(screenshot_dir.glob("progress_*.png"))

    print(f"📸 スクリーンショット: {len(screenshots)} 枚\n")

    # 同期分析
    print("=" * 80)
    print("同期分析結果")
    print("=" * 80)

    sync_results = []

    for screenshot in screenshots:
        ss_time = parse_screenshot_timestamp(screenshot.name)
        if not ss_time:
            continue

        # 最も近いイベントを検索
        nearest = find_nearest_event(ss_time, system_events)

        if nearest:
            event_type = nearest.get("event_type")
            data = nearest.get("data", {})
            system = data.get("system", "unknown")
            candidates = data.get("candidates")

            result = {
                "screenshot": screenshot.name,
                "screenshot_time": ss_time.strftime("%H:%M:%S"),
                "event_type": event_type,
                "system": system,
                "candidates": candidates,
                "event_timestamp": nearest.get("timestamp"),
            }
            sync_results.append(result)

    # 結果出力
    print(f"\n{'時刻':<10} {'スクショ':<35} {'イベント':<18} {'System':<8} {'候補':<6}")
    print("-" * 80)

    for r in sync_results[:20]:  # 最初の20件を表示
        cand_str = (
            str(r.get("candidates", "-")) if r.get("candidates") is not None else "-"
        )
        print(
            f"{r['screenshot_time']:<10} {r['screenshot']:<35} {r['event_type']:<18} {r['system']:<8} {cand_str:<6}"
        )

    if len(sync_results) > 20:
        print(f"\n   ... 他 {len(sync_results) - 20} 件\n")

    # JSON出力
    output_path = Path("screenshots/progress_tracking/sync_analysis.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sync_results, f, indent=2, ensure_ascii=False)

    print(f"\n📊 詳細結果: {output_path}")

    # 各システムのスクショ枚数をカウント
    print("\n【システム別スクショ枚数】")
    system_counts: dict[str, int] = {}
    for r in sync_results:
        system = r.get("system", "unknown")
        system_counts[system] = system_counts.get(system, 0) + 1

    for system in sorted(system_counts.keys()):
        print(f"   {system}: {system_counts[system]} 枚")

    # 代表スクショ抽出とMarkdownレポート生成 + 3点同期
    md_path = Path("screenshots/progress_tracking/ANALYSIS_REPORT.md")
    try:
        # タイムスタンプ文字列を datetime に変換
        def _to_dt(ts: str | None) -> datetime | None:
            try:
                if ts is None:
                    return None
                if "/" in ts:
                    return datetime.strptime(ts, "%Y/%m/%d %H:%M:%S")
                return datetime.fromisoformat(ts)
            except Exception:
                return None

        systems = [f"system{i}" for i in range(1, 8)]
        lines: list[str] = []
        lines.append("# 同期分析レポート\n")
        lines.append(f"生成時刻: {datetime.now().isoformat()}\n")
        lines.append(f"- JSONLイベント: {len(events)} 件")
        lines.append(f"- システム関連イベント: {len(system_events)} 件")
        lines.append(f"- スクリーンショット: {len(screenshots)} 枚\n")

        # スクショの時刻配列
        ss_times: list[tuple[datetime, Path]] = []
        for ss in screenshots:
            dt = parse_screenshot_timestamp(ss.name)
            if dt:
                ss_times.append((dt, ss))
        ss_times.sort(key=lambda x: x[0])

        def nearest_ss(ts: datetime) -> Path | None:
            best = None
            best_delta = None
            for dt, p in ss_times:
                delta = abs((dt - ts).total_seconds())
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best = p
            if best and best_delta is not None and best_delta <= 5.0:
                return best
            return best

        lines.append("## 代表スクリーンショット（各Systemの開始/完了 近傍）\n")
        lines.append(
            "| System | Start Event | Start SS | Complete Event | Complete SS |\n"
        )
        lines.append("|---|---|---|---|---|\n")

        def _find_evt(evt_type: str, sys_name: str) -> dict | None:
            return next(
                (
                    e
                    for e in system_events
                    if e.get("event_type") == evt_type
                    and e.get("data", {}).get("system") == sys_name
                ),
                None,
            )

        # 代表スクショを保存して後で per_system にも入れる
        rep_ss: dict[str, dict[str, str | None]] = {}
        for sys_name in systems:
            st_e = _find_evt("system_start", sys_name)
            cm_e = _find_evt("system_complete", sys_name)
            st_ts = _to_dt(st_e.get("timestamp")) if st_e else None
            cm_ts = _to_dt(cm_e.get("timestamp")) if cm_e else None
            st_ss = nearest_ss(st_ts) if st_ts else None
            cm_ss = nearest_ss(cm_ts) if cm_ts else None
            rep_ss[sys_name] = {
                "start_ss": st_ss.name if st_ss else None,
                "complete_ss": cm_ss.name if cm_ss else None,
            }

            st_evt = st_e.get("timestamp") if st_e else "-"
            cm_evt = cm_e.get("timestamp") if cm_e else "-"
            st_link = f"![{st_ss.name}](./{st_ss.name})" if st_ss else "-"
            cm_link = f"![{cm_ss.name}](./{cm_ss.name})" if cm_ss else "-"
            lines.append(
                f"| {sys_name} | {st_evt} | {st_link} | {cm_evt} | {cm_link} |\n"
            )

        # overall start/end を推定
        def _ev_time(e: dict | None) -> datetime | None:
            if not e:
                return None
            return _to_dt(e.get("timestamp"))

        ev_phase4 = next(
            (
                e
                for e in events
                if e.get("event_type") == "phase4_signal_generation_start"
            ),
            None,
        )
        overall_start = _ev_time(ev_phase4)
        if overall_start is None:
            ev_s_first = next(
                (e for e in events if e.get("event_type") == "system_start"),
                None,
            )
            overall_start = _ev_time(ev_s_first)

        ev_pipeline = next(
            (e for e in reversed(events) if e.get("event_type") == "pipeline_complete"),
            None,
        )
        ev_ui_wrap = next(
            (
                e
                for e in reversed(events)
                if e.get("event_type") == "phase_ui_wrapper_complete"
            ),
            None,
        )
        ev_alloc_done = next(
            (
                e
                for e in reversed(events)
                if e.get("event_type") == "phase5_allocation_complete"
            ),
            None,
        )
        overall_end = (
            _ev_time(ev_pipeline) or _ev_time(ev_ui_wrap) or _ev_time(ev_alloc_done)
        )

        # per-system start/end/candidates (+ 代表スクショ名)
        per_system: dict[str, dict] = {}
        for sys_name in systems:
            st_e = _find_evt("system_start", sys_name)
            cm_e = _find_evt("system_complete", sys_name)
            st_t = _ev_time(st_e)
            cm_t = _ev_time(cm_e)
            cand = None
            try:
                if cm_e and isinstance(cm_e.get("data"), dict):
                    cand_val = cm_e["data"].get("candidates")
                    if isinstance(cand_val, (int, float)):
                        cand = int(cand_val)
            except Exception:
                cand = None
            dur = None
            if st_t and cm_t:
                dur = int((cm_t - st_t).total_seconds())
            per_system[sys_name] = {
                "start": st_t.isoformat() if st_t else None,
                "end": cm_t.isoformat() if cm_t else None,
                "duration_sec": dur,
                "candidates": cand,
                "start_ss": rep_ss.get(sys_name, {}).get("start_ss"),
                "complete_ss": rep_ss.get(sys_name, {}).get("complete_ss"),
            }

        # Markdown: 概要と所要時間
        pc_flag = ev_pipeline is not None
        lines.append("\n## タイムラインと所要時間\n")
        lines.append(f"- pipeline_complete: {'あり' if pc_flag else 'なし'}\n")
        if overall_start:
            lines.append(f"- overall_start: {overall_start.isoformat()}\n")
        if overall_end:
            lines.append(f"- overall_end: {overall_end.isoformat()}\n")
        if overall_start and overall_end:
            dur_all = int((overall_end - overall_start).total_seconds())
            lines.append(f"- overall_duration: {dur_all} 秒\n")

        lines.append("\n### System別 所要時間・候補数\n")
        lines.append("| System | start | end | duration(s) | candidates |\n")
        lines.append("|---|---|---|---:|---:|\n")
        for sys_name in systems:
            info = per_system.get(sys_name, {})
            lines.append(
                f"| {sys_name} | {info.get('start', '-')} | {info.get('end', '-')} | "
                f"{info.get('duration_sec', '-')} | {info.get('candidates', '-')} |\n"
            )

        # ===== 3点同期: 診断スナップショットを探索し、JSONL候補数と突き合わせ =====
        def _iter_diag_files() -> Iterable[Path]:
            candidates_paths = [
                Path("results_csv/diagnostics_test"),
                Path("results_csv_test/diagnostics_test"),
            ]
            for base in candidates_paths:
                if base.exists():
                    yield from base.glob("diagnostics_snapshot_*.json")

        def _load_json(path: Path) -> dict[str, Any] | None:
            try:
                return cast(
                    dict[str, Any],
                    json.loads(path.read_text(encoding="utf-8")),
                )
            except Exception:
                return None

        # overall_end 近傍のスナップショットを選ぶ（±45分、同日ファイルを優先）。無ければ最新を採用。
        chosen_diag_path: Path | None = None
        chosen_diag: dict[str, Any] | None = None
        if overall_end is None:
            diag_files = sorted(_iter_diag_files())
            if diag_files:
                chosen_diag_path = diag_files[-1]
                chosen_diag = _load_json(chosen_diag_path)
        else:
            best_dt_diff: float | None = None
            window = timedelta(minutes=45)
            # 全候補を集めて同日ファイルを優先
            all_files = list(_iter_diag_files())
            same_day_files: list[Path] = []
            for diag_file in all_files:
                data0 = _load_json(diag_file)
                if not isinstance(data0, dict):
                    continue
                exp0 = data0.get("export_date")
                try:
                    exp_dt0 = (
                        datetime.fromisoformat(exp0) if isinstance(exp0, str) else None
                    )
                except Exception:
                    exp_dt0 = None
                if exp_dt0 and exp_dt0.date() == overall_end.date():
                    same_day_files.append(diag_file)

            scan_list = same_day_files or all_files
            for diag_file in scan_list:
                data = _load_json(diag_file)
                if not isinstance(data, dict):
                    continue
                exp = data.get("export_date")
                try:
                    exp_dt = (
                        datetime.fromisoformat(exp) if isinstance(exp, str) else None
                    )
                except Exception:
                    exp_dt = None
                if exp_dt is None:
                    continue
                delta = abs((exp_dt - overall_end).total_seconds())
                if delta <= window.total_seconds() and (
                    best_dt_diff is None or delta < best_dt_diff
                ):
                    best_dt_diff = delta
                    chosen_diag_path = diag_file
                    chosen_diag = data
            if chosen_diag is None:
                diag_files = sorted(_iter_diag_files())
                if diag_files:
                    chosen_diag_path = diag_files[-1]
                    chosen_diag = _load_json(chosen_diag_path)

        # 診断から per-system の ranked_top_n_count を抽出
        diag_counts: dict[str, int | None] = {s: None for s in systems}
        diag_mode: str | None = None
        diag_export: str | None = None
        if isinstance(chosen_diag, dict):
            diag_mode = chosen_diag.get("mode")
            diag_export = chosen_diag.get("export_date")
            try:
                for item in chosen_diag.get("systems", []) or []:
                    sid = item.get("system_id")
                    diag_obj = item.get("diagnostics") or {}
                    if sid in diag_counts:
                        val = diag_obj.get("ranked_top_n_count")
                        if isinstance(val, (int, float)):
                            diag_counts[sid] = int(val)
            except Exception:
                pass

        # JSONL 合計 vs allocation_start の total_candidates を確認
        sum_jsonl = 0
        for s in systems:
            v = per_system.get(s, {}).get("candidates")
            if isinstance(v, int):
                sum_jsonl += v
        alloc_evt = next(
            (e for e in events if e.get("event_type") == "phase5_allocation_start"),
            None,
        )
        alloc_total = None
        if alloc_evt and isinstance(alloc_evt.get("data"), dict):
            try:
                alloc_total = int(alloc_evt["data"].get("total_candidates"))
            except Exception:
                alloc_total = None

        # 3点同期の結果を組み立て
        tri_rows: list[dict[str, Any]] = []
        for s in systems:
            row = {
                "system": s,
                "jsonl_candidates": per_system.get(s, {}).get("candidates"),
                "diagnostics_ranked_top_n": diag_counts.get(s),
                "start_ss": per_system.get(s, {}).get("start_ss"),
                "complete_ss": per_system.get(s, {}).get("complete_ss"),
            }
            a = row["jsonl_candidates"]
            b = row["diagnostics_ranked_top_n"]
            if a is None and b is None:
                status = "missing"
            elif isinstance(a, int) and isinstance(b, int):
                status = "match" if a == b else "mismatch"
            else:
                status = "partial"
            row["status"] = status
            tri_rows.append(row)

        tri_report = {
            "selected_diagnostics": str(chosen_diag_path) if chosen_diag_path else None,
            "diagnostics_mode": diag_mode,
            "diagnostics_export_date": diag_export,
            "overall_start": overall_start.isoformat() if overall_start else None,
            "overall_end": overall_end.isoformat() if overall_end else None,
            "allocation_total_candidates": alloc_total,
            "sum_jsonl_candidates": sum_jsonl,
            "sum_jsonl_vs_alloc_match": (
                (alloc_total == sum_jsonl) if isinstance(alloc_total, int) else None
            ),
            "systems": tri_rows,
            "generated_at": datetime.now().isoformat(),
        }

        tri_path = md_path.parent / "tri_sync_report.json"
        tri_path.write_text(
            json.dumps(tri_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Markdown: 3点同期セクションを追記
        lines.append("\n## 3点同期チェック（JSONL × 診断 × スクショ代表）\n")
        if chosen_diag_path:
            lines.append(
                f"- 使用診断スナップショット: `{chosen_diag_path}`"
                f" （export: {diag_export or '-'}, mode: {diag_mode or '-'}）\n"
            )
        if isinstance(alloc_total, int):
            lines.append(
                f"- allocation_total_candidates: {alloc_total} / "
                f"sum(jsonl per-system): {sum_jsonl}"
                f" → {'一致' if alloc_total == sum_jsonl else '不一致'}\n"
            )
        lines.append(
            "\n| System | JSONL candidates | Diagnostics ranked_top_n | status | start SS | complete SS |\n"
        )
        lines.append("|---|---:|---:|---|---|---|\n")
        for r in tri_rows:
            lines.append(
                f"| {r['system']} | {r.get('jsonl_candidates', '-')} | "
                f"{r.get('diagnostics_ranked_top_n', '-')} | "
                f"{r.get('status', '-')} | {r.get('start_ss', '-')} | "
                f"{r.get('complete_ss', '-')} |\n"
            )

        # 末尾にJSONリンク
        lines.append("\n---\n")
        lines.append(f"詳細JSON: [sync_analysis.json](./{output_path.name})\n")
        lines.append("3点同期: [tri_sync_report.json](./tri_sync_report.json)\n")

        md_path.write_text("".join(lines), encoding="utf-8")
        print(f"📝 Markdownレポート: {md_path}")

        # 追加の要約JSONを書き出し
        summary = {
            "events_total": len(events),
            "system_events": len(system_events),
            "screenshots": len(screenshots),
            "pipeline_complete": pc_flag,
            "overall": {
                "start": overall_start.isoformat() if overall_start else None,
                "end": overall_end.isoformat() if overall_end else None,
                "duration_sec": (
                    int((overall_end - overall_start).total_seconds())
                    if (overall_start and overall_end)
                    else None
                ),
            },
            "systems": per_system,
        }
        summary_path = md_path.parent / "sync_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"🧾 サマリーJSON: {summary_path}")
    except Exception as e:
        print(f"⚠️ Markdownレポート生成に失敗: {e}")


if __name__ == "__main__":
    analyze_sync()
