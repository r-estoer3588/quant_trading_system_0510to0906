"""拡張パフォーマンスモニター - メモリ・CPU・ディスクI/O測定。

Phase 6で導入。既存のperf_snapshotに加えて、以下を測定：
- メモリ使用量（RSS, VMS）
- CPU使用率（プロセス単位、システム全体）
- ディスクI/O（読み書きバイト数）
- スレッド/プロセス数

使用例:
    from common.performance_monitor import PerformanceMonitor

    monitor = PerformanceMonitor(enabled=True)
    with monitor.measure("phase_name"):
        # 処理実行
        pass

    # 結果取得
    report = monitor.get_report()
    monitor.save_report("logs/perf/detailed_metrics.json")
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

if TYPE_CHECKING:
    import psutil

try:
    import psutil  # type: ignore[import]

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None  # type: ignore[assignment]


@dataclass
class ResourceSnapshot:
    """単一時点のリソーススナップショット。"""

    timestamp: float
    memory_rss_mb: float  # Resident Set Size (物理メモリ)
    memory_vms_mb: float  # Virtual Memory Size
    memory_percent: float  # メモリ使用率（%）
    cpu_percent: float  # CPU使用率（%）
    num_threads: int  # スレッド数
    io_read_bytes: int  # 読み込みバイト数（累積）
    io_write_bytes: int  # 書き込みバイト数（累積）


@dataclass
class PhaseMetrics:
    """フェーズごとのパフォーマンスメトリクス。"""

    name: str
    start_time: float
    end_time: Optional[float] = None
    start_snapshot: Optional[ResourceSnapshot] = None
    end_snapshot: Optional[ResourceSnapshot] = None

    # 計算値
    duration_sec: Optional[float] = None
    memory_delta_mb: Optional[float] = None  # メモリ増減
    memory_peak_mb: Optional[float] = None  # ピークメモリ
    cpu_avg_percent: Optional[float] = None  # 平均CPU使用率
    io_read_delta_mb: Optional[float] = None  # 読み込み量
    io_write_delta_mb: Optional[float] = None  # 書き込み量

    def finalize(self) -> None:
        """終了スナップショットから計算値を導出。"""
        if self.end_time is None or self.start_snapshot is None or self.end_snapshot is None:
            return

        self.duration_sec = self.end_time - self.start_time
        self.memory_delta_mb = self.end_snapshot.memory_rss_mb - self.start_snapshot.memory_rss_mb
        self.memory_peak_mb = max(
            self.start_snapshot.memory_rss_mb, self.end_snapshot.memory_rss_mb
        )

        # CPU平均（簡易的に開始と終了の平均）
        self.cpu_avg_percent = (
            self.start_snapshot.cpu_percent + self.end_snapshot.cpu_percent
        ) / 2.0

        # I/O差分（MB単位）
        io_read_delta = self.end_snapshot.io_read_bytes - self.start_snapshot.io_read_bytes
        io_write_delta = self.end_snapshot.io_write_bytes - self.start_snapshot.io_write_bytes
        self.io_read_delta_mb = io_read_delta / (1024 * 1024)
        self.io_write_delta_mb = io_write_delta / (1024 * 1024)


class PerformanceMonitor:
    """拡張パフォーマンスモニター。

    メモリ、CPU、ディスクI/Oを詳細測定。
    psutil が利用可能な場合のみ有効化。
    """

    def __init__(self, enabled: bool = False):
        """
        Args:
            enabled: 測定を有効化するか（psutil 未インストール時は自動的に False）
        """
        self.enabled = enabled and PSUTIL_AVAILABLE
        self.process: Optional[Any] = None
        self.phases: List[PhaseMetrics] = []
        self.system_info: Dict[str, Any] = {}

        if self.enabled:
            try:
                self.process = psutil.Process(os.getpid())
                self._capture_system_info()
            except Exception:
                self.enabled = False

    def _capture_system_info(self) -> None:
        """システム情報を取得（一度だけ）。"""
        if not self.enabled or self.process is None:
            return

        try:
            self.system_info = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            }
        except Exception:
            self.system_info = {}

    def _take_snapshot(self) -> Optional[ResourceSnapshot]:
        """現在のリソース状態をスナップショット。"""
        if not self.enabled or self.process is None:
            return None

        try:
            mem_info = self.process.memory_info()
            mem_percent = self.process.memory_percent()
            cpu_percent = self.process.cpu_percent(interval=0.1)  # 0.1秒間隔で測定
            num_threads = self.process.num_threads()

            # ディスクI/O（プラットフォーム依存、未対応なら0）
            try:
                io_counters = self.process.io_counters()
                io_read = io_counters.read_bytes
                io_write = io_counters.write_bytes
            except (AttributeError, NotImplementedError):
                io_read = 0
                io_write = 0

            return ResourceSnapshot(
                timestamp=time.perf_counter(),
                memory_rss_mb=mem_info.rss / (1024 * 1024),
                memory_vms_mb=mem_info.vms / (1024 * 1024),
                memory_percent=mem_percent,
                cpu_percent=cpu_percent,
                num_threads=num_threads,
                io_read_bytes=io_read,
                io_write_bytes=io_write,
            )
        except Exception:
            return None

    @contextmanager
    def measure(self, phase_name: str) -> Iterator[None]:
        """フェーズを測定するコンテキストマネージャー。

        Args:
            phase_name: フェーズ名（例: "phase1_symbols", "phase2_load"）

        使用例:
            with monitor.measure("phase2_load"):
                # データロード処理
                pass
        """
        if not self.enabled:
            yield
            return

        # 開始時スナップショット
        start_snapshot = self._take_snapshot()
        start_time = time.perf_counter()

        phase = PhaseMetrics(
            name=phase_name,
            start_time=start_time,
            start_snapshot=start_snapshot,
        )

        try:
            yield
        finally:
            # 終了時スナップショット
            end_time = time.perf_counter()
            end_snapshot = self._take_snapshot()

            phase.end_time = end_time
            phase.end_snapshot = end_snapshot
            phase.finalize()

            self.phases.append(phase)

    def get_report(self) -> Dict[str, Any]:
        """パフォーマンスレポートを辞書形式で取得。

        Returns:
            {
                "enabled": bool,
                "psutil_available": bool,
                "timestamp": str (ISO8601),
                "system_info": {...},
                "phases": [
                    {
                        "name": str,
                        "duration_sec": float,
                        "memory_delta_mb": float,
                        "cpu_avg_percent": float,
                        "io_read_delta_mb": float,
                        "io_write_delta_mb": float,
                        ...
                    },
                    ...
                ],
                "summary": {
                    "total_duration_sec": float,
                    "total_memory_delta_mb": float,
                    "peak_memory_mb": float,
                    "total_io_read_mb": float,
                    "total_io_write_mb": float,
                }
            }
        """
        report = {
            "enabled": self.enabled,
            "psutil_available": PSUTIL_AVAILABLE,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_info": self.system_info,
            "phases": [],
            "summary": {},
        }

        if not self.enabled:
            return report

        # フェーズ詳細
        phases_data = []
        for phase in self.phases:
            phases_data.append(
                {
                    "name": phase.name,
                    "duration_sec": round(phase.duration_sec or 0.0, 6),
                    "memory_delta_mb": round(phase.memory_delta_mb or 0.0, 2),
                    "memory_peak_mb": round(phase.memory_peak_mb or 0.0, 2),
                    "cpu_avg_percent": round(phase.cpu_avg_percent or 0.0, 2),
                    "io_read_delta_mb": round(phase.io_read_delta_mb or 0.0, 2),
                    "io_write_delta_mb": round(phase.io_write_delta_mb or 0.0, 2),
                }
            )

        report["phases"] = phases_data

        # サマリー集計
        if self.phases:
            total_duration = sum(p.duration_sec or 0.0 for p in self.phases)
            total_memory_delta = sum(p.memory_delta_mb or 0.0 for p in self.phases)
            peak_memory = max(p.memory_peak_mb or 0.0 for p in self.phases)
            total_io_read = sum(p.io_read_delta_mb or 0.0 for p in self.phases)
            total_io_write = sum(p.io_write_delta_mb or 0.0 for p in self.phases)

            report["summary"] = {
                "total_duration_sec": round(total_duration, 6),
                "total_memory_delta_mb": round(total_memory_delta, 2),
                "peak_memory_mb": round(peak_memory, 2),
                "total_io_read_mb": round(total_io_read, 2),
                "total_io_write_mb": round(total_io_write, 2),
            }

        return report

    def save_report(self, output_path: str | Path) -> None:
        """レポートをJSON形式で保存。

        Args:
            output_path: 出力先パス（例: "logs/perf/detailed_metrics.json"）
        """
        if not self.enabled:
            return

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        report = self.get_report()
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def print_summary(self) -> None:
        """サマリーをコンソールに出力（デバッグ用）。"""
        if not self.enabled:
            print("PerformanceMonitor: 無効（psutil未インストールまたはenabled=False）")
            return

        report = self.get_report()
        summary = report.get("summary", {})

        print("\n" + "=" * 60)
        print("パフォーマンス測定サマリー")
        print("=" * 60)
        print(f"総実行時間: {summary.get('total_duration_sec', 0):.2f} 秒")
        print(f"メモリ増減: {summary.get('total_memory_delta_mb', 0):+.2f} MB")
        print(f"ピークメモリ: {summary.get('peak_memory_mb', 0):.2f} MB")
        print(f"ディスクI/O 読み込み: {summary.get('total_io_read_mb', 0):.2f} MB")
        print(f"ディスクI/O 書き込み: {summary.get('total_io_write_mb', 0):.2f} MB")
        print("=" * 60)

        if report.get("phases"):
            print("\nフェーズ別詳細:")
            print("-" * 60)
            for phase in report["phases"]:
                print(f"  {phase['name']}")
                print(f"    時間: {phase['duration_sec']:.2f}秒")
                print(f"    メモリ: {phase['memory_delta_mb']:+.2f}MB")
                print(f"    CPU平均: {phase['cpu_avg_percent']:.1f}%")
                io_read = phase["io_read_delta_mb"]
                io_write = phase["io_write_delta_mb"]
                print(f"    I/O: 読み{io_read:.2f}MB / 書き{io_write:.2f}MB")
            print("-" * 60)
        print()


# グローバルインスタンス（オプション）
_global_monitor: Optional[PerformanceMonitor] = None


def get_global_monitor() -> Optional[PerformanceMonitor]:
    """グローバルモニターを取得。"""
    return _global_monitor


def enable_global_monitor(enabled: bool = True) -> PerformanceMonitor:
    """グローバルモニターを有効化。

    Args:
        enabled: 測定を有効化

    Returns:
        PerformanceMonitor インスタンス
    """
    global _global_monitor
    _global_monitor = PerformanceMonitor(enabled=enabled)
    return _global_monitor
