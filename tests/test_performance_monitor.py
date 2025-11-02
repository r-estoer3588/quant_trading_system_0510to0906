"""performance_monitor.py のテスト。

psutil が利用可能な環境でのみ実行。
未インストール時は自動スキップ。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from common.performance_monitor import (
    PSUTIL_AVAILABLE,
    PerformanceMonitor,
    enable_global_monitor,
    get_global_monitor,
)


@pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
class TestPerformanceMonitor:
    """PerformanceMonitor の基本機能テスト。"""

    def test_monitor_initialization(self):
        """モニター初期化のテスト。"""
        monitor = PerformanceMonitor(enabled=True)
        assert monitor.enabled is True
        assert monitor.process is not None
        assert len(monitor.phases) == 0
        assert monitor.system_info  # システム情報が取得されている

    def test_monitor_disabled(self):
        """無効化されたモニターのテスト。"""
        monitor = PerformanceMonitor(enabled=False)
        assert monitor.enabled is False

        with monitor.measure("test_phase"):
            time.sleep(0.01)

        # 無効時は何も記録されない
        assert len(monitor.phases) == 0

    def test_measure_single_phase(self):
        """単一フェーズの測定テスト。"""
        monitor = PerformanceMonitor(enabled=True)

        with monitor.measure("test_phase"):
            time.sleep(0.05)  # 50ms

        assert len(monitor.phases) == 1
        phase = monitor.phases[0]
        assert phase.name == "test_phase"
        assert phase.duration_sec is not None
        assert phase.duration_sec >= 0.05  # 少なくとも50ms以上
        assert phase.memory_delta_mb is not None
        assert phase.cpu_avg_percent is not None

    def test_measure_multiple_phases(self):
        """複数フェーズの測定テスト。"""
        monitor = PerformanceMonitor(enabled=True)

        with monitor.measure("phase1"):
            _ = [i**2 for i in range(1000)]  # 軽い処理

        with monitor.measure("phase2"):
            time.sleep(0.02)

        with monitor.measure("phase3"):
            _ = "test" * 10000  # メモリ割り当て

        assert len(monitor.phases) == 3
        assert [p.name for p in monitor.phases] == ["phase1", "phase2", "phase3"]

        # 各フェーズに duration が記録されている
        for phase in monitor.phases:
            assert phase.duration_sec is not None
            assert phase.duration_sec >= 0

    def test_get_report(self):
        """レポート生成のテスト。"""
        monitor = PerformanceMonitor(enabled=True)

        with monitor.measure("test_phase"):
            time.sleep(0.01)

        report = monitor.get_report()

        assert report["enabled"] is True
        assert report["psutil_available"] is True
        assert "timestamp" in report
        assert "system_info" in report
        assert len(report["phases"]) == 1

        phase_data = report["phases"][0]
        assert phase_data["name"] == "test_phase"
        assert "duration_sec" in phase_data
        assert "memory_delta_mb" in phase_data
        assert "cpu_avg_percent" in phase_data
        assert "io_read_delta_mb" in phase_data
        assert "io_write_delta_mb" in phase_data

        # サマリーが含まれている
        assert "summary" in report
        summary = report["summary"]
        assert "total_duration_sec" in summary
        assert "peak_memory_mb" in summary

    def test_save_report(self, tmp_path: Path):
        """レポート保存のテスト。"""
        monitor = PerformanceMonitor(enabled=True)

        with monitor.measure("test_phase"):
            time.sleep(0.01)

        output_path = tmp_path / "perf_report.json"
        monitor.save_report(output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "test_phase" in content
        assert "duration_sec" in content

    def test_memory_tracking(self):
        """メモリトラッキングのテスト。"""
        monitor = PerformanceMonitor(enabled=True)

        with monitor.measure("memory_test"):
            # 約10MBのメモリ割り当て
            _ = bytearray(10 * 1024 * 1024)

        phase = monitor.phases[0]
        assert phase.memory_delta_mb is not None
        # メモリが増加しているはず（少なくとも5MB以上）
        assert phase.memory_delta_mb > 5.0

    def test_global_monitor(self):
        """グローバルモニターのテスト。"""
        monitor = enable_global_monitor(enabled=True)
        assert get_global_monitor() is monitor
        assert monitor.enabled is True


class TestPerformanceMonitorDisabled:
    """psutil未インストール時の動作テスト。"""

    def test_monitor_disabled_without_psutil(self):
        """psutil 未インストール時は自動的に無効化。"""
        # 実際のPSUTIL_AVAILABLEに関わらず、enabled=Falseで初期化
        monitor = PerformanceMonitor(enabled=False)
        assert monitor.enabled is False

    def test_disabled_monitor_report(self):
        """無効モニターのレポートは空。"""
        monitor = PerformanceMonitor(enabled=False)

        with monitor.measure("test_phase"):
            time.sleep(0.01)

        report = monitor.get_report()
        assert report["enabled"] is False
        assert len(report["phases"]) == 0
        assert report["summary"] == {}


@pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
class TestResourceSnapshot:
    """ResourceSnapshot の動作確認。"""

    def test_snapshot_creation(self):
        """スナップショット作成のテスト。"""
        monitor = PerformanceMonitor(enabled=True)
        snapshot = monitor._take_snapshot()

        assert snapshot is not None
        assert snapshot.timestamp > 0
        assert snapshot.memory_rss_mb > 0
        assert snapshot.memory_vms_mb > 0
        assert snapshot.memory_percent > 0
        assert snapshot.num_threads > 0
        # I/Oカウンターはプラットフォーム依存（0でもOK）
        assert snapshot.io_read_bytes >= 0
        assert snapshot.io_write_bytes >= 0
