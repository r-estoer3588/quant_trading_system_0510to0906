"""
Tests for progress events functionality.
"""

import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from common.progress_events import (
    ProgressEventEmitter,
    emit_phase,
    emit_progress,
    emit_system_complete,
    emit_system_progress,
    emit_system_start,
    reset_progress_log,
)


class TestProgressEventEmitter:
    """Progress event emitter tests."""

    def test_singleton_pattern(self):
        """Test that ProgressEventEmitter follows singleton pattern."""
        emitter1 = ProgressEventEmitter()
        emitter2 = ProgressEventEmitter()
        assert emitter1 is emitter2

    @patch.dict(os.environ, {"ENABLE_PROGRESS_EVENTS": "0"})
    def test_disabled_by_environment(self):
        """Test that progress events can be disabled via environment."""
        # Create new instance to test environment variable
        ProgressEventEmitter._instance = None
        emitter = ProgressEventEmitter()
        assert not emitter.enabled

    @patch.dict(os.environ, {"ENABLE_PROGRESS_EVENTS": "1"})
    def test_enabled_by_environment(self):
        """Test that progress events are enabled via environment."""
        # Create new instance to test environment variable
        ProgressEventEmitter._instance = None
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("common.progress_events.get_settings") as mock_settings:
                settings_obj = type("Settings", (), {"LOGS_DIR": temp_dir})()
                mock_settings.return_value = settings_obj
                emitter = ProgressEventEmitter()
                assert emitter.enabled

    @patch.dict(os.environ, {"ENABLE_PROGRESS_EVENTS": "1"})
    def test_emit_event(self):
        """Test basic event emission to JSONL file."""
        ProgressEventEmitter._instance = None

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("common.progress_events.get_settings") as mock_settings:
                # Mock the settings object properly
                settings_obj = type("Settings", (), {"LOGS_DIR": temp_dir})()
                mock_settings.return_value = settings_obj

                emitter = ProgressEventEmitter()
                emitter.emit("test_event", {"key": "value"})

                # Check file was created and contains event
                log_file = Path(temp_dir) / "progress_today.jsonl"
                assert log_file.exists()

                lines = log_file.read_text(encoding="utf-8").strip().split("\n")
                assert len(lines) >= 2  # session_start + test_event

                # Parse last event (test_event)
                event = json.loads(lines[-1])
                assert event["event_type"] == "test_event"
                assert event["data"]["key"] == "value"
                assert event["level"] == "info"
                assert "timestamp" in event

    @patch.dict(os.environ, {"ENABLE_PROGRESS_EVENTS": "1"})
    def test_reset_clears_file(self):
        """Test that reset clears the log file."""
        ProgressEventEmitter._instance = None

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("common.progress_events.get_settings") as mock_settings:
                settings_obj = type("Settings", (), {"LOGS_DIR": temp_dir})()
                mock_settings.return_value = settings_obj

                emitter = ProgressEventEmitter()
                emitter.emit("test_event1")
                emitter.emit("test_event2")

                log_file = Path(temp_dir) / "progress_today.jsonl"
                lines_before = log_file.read_text(encoding="utf-8").strip().split("\n")
                assert len(lines_before) >= 3  # session_start + 2 test events

                emitter.reset()

                lines_after = log_file.read_text(encoding="utf-8").strip().split("\n")
                assert len(lines_after) == 1  # Only new session_start
                event = json.loads(lines_after[0])
                assert event["event_type"] == "session_start"

    @patch.dict(os.environ, {"ENABLE_PROGRESS_EVENTS": "1"})
    def test_system_convenience_methods(self):
        """Test system-specific convenience methods."""
        ProgressEventEmitter._instance = None

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("common.progress_events.get_settings") as mock_settings:
                settings_obj = type("Settings", (), {"LOGS_DIR": temp_dir})()
                mock_settings.return_value = settings_obj

                emitter = ProgressEventEmitter()

                # Test system start
                emitter.emit_system_start("system1", 100, {"extra": "data"})

                # Test system progress
                emitter.emit_system_progress("system1", 50, 100, {"stage": "processing"})

                # Test system complete
                emitter.emit_system_complete("system1", 95, {"filtered": 5})

                log_file = Path(temp_dir) / "progress_today.jsonl"
                lines = log_file.read_text(encoding="utf-8").strip().split("\n")
                assert len(lines) >= 4  # session_start + 3 system events

                # Check system_start event
                start_event = json.loads(lines[1])
                assert start_event["event_type"] == "system1_start"
                assert start_event["data"]["system"] == "system1"
                assert start_event["data"]["symbol_count"] == 100
                assert start_event["data"]["extra"] == "data"

                # Check system_progress event
                progress_event = json.loads(lines[2])
                assert progress_event["event_type"] == "system1_progress"
                assert progress_event["data"]["processed"] == 50
                assert progress_event["data"]["total"] == 100
                assert progress_event["data"]["percentage"] == 50.0
                assert progress_event["data"]["stage"] == "processing"

                # Check system_complete event
                complete_event = json.loads(lines[3])
                assert complete_event["event_type"] == "system1_complete"
                assert complete_event["data"]["final_count"] == 95
                assert complete_event["data"]["filtered"] == 5

    @patch.dict(os.environ, {"ENABLE_PROGRESS_EVENTS": "1"})
    def test_phase_convenience_method(self):
        """Test phase convenience method."""
        ProgressEventEmitter._instance = None

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("common.progress_events.get_settings") as mock_settings:
                settings_obj = type("Settings", (), {"LOGS_DIR": temp_dir})()
                mock_settings.return_value = settings_obj

                emitter = ProgressEventEmitter()

                emitter.emit_phase("allocation", "start", {"systems": 7})
                emitter.emit_phase("allocation", "complete", {"final_positions": 10})

                log_file = Path(temp_dir) / "progress_today.jsonl"
                lines = log_file.read_text(encoding="utf-8").strip().split("\n")
                assert len(lines) >= 3  # session_start + 2 phase events

                # Check phase start event
                start_event = json.loads(lines[1])
                assert start_event["event_type"] == "phase_allocation_start"
                assert start_event["data"]["phase"] == "allocation"
                assert start_event["data"]["status"] == "start"
                assert start_event["data"]["systems"] == 7

                # Check phase complete event
                complete_event = json.loads(lines[2])
                assert complete_event["event_type"] == "phase_allocation_complete"
                assert complete_event["data"]["status"] == "complete"
                assert complete_event["data"]["final_positions"] == 10

    @patch.dict(os.environ, {"ENABLE_PROGRESS_EVENTS": "0"})
    def test_disabled_emit_does_nothing(self):
        """Test that emit does nothing when disabled."""
        ProgressEventEmitter._instance = None

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("common.progress_events.get_settings") as mock_settings:
                settings_obj = type("Settings", (), {"LOGS_DIR": temp_dir})()
                mock_settings.return_value = settings_obj

                emitter = ProgressEventEmitter()
                emitter.emit("test_event", {"key": "value"})

                # No file should be created
                log_file = Path(temp_dir) / "progress_today.jsonl"
                assert not log_file.exists()


class TestGlobalFunctions:
    """Test global convenience functions."""

    @patch("common.progress_events._progress_emitter")
    def test_global_emit_progress(self, mock_emitter):
        """Test global emit_progress function."""
        emit_progress("global_test", {"data": "test"}, "warning")
        mock_emitter.emit.assert_called_once_with("global_test", {"data": "test"}, "warning")

    @patch("common.progress_events._progress_emitter")
    def test_global_reset_progress_log(self, mock_emitter):
        """Test global reset_progress_log function."""
        reset_progress_log()
        mock_emitter.reset.assert_called_once()

    @patch("common.progress_events._progress_emitter")
    def test_global_emit_system_start(self, mock_emitter):
        """Test global emit_system_start function."""
        emit_system_start("system1", 100, extra="data")
        mock_emitter.emit_system_start.assert_called_once_with("system1", 100, {"extra": "data"})

    @patch("common.progress_events._progress_emitter")
    def test_global_emit_system_progress(self, mock_emitter):
        """Test global emit_system_progress function."""
        emit_system_progress("system1", 50, 100, stage="mid")
        mock_emitter.emit_system_progress.assert_called_once_with("system1", 50, 100, {"stage": "mid"})

    @patch("common.progress_events._progress_emitter")
    def test_global_emit_system_complete(self, mock_emitter):
        """Test global emit_system_complete function."""
        emit_system_complete("system1", 95, filtered=5)
        mock_emitter.emit_system_complete.assert_called_once_with("system1", 95, {"filtered": 5})

    @patch("common.progress_events._progress_emitter")
    def test_global_emit_phase(self, mock_emitter):
        """Test global emit_phase function."""
        emit_phase("allocation", "start", count=7)
        mock_emitter.emit_phase.assert_called_once_with("allocation", "start", {"count": 7})
