"""
Test script to verify progress_events functionality.
"""

import os
import time
from pathlib import Path

# Test progress events
from common.progress_events import (
    emit_phase,
    emit_system_complete,
    emit_system_progress,
    emit_system_start,
    reset_progress_log,
)


def test_progress_events():
    """Test progress events functionality."""
    print("Testing progress events...")

    # Enable progress events
    os.environ["ENABLE_PROGRESS_EVENTS"] = "1"

    # Reset log
    reset_progress_log()

    # Simulate progress events
    emit_phase("initialization", "start", count=7)
    time.sleep(0.5)

    emit_system_start("system1", 150, strategy="momentum")
    time.sleep(0.2)

    emit_system_progress("system1", 50, 150)
    time.sleep(0.2)

    emit_system_progress("system1", 100, 150)
    time.sleep(0.2)

    emit_system_progress("system1", 150, 150)
    time.sleep(0.2)

    emit_system_complete("system1", 142, filtered=8)
    time.sleep(0.2)

    emit_phase("allocation", "start", systems=["system1", "system2"])
    time.sleep(0.2)

    emit_phase("allocation", "complete", final_positions=15)

    # Check if log file was created
    from config.settings import get_settings

    settings = get_settings()
    log_file = Path(settings.LOGS_DIR) / "progress_today.jsonl"

    if log_file.exists():
        print(f"✓ Progress log created: {log_file}")
        content = log_file.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        print(f"✓ Log contains {len(lines)} events")

        # Show last few events
        print("\nLatest events:")
        for line in lines[-3:]:
            import json

            try:
                event = json.loads(line)
                timestamp = event.get("timestamp", "").split("T")[-1].split(".")[0]
                event_type = event.get("event_type", "unknown")
                print(f"  {timestamp} - {event_type}")
            except json.JSONDecodeError:
                print(f"  Invalid JSON: {line[:50]}...")
    else:
        print("✗ Progress log not found")

    # Test render_digest_log
    try:
        import app_integrated

        # Create a mock container
        class MockContainer:
            def __init__(self):
                self.content = ""

            def markdown(self, content):
                self.content = content
                print("Rendered content preview:")
                print(content[:200] + "..." if len(content) > 200 else content)

            def info(self, message):
                print(f"INFO: {message}")

            def error(self, message):
                print(f"ERROR: {message}")

        container = MockContainer()
        app_integrated.render_digest_log(log_file, container)
        print("✓ render_digest_log executed successfully")

    except Exception as e:
        print(f"✗ render_digest_log failed: {e}")

    print("Progress events test completed!")


if __name__ == "__main__":
    test_progress_events()
