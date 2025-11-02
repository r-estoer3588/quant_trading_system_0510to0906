"""Wait until Streamlit URL ready then run the capture helper with fixed options.
This is a simple runner tailored for the current debugging session.
"""

from pathlib import Path
import subprocess
import sys
import time
import urllib.request

URL = "http://localhost:8501"
TIMEOUT = 60


def wait(url, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


if __name__ == "__main__":
    ok = wait(URL, TIMEOUT)
    if not ok:
        print(f"{URL} not ready after {TIMEOUT}s")
        sys.exit(2)
    print("URL ready, running capture helper...")
    py = sys.executable
    script = Path(__file__).resolve().parents[1] / "tools" / "capture_ui_screenshot.py"
    cmd = [
        str(py),
        str(script),
        "--url",
        URL,
        "--output",
        "results_images/today_signals_complete_after_fix4.png",
        "--click-button",
        "Generate Signals",
        "--wait-after-click",
        "960",
        "--wait-jsonl",
        "--wait-results",
        "--show-browser",
    ]
    print("CMD:", " ".join(cmd))
    rc = subprocess.call(cmd)
    print("capture exit code:", rc)
    sys.exit(rc)
