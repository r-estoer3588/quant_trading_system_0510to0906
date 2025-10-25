"""Poll Streamlit URL until ready, then run capture_ui_screenshot.py with given args.
Usage:
    python tools/run_capture_after_ready.py --url http://localhost:8501 --timeout 60 -- --output ... --click-button "Generate Signals" ...
Note: Arguments after '--' are passed to capture_ui_screenshot.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request


def wait_for_url(url: str, timeout: int) -> bool:
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                return True
        except Exception:
            time.sleep(1)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--timeout", type=int, default=60, help="seconds to wait for URL")
    parser.add_argument("--venv-python", default=None, help="path to python executable (uses sys.executable if not provided)")
    parser.add_argument("--capture-script", default="tools/capture_ui_screenshot.py")
    parser.add_argument("--wait-after-ready", type=int, default=2, help="seconds to wait a bit after URL ready before capture")
    parser.add_argument("--retry-capture", type=int, default=1, help="how many times to retry capture on failure")
    parser.add_argument("--", dest="_", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    url = args.url
    timeout = args.timeout
    print(f"Waiting up to {timeout}s for {url} ...")
    ok = wait_for_url(url, timeout)
    if not ok:
        print("URL not ready within timeout")
        return 2
    print("URL ready, sleeping a bit...")
    time.sleep(args.wait_after_ready)

    capture_cmd = []
    py = args.venv_python or sys.executable
    capture_cmd.append(py)
    capture_cmd.append(args.capture_script)
    # append remaining args passed after --
    if args._:
        capture_cmd.extend(args._)

    attempt = 0
    while attempt < max(1, args.retry_capture):
        attempt += 1
        print(f"Running capture (attempt {attempt}): {' '.join(capture_cmd)}")
        try:
            rc = subprocess.call(capture_cmd)
            if rc == 0:
                print("Capture succeeded")
                return 0
            else:
                print(f"Capture exited with code {rc}")
        except Exception as e:
            print(f"Capture execution failed: {e}")
        if attempt < args.retry_capture:
            print("Retrying capture after short wait...")
            time.sleep(3)
    print("All capture attempts failed")
    return 3


if __name__ == '__main__':
    raise SystemExit(main())
