from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import time

import pytest
import requests


def _get_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_streamlit_smoke_headless() -> None:
    port = _get_free_port()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    cmd = [
        "streamlit",
        "run",
        str(root / "tests" / "app_smoke.py"),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    proc = subprocess.Popen(
        cmd, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    try:
        url = f"http://localhost:{port}"
        for _ in range(20):
            try:
                resp = requests.get(url)
                if resp.status_code == 200:
                    break
            except requests.ConnectionError:
                time.sleep(0.5)
        else:
            output = proc.stdout.read().decode("utf-8", errors="ignore")
            pytest.fail(f"Streamlit did not start: {output}")
        assert resp.status_code == 200
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
