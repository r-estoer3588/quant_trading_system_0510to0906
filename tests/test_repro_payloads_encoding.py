from __future__ import annotations

import json
from pathlib import Path


def test_repro_jsons_are_utf8_and_parse():
    repo_root = Path(__file__).resolve().parents[1]
    payload_dir = repo_root / "repro_payloads"
    assert payload_dir.exists(), "repro_payloads directory not found"
    jsons = list(sorted(payload_dir.glob("*.json")))
    assert jsons, "no JSON files found in repro_payloads"
    for p in jsons:
        data = p.read_bytes()
        # ensure decodable as UTF-8
        try:
            s = data.decode("utf-8")
        except Exception as e:
            raise AssertionError(f"{p} is not UTF-8 decodable: {e}")
        # ensure valid JSON
        try:
            json.loads(s)
        except Exception as e:
            raise AssertionError(f"{p} failed to parse as JSON: {e}")
