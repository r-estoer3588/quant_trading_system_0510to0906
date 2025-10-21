import json
from pathlib import Path


def test_repro_payloads_json_are_utf8_and_valid():
    p = Path("repro_payloads")
    assert p.exists(), "repro_payloads directory must exist for this test"
    for f in p.glob("*.json"):
        # Ensure file bytes are UTF-8 decodable
        data = f.read_bytes()
        data.decode("utf-8")
        # Ensure JSON is parseable
        json.loads(data.decode("utf-8"))
