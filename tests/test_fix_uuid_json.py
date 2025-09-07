from __future__ import annotations

import uuid

from fix_uuid_json import make_json_serializable


def test_make_json_serializable_converts_uuid_keys_and_values() -> None:
    k1 = uuid.uuid4()
    data = {k1: {"inner": [k1]}}
    result = make_json_serializable(data)
    assert list(result.keys()) == [str(k1)]
    assert result[str(k1)]["inner"] == [str(k1)]
