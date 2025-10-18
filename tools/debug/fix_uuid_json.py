from __future__ import annotations

import json
from typing import Any
from uuid import UUID


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert UUID keys and values to strings for JSON serialization."""
    if isinstance(obj, dict):
        return {(str(k) if isinstance(k, UUID) else k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, UUID):
        return str(obj)
    return obj


def dumps(obj: Any, **kwargs: Any) -> str:
    """Serialize obj to JSON, converting UUID keys and values to strings."""
    return json.dumps(make_json_serializable(obj), **kwargs)


if __name__ == "__main__":
    import uuid

    sample = {uuid.uuid4(): {"id": uuid.uuid4()}}
    print(dumps(sample))
