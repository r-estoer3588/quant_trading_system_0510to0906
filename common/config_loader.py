from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def load_config_file(path: Path | str | None = None) -> Dict[str, Any]:
    """YAML/JSON を自動判別して辞書として返すシンプルなローダー。
    - path 未指定時: config/config.json → config/config.yaml の順で探索
    - フォーマット不明/読込失敗時は {} を返す
    """
    root = Path(__file__).resolve().parents[1]
    if path is None:
        candidates = [root / "config" / "config.json", root / "config" / "config.yaml"]
    else:
        candidates = [Path(path)]

    for p in candidates:
        try:
            if not p.exists():
                continue
            if p.suffix.lower() in {".yaml", ".yml"}:
                if yaml is None:
                    continue
                with p.open("r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            elif p.suffix.lower() == ".json":
                with p.open("r", encoding="utf-8") as f:
                    return json.load(f) or {}
        except Exception:
            continue
    return {}


__all__ = ["load_config_file"]

