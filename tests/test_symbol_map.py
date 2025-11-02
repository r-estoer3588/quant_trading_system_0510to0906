"""Unit tests for common.symbol_map helpers."""

from pathlib import Path

from common.symbol_map import (
    coerce_system_list,
    dump_symbol_system_map,
    load_symbol_system_map,
    normalize_symbol_system_map,
    resolve_primary_system,
    update_primary_system,
)


def test_coerce_system_list_basic() -> None:
    assert coerce_system_list("system3") == ["system3"]
    assert coerce_system_list(["system3", "system5", "system3"]) == [
        "system3",
        "system5",
    ]
    # nested and dict forms
    assert coerce_system_list({"primary": "system2", "other": ["system6"]}) == [
        "system2",
        "system6",
    ]


def test_resolve_primary_system() -> None:
    assert resolve_primary_system(["system5", "system3"]) == "system5"
    assert resolve_primary_system("system4") == "system4"
    assert resolve_primary_system([]) is None


def test_normalize_symbol_system_map() -> None:
    raw = {"aapl": "system3", "msft": ["system5", "system3"]}
    norm = normalize_symbol_system_map(raw)
    assert norm == {"AAPL": ["system3"], "MSFT": ["system5", "system3"]}


def test_coerce_system_list_all_token() -> None:
    # expand when ensure_all=True (default)
    out = coerce_system_list("all", ensure_all=True)
    assert isinstance(out, list) and len(out) >= 3
    # keep literal when ensure_all=False
    assert coerce_system_list("all", ensure_all=False) == ["all"]


def test_update_primary_system_moves_and_dedups() -> None:
    existing = ["system5", "system3", "system5"]
    updated = update_primary_system(existing, "system3")
    assert updated[0] == "system3"
    assert updated.count("system3") == 1
    assert updated.count("system5") == 1


def test_dump_and_load_symbol_map_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "symbol_system_map.json"
    mapping = {"AAPL": ["system3", "system5"], "MSFT": ["system4"]}
    dump_symbol_system_map(mapping, path=path)
    loaded = load_symbol_system_map(path)
    assert loaded == mapping
