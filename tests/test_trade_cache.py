import importlib
import json

import common.trade_cache as trade_cache


def test_store_and_pop_entry(tmp_path, monkeypatch):
    tmp = tmp_path / "cache.json"
    monkeypatch.setattr(trade_cache, "TRADE_CACHE_PATH", tmp)
    importlib.reload(trade_cache)

    trade_cache.store_entry("AAA", "2024-01-01", 100.0, path=tmp)
    data = json.loads(tmp.read_text())
    assert data["AAA"]["entry_price"] == 100.0

    info = trade_cache.pop_entry("AAA", path=tmp)
    assert info == {"entry_date": "2024-01-01", "entry_price": 100.0}
    assert trade_cache.pop_entry("AAA", path=tmp) is None
    assert tmp.read_text() == "{}"
