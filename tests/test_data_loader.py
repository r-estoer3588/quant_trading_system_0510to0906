import pandas as pd

from common.data_loader import iter_load_symbols, load_symbols


def test_load_symbols_empty_when_not_cached(tmp_path, monkeypatch):
    # キャッシュディレクトリを空の一時フォルダに差し替え
    out = load_symbols(["NO_SUCH_SYMBOL"], cache_dir=tmp_path, max_workers=2)
    assert isinstance(out, dict)
    assert len(out) in (0,)


def test_iter_load_symbols_prefetch(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "Date": ["2020-01-01"],
            "Open": [1],
            "High": [1],
            "Low": [1],
            "Close": [1],
            "Volume": [100],
        }
    )
    df.to_csv(tmp_path / "AAA.csv", index=False)
    df.to_csv(tmp_path / "BBB.csv", index=False)

    monkeypatch.setattr(
        "common.data_loader.load_base_cache",
        lambda symbol, rebuild_if_missing=True: None,
    )

    loaded = dict(
        iter_load_symbols(
            ["AAA", "BBB"],
            cache_dir=tmp_path,
            max_workers=2,
            prefetch=1,
        ),
    )
    assert set(loaded.keys()) == {"AAA", "BBB"}
    for df_loaded in loaded.values():
        assert not df_loaded.empty
