from common.data_loader import load_symbols


def test_load_symbols_empty_when_not_cached(tmp_path, monkeypatch):
    # キャッシュディレクトリを空の一時フォルダに差し替え
    out = load_symbols(["NO_SUCH_SYMBOL"], cache_dir=tmp_path, max_workers=2)
    assert isinstance(out, dict)
    assert len(out) in (0, )

