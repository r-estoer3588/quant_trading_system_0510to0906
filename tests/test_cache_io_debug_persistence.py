import pandas as pd

from common.cache_file_io import CacheFileIO


def test_attrs_debug_persist_to_column(tmp_path):
    io = CacheFileIO(settings=type("S", (), {"cache": None}))
    # create a simple df
    df = pd.DataFrame({"date": ["2020-01-01", "2020-01-02"], "close": [10, 11]})
    # put debug reasons in attrs
    df.attrs["_fdbg_reasons3"] = ["pass", "atr_missing"]

    p = tmp_path / "AAA.csv"
    io.write_dataframe_atomic(df, p)

    # Read back
    df2 = pd.read_csv(p)
    # column name should be present as _dbg_reasons3
    assert "_dbg_reasons3" in df2.columns
    assert df2["_dbg_reasons3"].tolist() == ["pass", "atr_missing"]
