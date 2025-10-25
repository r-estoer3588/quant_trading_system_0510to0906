import pandas as pd

from scripts.run_all_systems_today import compute_today_signals


def make_df(n=260):
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "date": dates,
            "Open": [100.0] * n,
            "High": [101.0] * n,
            "Low": [99.0] * n,
            "Close": [100.0] * n,
            "Volume": [2_000_000] * n,
        }
    )
    return df


def main():
    symdata = {"AAA": make_df(), "BBB": make_df(), "SPY": make_df()}
    final, per_sys = compute_today_signals(
        ["AAA", "BBB", "SPY"],
        save_csv=False,
        notify=False,
        symbol_data=symdata,
        parallel=False,
        log_callback=None,
        progress_callback=lambda a, b, c: None,
    )
    print("FINAL:", final.shape, list(final.columns)[:10])
    summary = {
        k: (v.shape if hasattr(v, "shape") else None) for k, v in per_sys.items()
    }
    print("PER_SYSTEM:", summary)


if __name__ == "__main__":
    main()
