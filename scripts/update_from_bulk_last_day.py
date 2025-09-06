import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY")


def fetch_bulk_last_day():
    url = f"https://eodhistoricaldata.com/api/eod-bulk-last-day/US?api_token={API_KEY}&fmt=json"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        print("Error fetching bulk data:", r.status_code)
        return None
    return pd.DataFrame(r.json())


def append_to_cache(df, output_dir="data_cache"):
    count = 0
    df["date"] = pd.to_datetime(df["date"])
    for _, row in df.iterrows():
        symbol = row["code"]
        path = os.path.join(output_dir, f"{symbol}.csv")
        new_row = {
            "Date": row["date"],
            "Open": row["open"],
            "High": row["high"],
            "Low": row["low"],
            "Close": row["close"],
            "AdjClose": row["adjusted_close"],
            "Volume": row["volume"],
        }
        new_df = pd.DataFrame([new_row]).set_index("Date")
        if os.path.exists(path):
            try:
                existing = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
                if new_row["Date"] not in existing.index:
                    updated = pd.concat([existing, new_df]).sort_index()
                    updated.to_csv(path)
                    count += 1
            except Exception as e:
                print(f"{symbol}: error appending - {e}")
        else:
            new_df.to_csv(path)
            count += 1
    print(f"✅ {count} files updated.")


def main():
    df = fetch_bulk_last_day()
    if df is None or df.empty:
        print("No data to update.")
        return
    append_to_cache(df)


if __name__ == "__main__":
    main()
