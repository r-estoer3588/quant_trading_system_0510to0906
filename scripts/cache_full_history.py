import os
import time
import pandas as pd
import requests
from dotenv import load_dotenv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
API_KEY = os.getenv("EODHD_API_KEY")


def get_all_symbols():
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    symbols = set()
    for url in urls:
        try:
            r = requests.get(url)
            lines = r.text.splitlines()
            for line in lines[1:]:
                if "|" in line:
                    parts = line.split("|")
                    if parts[0].isalpha():
                        symbols.add(parts[0])
        except Exception as e:
            print(f"Error fetching {url}: {e}")
    return sorted(symbols)


def fetch_history(symbol):
    url = f"https://eodhistoricaldata.com/api/eod/{symbol}.US?api_token={API_KEY}&period=d&fmt=json"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = (
            df.rename(
                columns={
                    "date": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "adjusted_close": "AdjClose",
                    "volume": "Volume",
                }
            )
            .set_index("Date")
            .sort_index()
        )
        return df
    except Exception as e:
        print(f"{symbol}: {e}")
        return None


RESERVED_WORDS = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def safe_filename(symbol):
    if symbol.upper() in RESERVED_WORDS:
        return symbol + "_RESV"
    return symbol


def save_history(symbol, output_dir):
    safe_symbol = safe_filename(symbol)
    filepath = os.path.join(output_dir, f"{safe_symbol}.csv")
    if os.path.exists(filepath):
        print(f"{symbol}: already cached")
        return
    df = fetch_history(symbol)
    if df is not None and not df.empty:
        df.to_csv(filepath)
        print(f"{symbol}: saved")
    else:
        print(f"{symbol}: failed")


def main():
    start_time = time.time()
    print("📅 開始:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    symbols = get_all_symbols()
    print(f"Total symbols: {len(symbols)}")
    output_dir = "data_cache"
    os.makedirs(output_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(save_history, s, output_dir) for s in symbols]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            time.sleep(1.2)
            if i % 1000 == 0:
                print("Sleeping 1 hour to avoid rate limits...")
                time.sleep(3600)

    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    print(f"✅ 完了！処理時間: {mins}分{secs}秒")


if __name__ == "__main__":
    main()
