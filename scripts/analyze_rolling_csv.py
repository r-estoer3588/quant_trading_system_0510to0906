#!/usr/bin/env python3
"""
簡易 CSV 解析スクリプト
- 対象: data_cache/rolling/ARBEW.csv
- 出力: HV50 の NaN 連続区間、HV50 の NaN 比率、Close が 0 または NaN の行一覧
"""
import csv

# unused imports removed

INPUT = "data_cache/rolling/ARBEW.csv"


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def parse_float(s):
    if s is None or s == "":
        return None
    try:
        v = float(s)
        if v != v:
            return None
        return v
    except Exception:
        return None


def analyze(rows):
    hv_col = None
    close_col = None
    # find HV50 and Close columns (case-insensitive)
    header = rows[0].keys()
    for h in header:
        if h.lower() in ("hv50", "hv_50"):
            hv_col = h
        if h.lower() == "close":
            close_col = h
    if hv_col is None:
        # try last column name 'HV50'
        hv_col = next((h for h in header if h.lower().startswith("hv")), None)
    if close_col is None:
        close_col = "close"

    total = len(rows)
    hv_nan_count = 0
    hv_nan_runs = []  # list of (start_idx, end_idx) inclusive, 0-based
    in_run = False
    run_start = None

    close_zero_or_nan = []  # list of (date, close_value)

    for i, r in enumerate(rows):
        date = r.get("date") or r.get("Date") or ""
        hv_raw = r.get(hv_col, "") if hv_col else ""
        hv = parse_float(hv_raw)
        if hv is None:
            hv_nan_count += 1
            if not in_run:
                in_run = True
                run_start = i
        else:
            if in_run:
                hv_nan_runs.append((run_start, i - 1))
                in_run = False
                run_start = None
        c_raw = r.get(close_col, "")
        c = parse_float(c_raw)
        if c is None or c == 0.0:
            close_zero_or_nan.append((date, c_raw))

    if in_run:
        hv_nan_runs.append((run_start, total - 1))

    hv_nan_ratio = hv_nan_count / total if total else 0

    # convert runs to readable ranges
    hv_nan_ranges = []
    for s, e in hv_nan_runs:
        hv_nan_ranges.append(
            {
                "start_index": s,
                "end_index": e,
                "start_date": rows[s].get("date"),
                "end_date": rows[e].get("date"),
                "length": e - s + 1,
            }
        )

    return {
        "total_rows": total,
        "hv_nan_count": hv_nan_count,
        "hv_nan_ratio": hv_nan_ratio,
        "hv_nan_ranges": hv_nan_ranges,
        "close_zero_or_nan": close_zero_or_nan,
        "hv_col": hv_col,
        "close_col": close_col,
    }


def main():
    rows = read_rows(INPUT)
    res = analyze(rows)
    print(f"File: {INPUT}")
    print(f"Total rows: {res['total_rows']}")
    print(f"HV column: {res['hv_col']}")
    print(f"Close column: {res['close_col']}")
    print(f"HV NaN count: {res['hv_nan_count']} ({res['hv_nan_ratio']:.2%})")
    if res["hv_nan_ranges"]:
        print("HV NaN ranges (start_date → end_date) and lengths:")
        for r in res["hv_nan_ranges"]:
            print(f" - {r['start_date']} -> {r['end_date']}, len={r['length']}")
    else:
        print("No HV NaN ranges")
    if res["close_zero_or_nan"]:
        print("Rows with Close==0 or NaN (date, raw_close):")
        for d, v in res["close_zero_or_nan"]:
            print(f" - {d}: {v}")
    else:
        print("No Close==0 or NaN rows found")


if __name__ == "__main__":
    main()
