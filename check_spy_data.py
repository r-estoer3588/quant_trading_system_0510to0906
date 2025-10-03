#!/usr/bin/env python3

import os
import sys

import pandas as pd

# プロジェクトルートディレクトリを追加
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("SPY data analysis:")

# rolling SPY.csv をチェック
rolling_spy = "data_cache/rolling/SPY.csv"
if os.path.exists(rolling_spy):
    df = pd.read_csv(rolling_spy)
    print("\nRolling SPY.csv stats:")
    print(f"Total rows: {len(df)}")
    print(f"Null dates: {df['date'].isnull().sum()}")
    print(f"Duplicate dates: {df['date'].duplicated().sum()}")
    print("First 10 dates:")
    print(df["date"].head(10).tolist())
    print("Last 10 dates:")
    print(df["date"].tail(10).tolist())

    # 空白行を除いたデータ
    clean_df = df.dropna(subset=["date"])
    print(f"Clean rows (non-null dates): {len(clean_df)}")
    print(f"Clean duplicate dates: {clean_df['date'].duplicated().sum()}")
else:
    print("Rolling SPY.csv not found")

# full_backup にもSPY.csvがあるかチェック
full_backup_spy = "data_cache/full_backup/SPY.csv"
if os.path.exists(full_backup_spy):
    print("\nFull backup SPY.csv exists")
    df_full = pd.read_csv(full_backup_spy)
    print(f"Full backup rows: {len(df_full)}")
    print(f"Full backup null dates: {df_full['date'].isnull().sum()}")
    print(f"Full backup duplicate dates: {df_full['date'].duplicated().sum()}")
    print("First 5 dates:")
    print(df_full["date"].head(5).tolist())
    print("Sample data:")
    print(df_full.head(3))

# base にもSPY.csvがあるかチェック
base_spy = "data_cache/base/SPY.csv"
if os.path.exists(base_spy):
    print("\nBase SPY.csv exists")
    df_base = pd.read_csv(base_spy)
    print(f"Base rows: {len(df_base)}")
    print(f"Base null dates: {df_base['date'].isnull().sum()}")
    print(f"Base duplicate dates: {df_base['date'].duplicated().sum()}")
else:
    print("Base SPY.csv not found")
