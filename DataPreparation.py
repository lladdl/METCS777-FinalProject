#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 00:00:03 2025

@author: lukeladd
"""

#!/usr/bin/env python3
"""
Prep Chicago crime data:
- Load full CSV
- Clean + engineer features
- Take stratified sample (~300k rows)
- Save modeling-ready CSV
"""

import pandas as pd

# === CONFIG ===
CSV_PATH = "Downloads/Chicago Crime Data.csv"    
OUTPUT_PATH = "chicago_crime_sample_prepped.csv"
SAMPLE_SIZE = 85000000                   
RANDOM_STATE = 42


def load_and_sample(csv_path=CSV_PATH,
                    sample_size=SAMPLE_SIZE,
                    random_state=RANDOM_STATE):
    df = pd.read_csv(csv_path)

    # Drop rows with missing Arrest
    df = df.dropna(subset=["Arrest"])
    # Ensure Arrest and Domestic are 0/1 ints
    df["Arrest"] = df["Arrest"].astype(int)
    df["Domestic"] = df["Domestic"].astype(int)

    # --- Parse Date and engineer time features ---
    print("Parsing dates and creating time features...")
    df["DateTime"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["DateTime"])

    df["Year"] = df["DateTime"].dt.year
    df["Month"] = df["DateTime"].dt.month
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek  
    df["Hour"] = df["DateTime"].dt.hour
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    # --- Seasonality feature ---
    df["Season"] = df["Month"].map({
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall"
})


    # --- Block frequency: how "busy" a block is ---
    print("Computing block frequencies...")
    block_counts = df["Block"].value_counts()
    df["BlockFreq"] = df["Block"].map(block_counts)

    if sample_size is not None and sample_size < len(df):
        frac = sample_size / len(df)
        print(f"Taking stratified sample of approx {sample_size} rows "
              f"(fraction={frac:.4f})...")

        df_sample = (
            df.groupby("Arrest", group_keys=False)
              .apply(lambda x: x.sample(frac=frac,
                                        random_state=random_state))
        )
    else:
        print("Sample size >= total rows; using full dataset.")
        df_sample = df

    cols_to_keep = [
        "Arrest",
        "Primary Type",
        "Description",
        "Location Description",
        "Domestic",
        "Beat",
        "Block",
        "BlockFreq",
        "District",
        "Ward",
        "Community Area",
        "Year",
        "Month",
        "DayOfWeek",
        "Hour",
        "IsWeekend",
        "Season",
    ]

    cols_to_keep = [c for c in cols_to_keep if c in df_sample.columns]
    df_sample = df_sample[cols_to_keep].copy()

    print(f"Final sample shape: {df_sample.shape}")
    print(f"Arrest rate: {df_sample['Arrest'].mean():.3f}")

    print(f"Saving prepped sample to {OUTPUT_PATH} ...")
    df_sample.to_csv(OUTPUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    load_and_sample()
