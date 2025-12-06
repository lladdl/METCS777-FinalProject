#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pyspark.sql import SparkSession

DEFAULT_INPUT  = "gs://metcs777-term-project/Crimes_-_2001_to_Present_20251202.csv"
DEFAULT_OUTPUT = "gs://metcs777-term-project/Crimes_2001_2025_Parquet/"

USECOLS = [
    "id", "case number", "date", "block", "iucr", "primary type",
    "description", "location description", "arrest", "domestic",
    "beat", "district", "ward", "community area", "fbi code"
]

if len(sys.argv) == 3:
    INPUT_CSV, OUTPUT_PARQUET = sys.argv[1], sys.argv[2]
else:
    INPUT_CSV, OUTPUT_PARQUET = DEFAULT_INPUT, DEFAULT_OUTPUT

spark = (
    SparkSession.builder
    .appName("csv-to-parquet-case-insensitive")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate()
)

print(f"\nReading CSV from: {INPUT_CSV}")
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", False)   # faster; we're just converting
    .csv(INPUT_CSV)
)

def norm(s: str) -> str:
    return (s or "").strip().lower()

actual_cols = df.columns
norm_map = {norm(c): c for c in actual_cols}

selected_actual = [norm_map[c] for c in USECOLS if c in norm_map]

if selected_actual:
    print(f"Selecting {len(selected_actual)} columns by case-insensitive match.")
    df = df.select(*selected_actual)
else:
    print("No case-insensitive matches found for USECOLS; writing all columns instead.")
    print(f"Available columns: {actual_cols}")

count = df.count()
print(f"Loaded {count:,} rows.")
print(f"Writing Parquet to: {OUTPUT_PARQUET}")

(df.write
   .mode("overwrite")
   .parquet(OUTPUT_PARQUET))

print("Done.")
spark.stop()
