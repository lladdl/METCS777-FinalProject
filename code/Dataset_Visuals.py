#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizations for Chicago crime arrest analysis using prepped sample data.

Assumes you have already created "chicago_crime_sample_prepped.csv"
with columns like:
- Arrest, Primary Type, Description, Location Description
- Domestic, Beat, Block, BlockFreq, District, Ward, Community Area
- Year, Month, DayOfWeek, Hour, IsWeekend, (optional) Season
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "chicago_crime_sample_prepped.csv"

sns.set(style="whitegrid")


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Make sure Arrest is numeric 0/1
    df["Arrest"] = df["Arrest"].astype(int)

    # ==============================
    # 1. Arrest percentage by BEAT
    # ==============================
    if "Beat" in df.columns:
        beat_rate = df.groupby("Beat")["Arrest"].mean().reset_index()
        beat_rate["Arrest"] *= 100

        beat_rate_sorted = beat_rate.sort_values("Arrest", ascending=False)

        plt.figure(figsize=(16, 6))
        sns.barplot(
            data=beat_rate_sorted,
            x="Beat",
            y="Arrest"
        )
        plt.xticks(rotation=90)
        plt.title("Arrest Percentage by Beat")
        plt.ylabel("Arrest Rate (%)")
        plt.xlabel("Beat")
        plt.tight_layout()
        plt.savefig("arrest_rate_by_beat.png", dpi=200)
        plt.show()
        print("Saved: arrest_rate_by_beat.png")

    # ======================================
    # 2. Top 20 busiest Blocks by arrest %
    # ======================================
    if "Block" in df.columns:
        block_stats = (
            df.groupby("Block")
              .agg(
                  arrest_rate=("Arrest", "mean"),
                  count=("Arrest", "size")
              )
              .sort_values("count", ascending=False)
        )

        # Only consider blocks with enough observations
        top_blocks = block_stats[block_stats["count"] > 200].head(20)
        top_blocks["arrest_rate"] *= 100

        plt.figure(figsize=(10, 6))
        sns.barplot(
            y=top_blocks.index,
            x=top_blocks["arrest_rate"]
        )
        plt.title("Top 20 Busiest Blocks — Arrest Percentage")
        plt.xlabel("Arrest Rate (%)")
        plt.ylabel("Block")
        plt.tight_layout()
        plt.savefig("top_blocks_arrest_rate.png", dpi=200)
        plt.show()
        print("Saved: top_blocks_arrest_rate.png")

    # ===============================================
    # 3. Heatmap: Arrest probability by Hour x DOW
    # ===============================================
    if "Hour" in df.columns and "DayOfWeek" in df.columns:
        pivot = df.pivot_table(
            index="DayOfWeek",
            columns="Hour",
            values="Arrest",
            aggfunc="mean"
        )

        plt.figure(figsize=(14, 6))
        sns.heatmap(pivot, cmap="rocket", annot=False)
        plt.title("Arrest Probability by Hour and Day of Week")
        plt.xlabel("Hour of Day")
        plt.ylabel("Day of Week (0 = Monday)")
        plt.tight_layout()
        plt.savefig("heatmap_hour_dayofweek_arrest.png", dpi=200)
        plt.show()
        print("Saved: heatmap_hour_dayofweek_arrest.png")

    # ====================================
    # 4. Arrest rate by Primary Crime Type
    # ====================================
    if "Primary Type" in df.columns:
        crime_rate = (
            df.groupby("Primary Type")["Arrest"]
              .mean()
              .sort_values(ascending=False)
              .reset_index()
        )
        crime_rate["Arrest"] *= 100

        plt.figure(figsize=(12, 10))
        sns.barplot(
            data=crime_rate,
            y="Primary Type",
            x="Arrest"
        )
        plt.title("Arrest Rate by Primary Crime Type")
        plt.xlabel("Arrest Rate (%)")
        plt.ylabel("Primary Crime Type")
        plt.tight_layout()
        plt.savefig("arrest_rate_by_primary_type.png", dpi=200)
        plt.show()
        print("Saved: arrest_rate_by_primary_type.png")

    # =====================================================
    # 5. Relationship between Block Frequency & Arrest Rate
    #    (binned BlockFreq -> average arrest rate by bin)
    # =====================================================
    if "BlockFreq" in df.columns:
        df2 = df.copy()
        df2["BlockFreqLog"] = np.log1p(df2["BlockFreq"])

        try:
            freq_bins = pd.qcut(df2["BlockFreqLog"], q=20, duplicates="drop")
            freq_rate = df2.groupby(freq_bins)["Arrest"].mean().reset_index()
            freq_rate["Arrest"] *= 100

            plt.figure(figsize=(12, 6))
            x_labels = freq_rate.index.astype(str)
            sns.lineplot(x=x_labels, y="Arrest", data=freq_rate, marker="o")
            plt.xticks(rotation=90)
            plt.title("Arrest Rate vs Block Crime Frequency (Log-Binned)")
            plt.xlabel("Block Frequency Bin (log-scaled, quantiles)")
            plt.ylabel("Arrest Rate (%)")
            plt.tight_layout()
            plt.savefig("arrest_rate_vs_blockfreq.png", dpi=200)
            plt.show()
            print("Saved: arrest_rate_vs_blockfreq.png")
        except ValueError:
            print("Not enough variation in BlockFreq to create quantile bins.")

    # =========================================
    # 6. Arrests by Time of Day (overall Hour)
    # =========================================
    if "Hour" in df.columns:
        hour_rate = df.groupby("Hour")["Arrest"].mean().reset_index()
        hour_rate["Arrest"] *= 100

        plt.figure(figsize=(10, 5))
        sns.lineplot(
            data=hour_rate,
            x="Hour",
            y="Arrest",
            marker="o"
        )
        plt.xticks(range(0, 24))
        plt.title("Arrest Rate by Hour of Day")
        plt.xlabel("Hour of Day (0–23)")
        plt.ylabel("Arrest Rate (%)")
        plt.tight_layout()
        plt.savefig("arrest_rate_by_hour.png", dpi=200)
        plt.show()
        print("Saved: arrest_rate_by_hour.png")

    # =========================================
    # 7. Arrest patterns through the years
    # =========================================
    if "Year" in df.columns:
        year_rate = df.groupby("Year")["Arrest"].mean().reset_index()
        year_rate["Arrest"] *= 100

        plt.figure(figsize=(10, 5))
        sns.lineplot(
            data=year_rate,
            x="Year",
            y="Arrest",
            marker="o"
        )
        plt.title("Arrest Rate Over the Years")
        plt.xlabel("Year")
        plt.ylabel("Arrest Rate (%)")
        plt.tight_layout()
        plt.savefig("arrest_rate_by_year.png", dpi=200)
        plt.show()
        print("Saved: arrest_rate_by_year.png")

    # ==========================================================
    # 8. "Map" of Beats: Beat vs Arrest Percentage (scatter-style)
    #    (Not geographic, just beat-level variation.)
    # ==========================================================
    if "Beat" in df.columns:
        beat_rate = df.groupby("Beat")["Arrest"].mean().reset_index()
        beat_rate["Arrest"] *= 100

        try:
            beat_rate["BeatNum"] = beat_rate["Beat"].astype(int)
            beat_rate_sorted = beat_rate.sort_values("BeatNum")
        except ValueError:
            beat_rate_sorted = beat_rate.sort_values("Beat")

        plt.figure(figsize=(14, 6))
        plt.scatter(
            x=beat_rate_sorted["Beat"],
            y=beat_rate_sorted["Arrest"]
        )
        plt.xticks(rotation=90)
        plt.title("Beat vs Arrest Percentage")
        plt.xlabel("Beat")
        plt.ylabel("Arrest Rate (%)")
        plt.tight_layout()
        plt.savefig("beat_vs_arrest_percentage_scatter.png", dpi=200)
        plt.show()
        print("Saved: beat_vs_arrest_percentage_scatter.png")

    # =============================================
    # 9. Arrest Rate by Season
    # =============================================
    if "Season" in df.columns:
        season_rate = (
            df.groupby("Season")["Arrest"]
              .mean()
              .reset_index()
        )
        season_rate["Arrest"] *= 100

        # Ensure a logical order if all present
        season_order = ["Winter", "Spring", "Summer", "Fall"]
        season_rate = (
            season_rate
            .set_index("Season")
            .reindex([s for s in season_order if s in season_rate.set_index("Season").index])
        )
        season_rate = season_rate.reset_index()

        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=season_rate,
            x="Season",
            y="Arrest",
            palette=["#66b2ff", "#99ff99", "#ffcc66", "#ff9999"][: len(season_rate)]
        )
        plt.title("Arrest Rate by Season")
        plt.xlabel("Season")
        plt.ylabel("Arrest Rate (%)")
        plt.tight_layout()
        plt.savefig("arrest_rate_by_season.png", dpi=200)
        plt.show()
        print("Saved: arrest_rate_by_season.png")

    # =============================================
    # 10. Arrest Rate by Month (Seasonality Cycle)
    # =============================================
    if "Month" in df.columns:
        month_rate = (
            df.groupby("Month")["Arrest"]
              .mean()
              .reset_index()
        )
        month_rate["Arrest"] *= 100

        plt.figure(figsize=(10, 5))
        sns.lineplot(
            data=month_rate,
            x="Month",
            y="Arrest",
            marker="o"
        )
        plt.xticks(range(1, 13))
        plt.title("Arrest Rate by Month")
        plt.xlabel("Month")
        plt.ylabel("Arrest Rate (%)")
        plt.tight_layout()
        plt.savefig("arrest_rate_by_month.png", dpi=200)
        plt.show()
        print("Saved: arrest_rate_by_month.png")

# ===================================================
# 11. Time-of-day intensity: counts + arrest rate
# ===================================================
    if "Hour" in df.columns:
        hour_stats = (
        df.groupby("Hour")["Arrest"]
          .agg(count="size", arrests="sum")
          .reset_index()
    )
    hour_stats["arrest_rate"] = 100 * hour_stats["arrests"] / hour_stats["count"]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    bar_color = "#4A90E2"
    line_color = "#D0021B"

    # Bars
    sns.barplot(
        data=hour_stats,
        x="Hour",
        y="count",
        ax=ax1,
        color=bar_color,
        alpha=0.75
    )
    ax1.set_xlabel("Hour of Day (0–23)")
    ax1.set_ylabel("Number of Incidents", color=bar_color)
    ax1.tick_params(axis="y", labelcolor=bar_color)

    # Line
    ax2 = ax1.twinx()
    sns.lineplot(
        data=hour_stats,
        x="Hour",
        y="arrest_rate",
        marker="o",
        linewidth=2,
        markersize=7,
        ax=ax2,
        color=line_color
    )
    ax2.set_ylabel("Arrest Rate (%)", color=line_color)
    ax2.tick_params(axis="y", labelcolor=line_color)

    avg_rate_percent = 26.9
    ax2.axhline(
        avg_rate_percent,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Overall Avg ({avg_rate_percent:.1f}%)"
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=bar_color, lw=10, label="Incident Count (bars)"),
        Line2D([0], [0], color=line_color, lw=2, marker="o", label="Arrest Rate (line)"),
        Line2D([0], [0], color="gray", linestyle="--", lw=2, label="Overall Avg Arrest Rate")
    ]
    ax1.legend(handles=legend_elements, loc="lower right")

    plt.title("Crime Intensity and Arrest Rate by Hour of Day")
    plt.tight_layout()
    plt.savefig("intensity_and_arrest_rate_by_hour.png", dpi=200)
    plt.show()

    print("Saved: intensity_and_arrest_rate_by_hour.png")

    # ===================================================
    # 12. Arrest Rate by Day of Week
    # ===================================================
    if "DayOfWeek" in df.columns:
        dow_rate = (
            df.groupby("DayOfWeek")["Arrest"]
              .mean()
              .reset_index()
        )
        dow_rate["Arrest"] *= 100

        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=dow_rate,
            x="DayOfWeek",
            y="Arrest"
        )
        plt.title("Arrest Rate by Day of Week (0 = Monday)")
        plt.xlabel("Day of Week")
        plt.ylabel("Arrest Rate (%)")
        plt.tight_layout()
        plt.savefig("arrest_rate_by_dayofweek.png", dpi=200)
        plt.show()
        print("Saved: arrest_rate_by_dayofweek.png")

    # ===================================================
    # 13. Hour-of-day patterns: Weekend vs Weekday
    # ===================================================
    if "Hour" in df.columns and "IsWeekend" in df.columns:
        # IsWeekend assumed 0/1
        hour_weekend = (
            df.groupby(["Hour", "IsWeekend"])["Arrest"]
              .mean()
              .reset_index()
        )
        hour_weekend["Arrest"] *= 100

        plt.figure(figsize=(10, 5))
        sns.lineplot(
            data=hour_weekend,
            x="Hour",
            y="Arrest",
            hue="IsWeekend",
            marker="o"
        )
        plt.xticks(range(0, 24))
        plt.title("Arrest Rate by Hour: Weekday vs Weekend")
        plt.xlabel("Hour of Day (0–23)")
        plt.ylabel("Arrest Rate (%)")
        plt.legend(title="IsWeekend", labels=["Weekday (0)", "Weekend (1)"])
        plt.tight_layout()
        plt.savefig("arrest_rate_by_hour_weekend_split.png", dpi=200)
        plt.show()
        print("Saved: arrest_rate_by_hour_weekend_split.png")


if __name__ == "__main__":
    main()
