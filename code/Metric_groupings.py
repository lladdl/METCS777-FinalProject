#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 16:14:44 2025

@author: lukeladd
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load pre-computed RF/XGB metrics and plot
district + community-area choropleth maps.
"""

import pandas as pd
import geopandas as gpd
gpd.options.io_engine = "pyogrio"
import matplotlib.pyplot as plt


# -----------------------------
# FILE PATHS – EDIT THESE
# -----------------------------

# Metrics CSVs (already created by your scripts)
RF_DIST_CSV = "rf_metrics_by_district.csv"
XGB_DIST_CSV = "xgb_metrics_by_district.csv"

RF_CA_CSV   = "rf_metrics_by_community_area.csv"
XGB_CA_CSV  = "xgb_metrics_by_community_area.csv"

# Shapefiles (from Chicago Open Data)
DISTRICT_SHP_PATH  = "Boundaries-Police-Districts.shp"
COMMUNITY_SHP_PATH = "Boundaries-Community-Areas.shp"

# Column names in shapefiles (edit if different)
DISTRICT_ID_COL  = "DIST_NUM"    # in district shapefile
COMMUNITY_ID_COL = "AREA_NUMBE"  # in community-area shapefile


# -----------------------------
# Helper for plotting
# -----------------------------

def plot_choropleth(gdf, column, title, cmap="OrRd", vmin=None, vmax=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    gdf.plot(
        column=column,
        cmap=cmap,
        legend=True,
        linewidth=0.5,
        edgecolor="black",
        ax=ax,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    # -----------------------------
    # 1. Load metrics
    # -----------------------------
    rf_dist  = pd.read_csv(RF_DIST_CSV)
    xgb_dist = pd.read_csv(XGB_DIST_CSV)

    rf_ca  = pd.read_csv(RF_CA_CSV)
    xgb_ca = pd.read_csv(XGB_CA_CSV)

    # Ensure join keys are numeric
    if "District" in rf_dist.columns:
        rf_dist["District"] = rf_dist["District"].astype(int)
    if "District" in xgb_dist.columns:
        xgb_dist["District"] = xgb_dist["District"].astype(int)

    if "Community Area" in rf_ca.columns:
        rf_ca["Community Area"] = rf_ca["Community Area"].astype(int)
    if "Community Area" in xgb_ca.columns:
        xgb_ca["Community Area"] = xgb_ca["Community Area"].astype(int)

    # -----------------------------
    # 2. Load shapefiles
    # -----------------------------
    districts_gdf = gpd.read_file(DISTRICT_SHP_PATH)
    community_gdf = gpd.read_file(COMMUNITY_SHP_PATH)

    # Make numeric join keys from shapefile
    districts_gdf["dist_num"] = districts_gdf[DISTRICT_ID_COL].astype(int)
    community_gdf["area_num"] = community_gdf[COMMUNITY_ID_COL].astype(int)

    # -----------------------------
    # 3. Merge metrics with shapes
    # -----------------------------
    rf_dist_map = districts_gdf.merge(
        rf_dist, left_on="dist_num", right_on="District", how="inner"
    )
    xgb_dist_map = districts_gdf.merge(
        xgb_dist, left_on="dist_num", right_on="District", how="inner"
    )

    rf_ca_map = community_gdf.merge(
        rf_ca, left_on="area_num", right_on="Community Area", how="inner"
    )
    xgb_ca_map = community_gdf.merge(
        xgb_ca, left_on="area_num", right_on="Community Area", how="inner"
    )

    # -----------------------------
    # 4. Plot maps
    # Comment out what you don't need.
    # -----------------------------

    # ---- District: RF ----
    plot_choropleth(
        rf_dist_map,
        column="ArrestRate_true",
        title="RF: True Arrest Rate by District",
        cmap="OrRd",
    )
    plot_choropleth(
        rf_dist_map,
        column="AUC",
        title="RF: AUC by District",
        cmap="viridis",
    )
    plot_choropleth(
        rf_dist_map,
        column="FP_rate",
        title="RF: False Positive Rate by District",
        cmap="Purples",
    )
    plot_choropleth(
        rf_dist_map,
        column="FN_rate",
        title="RF: False Negative Rate by District",
        cmap="Blues",
    )

    # ---- District: XGB ----
    plot_choropleth(
        xgb_dist_map,
        column="ArrestRate_true",
        title="XGB: True Arrest Rate by District",
        cmap="OrRd",
    )
    plot_choropleth(
        xgb_dist_map,
        column="AUC",
        title="XGB: AUC by District",
        cmap="viridis",
    )
    plot_choropleth(
        xgb_dist_map,
        column="FP_rate",
        title="XGB: False Positive Rate by District",
        cmap="Purples",
    )
    plot_choropleth(
        xgb_dist_map,
        column="FN_rate",
        title="XGB: False Negative Rate by District",
        cmap="Blues",
    )

    # ---- Community Area: RF ----
    plot_choropleth(
        rf_ca_map,
        column="ArrestRate_true",
        title="RF: True Arrest Rate by Community Area",
        cmap="OrRd",
    )
    plot_choropleth(
        rf_ca_map,
        column="FP_rate",
        title="RF: False Positive Rate by Community Area",
        cmap="Purples",
    )
    plot_choropleth(
        rf_ca_map,
        column="FN_rate",
        title="RF: False Negative Rate by Community Area",
        cmap="Blues",
    )

    # ---- Community Area: XGB ----
    plot_choropleth(
        xgb_ca_map,
        column="ArrestRate_true",
        title="XGB: True Arrest Rate by Community Area",
        cmap="OrRd",
    )
    plot_choropleth(
        xgb_ca_map,
        column="FP_rate",
        title="XGB: False Positive Rate by Community Area",
        cmap="Purples",
    )
    plot_choropleth(
        xgb_ca_map,
        column="FN_rate",
        title="XGB: False Negative Rate by Community Area",
        cmap="Blues",
    )


if __name__ == "__main__":
    main()
