#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 19:00:14 2025

@author: lukeladd
"""

import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "chicago_crime_sample_prepped.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

RF_PARAMS = dict(
    n_estimators=400,
    max_depth=20,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

sns.set(style="whitegrid")


# ===========================
# Benchmark helper structures
# ===========================

@dataclass
class BenchRow:
    model: str
    auc_05: float; acc_05: float; f1_05: float; tpr_05: float; tnr_05: float
    thr_rec: float; auc_rec: float; acc_rec: float; f1_rec: float; tpr_rec: float; tnr_rec: float
    t_pre: float; t_cv: float; t_pred: float
    n_features: int
    note: str


def best_threshold_for_recall(y_true, proba, min_precision=0.60):

    thresholds = np.unique(proba)
    best_thr = 0.5
    best_rec = -1.0

    for thr in thresholds:
        y_pred = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0

        if prec >= min_precision and rec > best_rec:
            best_rec = rec
            best_thr = thr

    return best_thr


def benchmark_model(model_name, pipe, X_test, y_test,
                    t_pre=0.0, t_cv=0.0, note=""):

    # Prediction timing
    t2 = time.perf_counter()
    proba = pipe.predict_proba(X_test)[:, 1]
    t_pred = time.perf_counter() - t2

    # Default threshold 0.5
    y_pred_05 = (proba >= 0.5).astype(int)
    auc05 = roc_auc_score(y_test, proba)
    acc05 = accuracy_score(y_test, y_pred_05)
    f105  = f1_score(y_test, y_pred_05)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_05).ravel()
    tpr05 = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr05 = tn / (tn + fp) if (tn + fp) else float("nan")

    # Recall-optimized threshold with precision floor
    thr_rec = best_threshold_for_recall(y_test, proba, min_precision=0.60)
    y_pred_rec = (proba >= thr_rec).astype(int)
    aucrec = roc_auc_score(y_test, proba)
    accrec = accuracy_score(y_test, y_pred_rec)
    f1rec  = f1_score(y_test, y_pred_rec)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rec).ravel()
    tprrec = tp / (tp + fn) if (tp + fn) else float("nan")
    tnrrec = tn / (tn + fp) if (tn + fp) else float("nan")

    # Number of features after preprocessing (if accessible)
    try:
        X_te_trans = pipe.named_steps["preprocess"].transform(X_test)
        n_features = X_te_trans.shape[1]
    except Exception:
        n_features = -1

    row = BenchRow(
        model=model_name,
        auc_05=auc05, acc_05=acc05, f1_05=f105, tpr_05=tpr05, tnr_05=tnr05,
        thr_rec=thr_rec, auc_rec=aucrec, acc_rec=accrec, f1_rec=f1rec, tpr_rec=tprrec, tnr_rec=tnrrec,
        t_pre=t_pre, t_cv=t_cv, t_pred=t_pred,
        n_features=n_features,
        note=note,
    )
    return row


# ===========================
# Existing RF error analysis
# ===========================

def build_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["Arrest"] = df["Arrest"].astype(int)

    categorical_features = [
        "Primary Type",
        "Description",
        "Location Description",
        "Beat",
        "Block",
        "District",
        "Ward",
        "Community Area",
        "Season",
    ]
    categorical_features = [c for c in categorical_features if c in df.columns]

    numeric_features = [
        "Domestic",
        "BlockFreq",
        "Year",
        "Month",
        "DayOfWeek",
        "Hour",
        "IsWeekend",
    ]
    numeric_features = [c for c in numeric_features if c in df.columns]

    X = df[categorical_features + numeric_features]
    y = df["Arrest"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, categorical_features, numeric_features


def build_pipeline(categorical_features, numeric_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    rf = RandomForestClassifier(**RF_PARAMS)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("rf", rf),
        ]
    )
    return pipe


def basic_evaluation(y_test, y_pred, y_proba):
    print("\n=== BASIC EVALUATION (Random Forest) ===")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=3))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.3f}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


def build_results_df(X_test, y_test, y_pred, y_proba):
    results = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "proba": y_proba,
    }).reset_index(drop=True)

    features = X_test.reset_index(drop=True)
    results = pd.concat([results, features], axis=1)
    return results


def summarize_error_sets(results):
    FP = results[(results.y_pred == 1) & (results.y_true == 0)]
    FN = results[(results.y_pred == 0) & (results.y_true == 1)]
    TP = results[(results.y_pred == 1) & (results.y_true == 1)]
    TN = results[(results.y_pred == 0) & (results.y_true == 0)]

    print("\n=== ERROR SET SIZES (Random Forest) ===")
    print(f"False Positives (FP): {len(FP)}")
    print(f"False Negatives (FN): {len(FN)}")
    print(f"True Positives (TP):  {len(TP)}")
    print(f"True Negatives (TN):  {len(TN)}")

    return FP, FN, TP, TN


def error_rates_by_category(results, FP, FN, column, top_n=15):
    if column not in results.columns:
        print(f"\nColumn '{column}' not in results; skipping.")
        return

    print(f"\n=== ERROR RATES BY {column.upper()} (RF) ===")

    counts = results.groupby(column).size().rename("total")
    fp_counts = FP.groupby(column).size().rename("FP")
    fn_counts = FN.groupby(column).size().rename("FN")

    df_err = pd.concat([counts, fp_counts, fn_counts], axis=1).fillna(0)
    df_err["FP_rate"] = df_err["FP"] / df_err["total"]
    df_err["FN_rate"] = df_err["FN"] / df_err["total"]

    print("\nTop categories by FN_rate:")
    print(df_err.sort_values("FN_rate", ascending=False).head(top_n))

    print("\nTop categories by FP_rate:")
    print(df_err.sort_values("FP_rate", ascending=False).head(top_n))


def error_rate_by_numeric(results, column, bins=10):
    if column not in results.columns:
        print(f"\nColumn '{column}' not in results; skipping.")
        return

    print(f"\n=== ERROR RATES ACROSS {column} BINS (RF) ===")

    try:
        if column == "Hour":
            results["_bin"] = results["Hour"]
        else:
            results["_bin"] = pd.qcut(results[column], q=bins, duplicates="drop")
    except Exception as e:
        print(f"Could not bin {column}: {e}")
        return

    FP = results[(results.y_pred == 1) & (results.y_true == 0)]
    FN = results[(results.y_pred == 0) & (results.y_true == 1)]

    counts = results.groupby("_bin").size().rename("total")
    fp_counts = FP.groupby("_bin").size().rename("FP")
    fn_counts = FN.groupby("_bin").size().rename("FN")

    df_err = pd.concat([counts, fp_counts, fn_counts], axis=1).fillna(0)
    df_err["FP_rate"] = df_err["FP"] / df_err["total"]
    df_err["FN_rate"] = df_err["FN"] / df_err["total"]

    print(df_err)

    df_plot = df_err.reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(df_plot["_bin"].astype(str), df_plot["FN_rate"], marker="o", label="FN_rate")
    plt.plot(df_plot["_bin"].astype(str), df_plot["FP_rate"], marker="o", label="FP_rate")
    plt.xticks(rotation=45)
    plt.title(f"FP/FN Rates Across {column} Bins (RF)")
    plt.xlabel(column)
    plt.ylabel("Error Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    results.drop(columns="_bin", inplace=True, errors="ignore")


def main():
    X_train, X_test, y_train, y_test, cat_feats, num_feats = build_data()
    pipe = build_pipeline(cat_feats, num_feats)

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    basic_evaluation(y_test, y_pred, y_proba)

    results = build_results_df(X_test, y_test, y_proba, y_proba)
    FP, FN, TP, TN = summarize_error_sets(results)

    # Categorical error breakdowns
    for col, top_n in [
        ("Primary Type", 15),
        ("Beat", 20),
        ("Season", 10),
        ("Domestic", 5),
    ]:
        if col in results.columns:
            error_rates_by_category(results, FP, FN, col, top_n=top_n)

    # Numeric: Hour and Year
    if "Hour" in results.columns:
        error_rate_by_numeric(results, "Hour", bins=24)
    if "Year" in results.columns:
        error_rate_by_numeric(results, "Year", bins=10)

    # Save FP/FN
    FP.to_csv("rf_false_positives_details.csv", index=False)
    FN.to_csv("rf_false_negatives_details.csv", index=False)

    # -----------------------
    # Benchmark summary row
    # -----------------------
    bench_row = benchmark_model(
        model_name="RandomForest",
        pipe=pipe,
        X_test=X_test,
        y_test=y_test,
        t_pre=0.0,          
        t_cv=0.0,           
        note=str(RF_PARAMS) 
    )

    summary = pd.DataFrame([asdict(bench_row)])
    print("\n=== BENCHMARK SUMMARY (RF) ===")
    print(summary.to_string(index=False))

    summary.to_csv("rf_benchmark_summary.csv", index=False)
    print("\nSaved RF benchmark summary to rf_benchmark_summary.csv.")


if __name__ == "__main__":
    main()
