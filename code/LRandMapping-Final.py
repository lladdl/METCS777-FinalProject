#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import time
import os
import math
import json
import tempfile
import subprocess
import numpy as np

from pyspark.sql import SparkSession, functions as F
from pyspark.sql import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, FeatureHasher, VectorAssembler
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.functions import vector_to_array

try:
    import pandas as pd
    import folium
    from folium.features import GeoJson, GeoJsonTooltip
except Exception:
    pd = folium = None

DESIRED_LOWER = [
    "date", "primary_type", "location_description", "beat", "district", "ward",
    "community_area", "arrest", "domestic", "fbi_code", "iucr", "block"
]

def parse_args():
    p = argparse.ArgumentParser("Fast Spark LR (hashing, no grid)")
    p.add_argument("--in",  dest="in_path",
                   default="gs://metcs777-term-project/Crimes_2001_2025_Parquet/",
                   help="Input Parquet directory (GCS or local).")
    ts_default = time.strftime("lr_fast_%Y%m%d_%H%M%S")
    p.add_argument("--out", dest="out_dir",
                   default=f"gs://metcs777-term-project/output/{ts_default}/",
                   help="Output directory (GCS or local).")
    p.add_argument("--partitions", type=int, default=256, help="spark.sql.shuffle.partitions.")
    p.add_argument("--numFeatures", type=int, default=1 << 18, help="FeatureHasher size (power of 2 works best).")
    p.add_argument("--neg_downsample", type=float, default=1.0,
                   help="Keep this fraction of negatives (0 < f ≤ 1). Example: 0.2 keeps 20%.")
    p.add_argument("--maxIter", type=int, default=50)
    p.add_argument("--regParam", type=float, default=0.1)
    p.add_argument("--elasticNetParam", type=float, default=0.0)
    p.add_argument("--map_geojson_path",
                   default="gs://metcs777-term-project/Boundaries_-_Community_Areas_20251205.geojson",
                   help="Chicago Community Areas GeoJSON path.")
    p.add_argument("--map_html_out",
                   default="/tmp/chicago_accuracy_map.html",
                   help="Local HTML path for the interactive map.")
    p.add_argument("--map_upload_gcs",
                   default="",
                   help="Optional gs:// path to upload the map once saved.")
    p.add_argument("--map_min_support", type=int, default=200,
                   help="Only color areas with at least this many test rows; others stay gray.")
    return p.parse_args()

def robust_timestamp(col):
    patterns = [
        "MM/dd/yyyy hh:mm:ss a", "MM/dd/yyyy HH:mm:ss", "MM/dd/yyyy HH:mm",
        "yyyy-MM-dd HH:mm:ss", "yyyy-MM-dd'T'HH:mm:ss"
    ]
    ts = F.to_timestamp(col)
    for pat in patterns:
        ts = F.coalesce(ts, F.to_timestamp(col, pat))
    return ts

def basic_evaluation_spark(pred_df):
    agg = (pred_df
           .select(
               F.sum(F.when((F.col("label") == 0) & (F.col("prediction") == 0), 1).otherwise(0)).alias("tn"),
               F.sum(F.when((F.col("label") == 0) & (F.col("prediction") == 1), 1).otherwise(0)).alias("fp"),
               F.sum(F.when((F.col("label") == 1) & (F.col("prediction") == 0), 1).otherwise(0)).alias("fn"),
               F.sum(F.when((F.col("label") == 1) & (F.col("prediction") == 1), 1).otherwise(0)).alias("tp"),
               F.count("*").alias("n")
           )
           .collect()[0])

    tn, fp, fn, tp, n = int(agg.tn), int(agg.fp), int(agg.fn), int(agg.tp), int(agg.n)
    accuracy  = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("\nQuick eval (LogReg on Spark)")
    print(f"accuracy:  {accuracy:.3f}")
    print(f"precision: {precision:.3f}")
    print(f"recall:    {recall:.3f}")
    print(f"f1-score:  {f1:.3f}")
    print("\nconfusion matrix (rows=true, cols=pred):")
    print(np.array([[tn, fp], [fn, tp]]))
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "n": n, "accuracy": accuracy}

def export_curves_via_buckets(pred_df, score_col="score", label_col="label", steps=200, out_base=None):
    """
    Build light-weight ROC/PR curves by binning scores into 'steps' buckets.
    If out_base is set, write CSVs to {out_base}/roc and {out_base}/pr.
    """
    totals = pred_df.agg(
        F.sum(F.when(F.col(label_col) == 1, 1).otherwise(0)).alias("P"),
        F.sum(F.when(F.col(label_col) == 0, 1).otherwise(0)).alias("N"),
    ).collect()[0]

    P = float(totals.P or 0.0)
    N = float(totals.N or 0.0)
    if P == 0 or N == 0:
        print("[curves] can't build ROC/PR: P or N is zero")
        return None, None

    bucket = F.floor(F.col(score_col) * steps) / steps
    by_bucket = (pred_df
        .withColumn("bucket", bucket)
        .groupBy("bucket")
        .agg(
            F.sum(F.when(F.col(label_col) == 1, 1).otherwise(0)).alias("tp_bucket"),
            F.sum(F.when(F.col(label_col) == 0, 1).otherwise(0)).alias("fp_bucket"),
        ))

    w = Window.orderBy(F.desc("bucket")).rowsBetween(Window.unboundedPreceding, 0)
    cum = (by_bucket
        .select(
            "bucket",
            F.sum("tp_bucket").over(w).alias("cum_tp"),
            F.sum("fp_bucket").over(w).alias("cum_fp"),
        )
        .orderBy(F.desc("bucket")))

    roc = cum.select((F.col("cum_fp")/F.lit(N)).alias("fpr"),
                     (F.col("cum_tp")/F.lit(P)).alias("tpr"))

    pr = cum.select((F.col("cum_tp")/F.lit(P)).alias("recall"),
                    (F.col("cum_tp")/(F.col("cum_tp")+F.col("cum_fp"))).alias("precision"))

    if out_base:
        roc.coalesce(1).write.mode("overwrite").option("header", True).csv(out_base + "/roc")
        pr.coalesce(1).write.mode("overwrite").option("header", True).csv(out_base + "/pr")
        print(f"[curves] wrote CSVs to {out_base}/roc and {out_base}/pr")

    return roc, pr

def main():
    start_time = time.time()
    args = parse_args()

    run_tag = os.path.basename(args.out_dir.rstrip("/"))
    default_map_upload = f"{args.out_dir.rstrip('/')}/maps/{run_tag}/chicago_accuracy_map.html"
    print(f"run: {run_tag}")
    print(f"map upload target (default): {default_map_upload}")

    spark = (
        SparkSession.builder
        .appName("ChicagoCrime-LR-Fast")
        .config("spark.sql.shuffle.partitions", str(args.partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .getOrCreate()
    )
    print(f"paths -> in: {args.in_path} | out: {args.out_dir} | partitions: {args.partitions}")

    # 1) Read
    df = spark.read.parquet(args.in_path)
    idx = {c.lower(): c for c in df.columns}
    keep = [idx[c] for c in DESIRED_LOWER if c in idx]
    if not keep:
        raise RuntimeError(f"No expected columns found. Got: {df.columns}")
    df = df.select(*keep)

    print("after normalization cols:", df.columns)
    alt_names = ["community area", "Community Area", "Community_Area", "COMMUNITY_AREA",
                 "communityarea", "COMMUNITYAREA", "CommunityArea"]
    for alt in alt_names:
        if alt in idx and "community_area" not in df.columns:
            df = df.withColumnRenamed(idx[alt], "community_area")
            print(f"renamed '{alt}' -> 'community_area'")
            break

    if "community_area" in df.columns:
        df = df.withColumn("community_area", F.col("community_area").cast("int"))
        non_null_area = df.filter(F.col("community_area").isNotNull()).count()
        total_rows = df.count()
        print(f"community_area non-null: {non_null_area:,} / {total_rows:,}")
    else:
        print("note: 'community_area' still missing after rescue attempts")

    for low in DESIRED_LOWER:
        if low in idx:
            df = df.withColumnRenamed(idx[low], low)

    # Labels
    df = df.na.drop(subset=["date", "arrest"])
    arrest_norm = F.lower(F.trim(F.col("arrest").cast("string")))
    df = df.withColumn(
        "label",
        F.when(arrest_norm.isin("true", "t", "1", "y", "yes"), 1.0)
         .when(arrest_norm.isin("false", "f", "0", "n", "no"), 0.0)
    ).filter(F.col("label").isNotNull())

    if "domestic" in df.columns:
        dom_norm = F.lower(F.trim(F.col("domestic").cast("string")))
        df = df.withColumn(
            "DomesticNum",
            F.when(dom_norm.isin("true", "t", "1", "y", "yes"), 1.0)
             .when(dom_norm.isin("false", "f", "0", "n", "no"), 0.0)
        )
    else:
        df = df.withColumn("DomesticNum", F.lit(None).cast("double"))

    df = df.withColumn("ts", robust_timestamp(F.col("date"))).filter(F.col("ts").isNotNull())
    df = (df.withColumn("Year", F.year("ts").cast("double"))
            .withColumn("Month", F.month("ts").cast("double"))
            .withColumn("DayOfWeek", F.dayofweek("ts").cast("double"))
            .withColumn("Hour", F.hour("ts").cast("double"))
            .withColumn("IsWeekend", F.when(F.col("DayOfWeek").isin(1, 7), 1.0).otherwise(0.0))
            .drop("ts"))
    
    if args.neg_downsample < 1.0:
        pos = df.filter("label=1")
        neg = df.filter("label=0").sample(False, args.neg_downsample, seed=42)
        df = pos.unionByName(neg).repartition(args.partitions)

    cat_cols = [c for c in ["primary_type", "location_description", "beat", "district", "ward",
                            "community_area", "fbi_code", "iucr", "block"] if c in df.columns]
    num_cols = [c for c in ["DomesticNum", "Year", "Month", "DayOfWeek", "Hour", "IsWeekend"] if c in df.columns]

    if cat_cols:
        df = df.fillna({c: "__NA__" for c in cat_cols})
    for c in num_cols:
        df = df.withColumn(c, F.col(c).cast("double"))

    imputer = Imputer(strategy="median",
                      inputCols=num_cols,
                      outputCols=[f"{c}_imp" for c in num_cols]) if num_cols else None

    num_imp = [f"{c}_imp" for c in num_cols]
    num_vec = VectorAssembler(inputCols=num_imp, outputCol="num_vec", handleInvalid="skip") if num_imp else None
    hasher  = FeatureHasher(inputCols=cat_cols, outputCol="cat_hashed", numFeatures=args.numFeatures) if cat_cols else None

    inputs = []
    if hasher:  inputs.append("cat_hashed")
    if num_vec: inputs.append("num_vec")
    assembler = VectorAssembler(inputCols=inputs, outputCol="features", handleInvalid="skip")

    lr = LogisticRegression(
        labelCol="label",
        featuresCol="features",
        maxIter=args.maxIter,
        regParam=args.regParam,
        elasticNetParam=args.elasticNetParam,
        standardization=True,
        aggregationDepth=2
    )

    stages = []
    if imputer: stages.append(imputer)
    if num_vec: stages.append(num_vec)
    if hasher:  stages.append(hasher)
    stages.extend([assembler, lr])
    pipeline = Pipeline(stages=stages)

    # train-test split
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    print(f"split sizes -> train: {train.count():,} | test: {test.count():,}")
    model = pipeline.fit(train)

    keep_for_pred = ["label"] + [c for c in ["primary_type", "fbi_code", "iucr", "block", "community_area"] if c in df.columns]
    pred = model.transform(test).select(*keep_for_pred, "rawPrediction", "probability", "prediction")
    pred = (pred
            .withColumn("prob_arr", vector_to_array(F.col("probability")))
            .withColumn("score", F.col("prob_arr")[1].cast("double"))
            .drop("prob_arr"))

    eval_summary = basic_evaluation_spark(pred)

    try:
        lr_model = next((s for s in model.stages if isinstance(s, LogisticRegressionModel)), None)
        if lr_model is not None:
            coef_vec = np.array(lr_model.coefficients)
            cat_dim = args.numFeatures if hasher is not None else 0
            num_feature_names = num_imp
            num_dim = len(num_feature_names)
            if num_dim > 0:
                num_slice = coef_vec[cat_dim:cat_dim + num_dim]
                rows = [{"feature": fn, "coef": float(w)} for fn, w in zip(num_feature_names, num_slice)]
                if pd is not None:
                    coef_df = pd.DataFrame(rows).sort_values("coef", ascending=False)
                    print("\nstrongest positive numeric predictors:")
                    print(coef_df.head(10).to_string(index=False))
                    print("\nstrongest negative numeric predictors:")
                    print(coef_df.tail(10).sort_values("coef").to_string(index=False))
                else:
                    print("\nstrongest positive numeric predictors:")
                    for r in sorted(rows, key=lambda x: x["coef"], reverse=True)[:10]:
                        print(f"{r['feature']:>18}  {r['coef']: .6f}")
                    print("\nstrongest negative numeric predictors:")
                    for r in sorted(rows, key=lambda x: x["coef"])[:10]:
                        print(f"{r['feature']:>18}  {r['coef']: .6f}")
            else:
                print("\nno numeric features to inspect")
        else:
            print("\nlogistic regression stage not found, skipping coefficient peek")
    except Exception as e:
        print(f"\ncoef inspection failed: {e}")
        print("note: hashed categorical features can't be mapped back to original names")

    # accuracy by community area
    has_area = ("community_area" in pred.columns) and (
        pred.filter(F.col("community_area").isNotNull()).limit(1).count() > 0
    )
    if has_area:
        acc_by_area = (
            pred.groupBy("community_area")
                .agg(F.mean((F.col("label") == F.col("prediction")).cast("double")).alias("accuracy"),
                     F.count("*").alias("n"))
                .withColumn("accuracy", F.round(F.col("accuracy"), 4))
                .withColumn("community_area", F.col("community_area").cast("int"))
        )
        out_csv_dir = args.out_dir.rstrip("/") + "/maps/accuracy_by_community_csv"
        (acc_by_area.orderBy(F.col("community_area"))
            .coalesce(1).write.mode("overwrite").option("header", True).csv(out_csv_dir))
        print(f"wrote accuracy-by-community CSV -> {out_csv_dir}")
        acc_pdf = acc_by_area.toPandas() if pd is not None else None
    else:
        print("skipping map: community_area unavailable")
        acc_pdf = None

    # AUC
    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    auc_roc = evaluator.evaluate(pred)

    metrics_rdd = pred.select("score", F.col("label").cast("double")).rdd.map(tuple)
    metrics = BinaryClassificationMetrics(metrics_rdd)
    auc_pr = metrics.areaUnderPR
    print(f"AUCs -> ROC: {auc_roc:.4f} | PR: {auc_pr:.4f}")

    curves_base = args.out_dir.rstrip("/") + "/curves"
    export_curves_via_buckets(pred, score_col="score", label_col="label", steps=200, out_base=curves_base)

    model.write().overwrite().save(args.out_dir.rstrip("/") + "/model_fast")

    # map
    if acc_pdf is not None and folium is not None:
        try:
            print(f"reading GeoJSON: {args.map_geojson_path}")
            gj_lines = spark.read.text(args.map_geojson_path).rdd.map(lambda r: r[0]).collect()
            if not gj_lines:
                raise RuntimeError(f"empty GeoJSON at {args.map_geojson_path}")
            gj_obj = json.loads("\n".join(gj_lines))

            acc_map = {int(r.community_area): (float(r.accuracy), int(r.n)) for _, r in acc_pdf.iterrows()}
            min_support = int(args.map_min_support)

            key_candidates = ["area_numbe", "area_num_1", "area_num", "AREA_NUMBE", "AREA_NUM_1", "AREA_NUM"]
            sample_props = gj_obj["features"][0].get("properties", {}) if gj_obj.get("features") else {}
            left_key = next((k for k in key_candidates if k in sample_props), None)
            if left_key is None:
                for f in gj_obj.get("features", []):
                    p = f.get("properties", {}) or {}
                    left_key = next((k for k in key_candidates if k in p), None)
                    if left_key:
                        break
            if left_key is None:
                raise RuntimeError(f"no community-area id key found in GeoJSON; saw keys like: {list(sample_props.keys())[:12]}")
            print(f"joining on GeoJSON key: '{left_key}'")

            for feat in gj_obj.get("features", []):
                props = feat.get("properties", {})
                ca = None
                if left_key in props and props[left_key] is not None and str(props[left_key]).strip() != "":
                    try:
                        ca = int(props[left_key])
                    except Exception:
                        ca = None
                if ca is not None and ca in acc_map:
                    acc, n = acc_map[ca]
                    props["accuracy"] = acc
                    props["n"] = n
                    props["acc_plot"] = acc if n >= min_support else None
                else:
                    props["accuracy"] = None
                    props["n"] = 0
                    props["acc_plot"] = None

            def color_from_acc(a):
                if a is None or (isinstance(a, float) and math.isnan(a)):
                    return "#dddddd"
                lo, hi = 0.4, 0.9
                t = (max(lo, min(hi, float(a))) - lo) / (hi - lo + 1e-9)
                c0, c1 = (33, 113, 181), (215, 25, 28)
                r = int(c0[0] + t * (c1[0] - c0[0]))
                g = int(c0[1] + t * (c1[1] - c0[1]))
                b = int(c0[2] + t * (c1[2] - c0[2]))
                return f"#{r:02x}{g:02x}{b:02x}"

            m = folium.Map(location=[41.8781, -87.6298], zoom_start=10, tiles="cartodbpositron")

            def _style_fn(feat):
                acc = feat["properties"].get("acc_plot", None)
                return {
                    "fillColor": color_from_acc(acc),
                    "color": "#202020",
                    "weight": 0.5,
                    "fillOpacity": 0.75 if acc is not None else 0.3
                }

            tooltip_fields = [left_key, "accuracy", "n"]
            aliases = ["Area #", "Accuracy", "Test N"]
            if "community" in sample_props:
                tooltip_fields = [left_key, "community", "accuracy", "n"]
                aliases = ["Area #", "Community", "Accuracy", "Test N"]
            tooltip = GeoJsonTooltip(fields=tooltip_fields, aliases=aliases, localize=True, sticky=False)

            GeoJson(gj_obj, style_function=_style_fn, tooltip=tooltip, name="Model accuracy (test)").add_to(m)
            folium.LayerControl(collapsed=True).add_to(m)

            local_target = args.map_html_out or "/tmp/chicago_accuracy_map.html"
            gcs_target = args.map_upload_gcs.strip() or default_map_upload

            m.save(local_target)
            print(f"saved map -> {local_target}")

            if gcs_target.startswith("gs://"):
                subprocess.run(["gsutil", "cp", local_target, gcs_target], check=True)
                print(f"uploaded map -> {gcs_target}")

            print(f"areas with fewer than {min_support} test rows are left gray to avoid noisy color swings")
        except Exception as e:
            print(f"map rendering failed: {e}")

    runtime_min = (time.time() - start_time) / 60.0
    tn, fp, fn, tp = (eval_summary["tn"], eval_summary["fp"], eval_summary["fn"], eval_summary["tp"])
    accuracy = float(eval_summary["accuracy"])

    summary_row = [(
        float(auc_roc), float(auc_pr), accuracy,
        tp, tn, fp, fn,
        args.in_path, args.out_dir,
        time.strftime("%Y-%m-%d %H:%M:%S"), float(runtime_min)
    )]
    schema = ["roc_auc", "pr_auc", "accuracy", "tp", "tn", "fp", "fn",
              "input_path", "output_dir", "timestamp", "runtime_min"]

    (spark.createDataFrame(summary_row, schema)
         .coalesce(1)
         .write.mode("overwrite")
         .json(args.out_dir.rstrip("/") + "/metrics"))

    spark.stop()

if __name__ == "__main__":
    main()
