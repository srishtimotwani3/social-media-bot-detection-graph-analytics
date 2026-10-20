"""
05_classification.py
--------------------
Stage 5: Train a Random Forest classifier (MLlib) to detect bots.

Two models trained:
  1. WITH graph features   — baseline + graph signals
  2. WITHOUT graph features — only user-profile counts (the baseline)

This isolates the actual contribution of graph analytics.

Metrics reported: Precision, Recall, F1 (macro), ROC-AUC
Results saved:    results/metrics/classification_results.csv
"""

import sys, os
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_hadoop_home = str(PROJECT_ROOT / "hadoop")
os.environ["HADOOP_HOME"] = _hadoop_home
_hadoop_bin = str(PROJECT_ROOT / "hadoop" / "bin")
if _hadoop_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _hadoop_bin + os.pathsep + os.environ.get("PATH", "")

import json
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR  = PROJECT_ROOT / "results" / "metrics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature column sets ───────────────────────────────────────────────────────
PROFILE_FEATURES = [
    "followers_count", "friends_count", "statuses_count",
]

GRAPH_FEATURES = [
    "inDegree", "outDegree", "pagerank", "triangle_count",
    "mutual_retweet_count", "shared_target_count",
    "community_size", "community_bot_ratio",
]


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("BotDetection-Classification")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def build_pipeline(feature_cols: list, n_trees: int = 100) -> Pipeline:
    """Build a Spark ML Pipeline: impute → assemble → scale → RF."""
    imputer = Imputer(inputCols=feature_cols, outputCols=[f"{c}_imp" for c in feature_cols])
    imputed_cols = [f"{c}_imp" for c in feature_cols]

    assembler = VectorAssembler(inputCols=imputed_cols, outputCol="features_raw")
    scaler    = StandardScaler(inputCol="features_raw", outputCol="features",
                               withMean=True, withStd=True)
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label_bin",
        numTrees=n_trees,
        maxDepth=10,
        seed=42,
    )
    return Pipeline(stages=[imputer, assembler, scaler, rf])


def evaluate(predictions, label=""):
    """Compute Precision, Recall, F1, AUC from a predictions DataFrame."""
    mc_eval = MulticlassClassificationEvaluator(
        labelCol="label_bin", predictionCol="prediction"
    )
    bin_eval = BinaryClassificationEvaluator(
        labelCol="label_bin", rawPredictionCol="rawPrediction"
    )

    precision = mc_eval.setMetricName("weightedPrecision").evaluate(predictions)
    recall    = mc_eval.setMetricName("weightedRecall").evaluate(predictions)
    f1        = mc_eval.setMetricName("f1").evaluate(predictions)
    auc       = bin_eval.setMetricName("areaUnderROC").evaluate(predictions)

    print(f"\n-- {label} -------------------------------------")
    print(f"   Precision : {precision:.4f}")
    print(f"   Recall    : {recall:.4f}")
    print(f"   F1        : {f1:.4f}")
    print(f"   AUC-ROC   : {auc:.4f}")

    return {"model": label, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    features_df = spark.read.parquet(str(PROC_DIR / "features.parquet"))

    # Drop rows with no label
    features_df = features_df.filter(F.col("label_bin").isNotNull())

    # Train/test split (80/20, stratified approximation via random split)
    train, test = features_df.randomSplit([0.8, 0.2], seed=42)
    train.cache()
    test.cache()

    print(f"[INFO] Train: {train.count()} | Test: {test.count()}")

    results = []

    # ── Model A: WITH graph features ──────────────────────────────────────────
    all_features = PROFILE_FEATURES + GRAPH_FEATURES
    pipe_a = build_pipeline(all_features)
    model_a = pipe_a.fit(train)
    preds_a = model_a.transform(test)
    results.append(evaluate(preds_a, label="WITH graph features"))

    # ── Model B: WITHOUT graph features (baseline) ────────────────────────────
    pipe_b = build_pipeline(PROFILE_FEATURES)
    model_b = pipe_b.fit(train)
    preds_b = model_b.transform(test)
    results.append(evaluate(preds_b, label="WITHOUT graph features (baseline)"))

    # ── Feature importances (model A) ─────────────────────────────────────────
    rf_model = model_a.stages[-1]
    importances = rf_model.featureImportances.toArray()
    imp_df = pd.DataFrame({
        "feature": all_features,
        "importance": importances[:len(all_features)],
    }).sort_values("importance", ascending=False)
    print("\n-- Feature Importances (WITH graph) -------------")
    print(imp_df.to_string(index=False))

    imp_df.to_csv(str(RESULTS_DIR / "feature_importances.csv"), index=False)

    # ── Save classification results ────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    out_path = str(RESULTS_DIR / "classification_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n[INFO] Results saved to {out_path}")
    print(results_df.to_string(index=False))

    print("\n-- Done: 05_classification.py -------------------")
    spark.stop()


if __name__ == "__main__":
    main()
