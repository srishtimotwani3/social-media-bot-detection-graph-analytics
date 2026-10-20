"""
06_evaluation.py
----------------
Stage 6: Detailed evaluation beyond the classification step.

1. Per-class Precision / Recall / F1  (bot vs. genuine)
2. Confusion matrix
3. Community purity: for each detected community, what fraction are bots?
   - Report distribution of bot-ratio across communities
   - Identify "pure bot communities" (bot_ratio >= PURITY_THRESHOLD)
4. Flag analysis: how many accounts are flagged as suspicious by
   each graph signal (high PageRank, dense communities, mutual retweeters)?

Outputs:
  results/metrics/per_class_metrics.csv
  results/metrics/community_purity.csv
  results/metrics/flag_summary.csv
  results/plots/confusion_matrix.png
  results/plots/community_purity_hist.png
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

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer
from pyspark.ml.classification import RandomForestClassifier

PROJECT_ROOT    = pathlib.Path(__file__).resolve().parents[1]
PROC_DIR        = PROJECT_ROOT / "data" / "processed"
RESULTS_METRICS = PROJECT_ROOT / "results" / "metrics"
RESULTS_PLOTS   = PROJECT_ROOT / "results" / "plots"
RESULTS_METRICS.mkdir(parents=True, exist_ok=True)
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)

PURITY_THRESHOLD  = 0.8   # a community is "pure bot" if ≥80% are bots
PAGERANK_CUTOFF   = 0.95  # top 5% by pagerank → suspicious amplifier
DEGREE_CUTOFF     = 0.95  # top 5% by in-degree
PROFILE_FEATURES = ["followers_count", "friends_count", "statuses_count"]
GRAPH_FEATURES   = [
    "inDegree", "outDegree", "pagerank", "triangle_count",
    "mutual_retweet_count", "shared_target_count",
    "community_size", "community_bot_ratio",
]


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("BotDetection-Evaluation")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def build_predictions(spark):
    """Re-train (or reload) and generate predictions for the full dataset."""
    features_df = spark.read.parquet(str(PROC_DIR / "features.parquet")) \
        .filter(F.col("label_bin").isNotNull())
    train, test = features_df.randomSplit([0.8, 0.2], seed=42)
    all_features = PROFILE_FEATURES + GRAPH_FEATURES
    imputer  = Imputer(inputCols=all_features, outputCols=[f"{c}_imp" for c in all_features])
    asm      = VectorAssembler(inputCols=[f"{c}_imp" for c in all_features], outputCol="features_raw")
    scaler   = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    rf       = RandomForestClassifier(featuresCol="features", labelCol="label_bin",
                                      numTrees=100, maxDepth=10, seed=42)
    pipeline = Pipeline(stages=[imputer, asm, scaler, rf])
    model    = pipeline.fit(train)
    return model.transform(test)


def per_class_metrics(preds_pd: pd.DataFrame) -> pd.DataFrame:
    """Compute per-class P/R/F1 from a pandas DataFrame."""
    from sklearn.metrics import classification_report
    report = classification_report(
        preds_pd["label_bin"], preds_pd["prediction"].astype(int),
        target_names=["genuine (0)", "bot (1)"],
        output_dict=True,
    )
    rows = []
    for cls in ["genuine (0)", "bot (1)", "macro avg", "weighted avg"]:
        if cls in report:
            row = {"class": cls}
            row.update(report[cls])
            rows.append(row)
    return pd.DataFrame(rows)


def confusion_matrix_plot(preds_pd: pd.DataFrame):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(preds_pd["label_bin"], preds_pd["prediction"].astype(int))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["genuine", "bot"],
                yticklabels=["genuine", "bot"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (WITH graph features)")
    fig.tight_layout()
    out = str(RESULTS_PLOTS / "confusion_matrix.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved -> {out}")


def community_purity(spark):
    """Analyse bot concentration in LPA communities."""
    feat = spark.read.parquet(str(PROC_DIR / "features.parquet"))

    purity = (
        feat.filter(F.col("community_id").isNotNull())
        .groupBy("community_id")
        .agg(
            F.count("*").alias("community_size"),
            F.sum(F.col("label_bin")).alias("bot_count"),
            F.mean(F.col("label_bin")).alias("bot_ratio"),
        )
        .withColumn(
            "is_pure_bot_community",
            F.col("bot_ratio") >= PURITY_THRESHOLD,
        )
    )

    purity_pd = purity.toPandas()
    purity_pd.to_csv(str(RESULTS_METRICS / "community_purity.csv"), index=False)

    # Plot distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(purity_pd["bot_ratio"], bins=20, color="steelblue", edgecolor="white")
    ax.axvline(PURITY_THRESHOLD, color="red", linestyle="--", label=f"Purity threshold ({PURITY_THRESHOLD})")
    ax.set_xlabel("Bot ratio per community")
    ax.set_ylabel("Number of communities")
    ax.set_title("Community Purity Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(RESULTS_PLOTS / "community_purity_hist.png"), dpi=150)
    plt.close(fig)

    n_pure = purity_pd["is_pure_bot_community"].sum()
    print(f"[INFO] Communities: {len(purity_pd)} total, {n_pure} pure-bot (>={PURITY_THRESHOLD*100:.0f}%)")
    return purity_pd


def flag_summary(spark):
    """Count how many accounts are flagged by each individual graph signal."""
    feat = spark.read.parquet(str(PROC_DIR / "features.parquet"))

    pr_thresh  = feat.approxQuantile("pagerank",  [PAGERANK_CUTOFF], 0.01)[0]
    ind_thresh = feat.approxQuantile("inDegree",  [DEGREE_CUTOFF],  0.01)[0]

    flagged = feat.withColumn("flag_high_pagerank",   F.col("pagerank")             >= pr_thresh) \
                  .withColumn("flag_high_indegree",   F.col("inDegree")             >= ind_thresh) \
                  .withColumn("flag_mutual_rt",       F.col("mutual_retweet_count") >  0) \
                  .withColumn("flag_shared_target",   F.col("shared_target_count")  >  0) \
                  .withColumn("flag_dense_community", F.col("community_bot_ratio")  >= PURITY_THRESHOLD)

    flag_cols = ["flag_high_pagerank", "flag_high_indegree",
                 "flag_mutual_rt", "flag_shared_target", "flag_dense_community"]

    rows = []
    for fc in flag_cols:
        sub = flagged.filter(F.col(fc) == True)
        total     = sub.count()
        actual_bot = sub.filter(F.col("label_bin") == 1).count()
        rows.append({"flag": fc, "flagged_accounts": total,
                     "actual_bots": actual_bot,
                     "precision": actual_bot / total if total else 0.0})

    flag_df = pd.DataFrame(rows)
    flag_df.to_csv(str(RESULTS_METRICS / "flag_summary.csv"), index=False)
    print("\n-- Flag Summary ---------------------------------")
    print(flag_df.to_string(index=False))
    return flag_df


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    # -- Predictions ------------------------------------------------------------
    print("[INFO] Generating predictions...")
    preds = build_predictions(spark)
    preds_pd = preds.select("label_bin", "prediction").toPandas()

    # -- Per-class metrics ------------------------------------------------------
    metrics = per_class_metrics(preds_pd)
    metrics.to_csv(str(RESULTS_METRICS / "per_class_metrics.csv"), index=False)
    print("\n-- Per-Class Metrics -----------------------------")
    print(metrics.to_string(index=False))

    # -- Confusion matrix -------------------------------------------------------
    confusion_matrix_plot(preds_pd)

    # -- Community purity -------------------------------------------------------
    community_purity(spark)

    # -- Flag analysis ----------------------------------------------------------
    flag_summary(spark)

    print("\n-- Done: 06_evaluation.py -----------------------")
    spark.stop()


if __name__ == "__main__":
    main()
