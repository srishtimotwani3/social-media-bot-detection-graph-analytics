"""
07_scalability.py
-----------------
Stage 7: Scalability experiment — prove the pipeline scales.

Runs the full graph-build + analytics pipeline at three data fractions:
  10%, 50%, 100%

For each fraction:
  - Samples the tweets and users DataFrames
  - Rebuilds graphs (edge/vertex counts)
  - Runs PageRank + LPA
  - Records wall-clock time

Output: results/metrics/scalability.csv
        results/plots/scalability_plot.png
"""

import sys
import pathlib
import time
import os
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

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

os.environ.setdefault(
    "PYSPARK_SUBMIT_ARGS",
    "--packages graphframes:graphframes:0.8.3-spark3.5-s_2.12 pyspark-shell",
)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from graphframes import GraphFrame

PROJECT_ROOT    = pathlib.Path(__file__).resolve().parents[1]
PROC_DIR        = PROJECT_ROOT / "data" / "processed"
RESULTS_METRICS = PROJECT_ROOT / "results" / "metrics"
RESULTS_PLOTS   = PROJECT_ROOT / "results" / "plots"
RESULTS_METRICS.mkdir(parents=True, exist_ok=True)
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)

FRACTIONS = [0.10, 0.50, 1.00]
MIN_CO_OCCUR = 2
TIME_BUCKET_SECS = 3600


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("BotDetection-Scalability")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12")
        .getOrCreate()
    )


def run_fraction(spark, tweets_full, users_full, frac: float) -> dict:
    """Run graph build + PageRank + LPA on `frac` fraction of the data."""
    seed = 42
    tweets = tweets_full.sample(fraction=frac, seed=seed)
    users  = users_full.sample(fraction=frac, seed=seed)

    n_tweets = tweets.count()
    n_users  = users.count()
    print(f"\n[INFO] Fraction={frac:.0%} -> tweets={n_tweets}, users={n_users}")

    t0 = time.time()

    # ── Retweet graph ──────────────────────────────────────────────────────────
    retweets = tweets.filter(F.col("retweeted_status_id").isNotNull()) \
        .select(F.col("user_id").alias("src"), F.col("retweeted_status_id").alias("rt_id"))
    tweet_author = tweets.select(F.col("id").alias("tweet_id"), F.col("user_id").alias("dst"))
    rt_edges = (
        retweets.join(tweet_author, retweets.rt_id == tweet_author.tweet_id, "inner")
        .filter(F.col("src") != F.col("dst"))
        .groupBy("src", "dst").agg(F.count("*").alias("weight"))
        .filter(F.col("weight") >= 1)
    )
    rt_vids = rt_edges.select(F.col("src").alias("id")).union(rt_edges.select(F.col("dst").alias("id"))).distinct()
    rt_verts = rt_vids.join(users.select("id", "label"), on="id", how="left").fillna({"label": "unknown"})

    if rt_verts.count() == 0 or rt_edges.count() == 0:
        print(f"[WARN] No retweet graph at fraction={frac}")
        elapsed = time.time() - t0
        return {"fraction": frac, "n_tweets": n_tweets, "n_users": n_users,
                "rt_edges": 0, "ca_edges": 0, "elapsed_sec": elapsed}

    g_rt = GraphFrame(rt_verts, rt_edges)

    # PageRank
    chk_dir = PROJECT_ROOT / "data" / "checkpoints"
    chk_dir.mkdir(parents=True, exist_ok=True)
    spark.sparkContext.setCheckpointDir(str(chk_dir))
    pr = g_rt.pageRank(resetProbability=0.15, maxIter=5)
    pr.vertices.count()  # force materialisation

    # ── Co-activity graph ──────────────────────────────────────────────────────
    df = tweets.withColumn("ts_unix",
        F.coalesce(
            F.unix_timestamp("created_at", "EEE MMM dd HH:mm:ss Z yyyy"),
            F.unix_timestamp("created_at"),
        )
    ).withColumn("time_bucket", (F.col("ts_unix") / TIME_BUCKET_SECS).cast("long") * TIME_BUCKET_SECS)

    hashtags = (
        df.select("user_id", "time_bucket",
                  F.regexp_extract_all(F.lower(F.coalesce(F.col("text"), F.lit(""))),
                                       F.lit(r"#(\w+)"), 1).alias("tags"))
        .filter(F.size("tags") > 0)
        .select("user_id", "time_bucket", F.explode("tags").alias("item"))
    )

    a = hashtags.alias("a")
    b = hashtags.alias("b")
    ca_edges = (
        a.join(b,
               (F.col("a.item")        == F.col("b.item")) &
               (F.col("a.time_bucket") == F.col("b.time_bucket")) &
               (F.col("a.user_id")     <  F.col("b.user_id")))
        .select(F.col("a.user_id").alias("src"), F.col("b.user_id").alias("dst"))
        .groupBy("src", "dst").agg(F.count("*").alias("weight"))
        .filter(F.col("weight") >= MIN_CO_OCCUR)
    )
    ca_vids = ca_edges.select(F.col("src").alias("id")).union(ca_edges.select(F.col("dst").alias("id"))).distinct()
    ca_verts = ca_vids.join(users.select("id", "label"), on="id", how="left").fillna({"label": "unknown"})

    n_ca_edges = ca_edges.count()
    n_rt_edges = rt_edges.count()

    if ca_verts.count() > 0 and n_ca_edges > 0:
        g_ca = GraphFrame(ca_verts.drop("label"), ca_edges)
        lpa  = g_ca.labelPropagation(maxIter=5)
        lpa.count()  # materialise

    elapsed = time.time() - t0
    print(f"       rt_edges={n_rt_edges}, ca_edges={n_ca_edges}, elapsed={elapsed:.1f}s")

    return {
        "fraction": frac,
        "n_tweets": n_tweets,
        "n_users":  n_users,
        "rt_edges": n_rt_edges,
        "ca_edges": n_ca_edges,
        "elapsed_sec": round(elapsed, 2),
    }


def plot_scalability(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(df["fraction"] * 100, df["elapsed_sec"], "o-", color="steelblue", linewidth=2)
    axes[0].set_xlabel("Data fraction (%)")
    axes[0].set_ylabel("Wall-clock time (seconds)")
    axes[0].set_title("Runtime vs. Data Size")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["fraction"] * 100, df["rt_edges"], "s-", color="coral",
                 linewidth=2, label="Retweet edges")
    axes[1].plot(df["fraction"] * 100, df["ca_edges"], "^-", color="green",
                 linewidth=2, label="Co-activity edges")
    axes[1].set_xlabel("Data fraction (%)")
    axes[1].set_ylabel("Edge count")
    axes[1].set_title("Graph Size vs. Data Size")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = str(RESULTS_PLOTS / "scalability_plot.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] Scalability plot saved -> {out}")


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    tweets_full = spark.read.parquet(str(PROC_DIR / "tweets_clean.parquet"))
    users_full  = spark.read.parquet(str(PROC_DIR / "users_clean.parquet"))

    records = []
    for frac in FRACTIONS:
        rec = run_fraction(spark, tweets_full, users_full, frac)
        records.append(rec)

    df = pd.DataFrame(records)
    out_csv = str(RESULTS_METRICS / "scalability.csv")
    df.to_csv(out_csv, index=False)
    print("\n-- Scalability Results -------------------------")
    print(df.to_string(index=False))

    plot_scalability(df)

    print("\n-- Done: 07_scalability.py ----------------------")
    spark.stop()


if __name__ == "__main__":
    main()
