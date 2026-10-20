"""
04_feature_engineering.py
--------------------------
Stage 4: Join all per-account graph signals into a single feature DataFrame,
then join with ground-truth labels.

Features assembled (per account):
  - in_degree, out_degree       (from retweet graph)
  - pagerank                    (from retweet graph)
  - triangle_count              (from symmetrised retweet graph)
  - mutual_retweet_count        (count of mutual-retweet partners)
  - shared_target_count         (how many accounts co-amplify the same target)
  - component_id                (from co-activity graph)
  - community_id                (LPA community, from co-activity graph)
  - community_size              (number of accounts in same LPA community)
  - community_bot_ratio         (fraction of bots in community — target signal)
  - followers_count, friends_count, statuses_count  (from user profile)

Also adds binary label:  label_bin  (1 = bot, 0 = genuine)

Output: data/processed/features.parquet
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

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("BotDetection-Features")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ── Load base user table ───────────────────────────────────────────────────
    users = spark.read.parquet(str(PROC_DIR / "users_clean.parquet")) \
        .select("id", "label", "followers_count", "friends_count", "statuses_count") \
        .fillna(0, subset=["followers_count", "friends_count", "statuses_count"])

    # ── Degrees ────────────────────────────────────────────────────────────────
    degrees = spark.read.parquet(str(PROC_DIR / "rt_degrees.parquet"))

    # ── PageRank ───────────────────────────────────────────────────────────────
    pagerank = spark.read.parquet(str(PROC_DIR / "rt_pagerank.parquet"))

    # ── Triangle count ─────────────────────────────────────────────────────────
    triangles = spark.read.parquet(str(PROC_DIR / "rt_triangles.parquet"))

    # ── Mutual retweet: count of partners per user ─────────────────────────────
    mutual_raw = spark.read.parquet(str(PROC_DIR / "rt_motif_mutual.parquet"))
    mutual_a = mutual_raw.groupBy("user_a").agg(F.count("*").alias("mutual_retweet_count")) \
        .withColumnRenamed("user_a", "id")
    mutual_b = mutual_raw.groupBy("user_b").agg(F.count("*").alias("mutual_retweet_count_b")) \
        .withColumnRenamed("user_b", "id")
    mutual_count = mutual_a.join(mutual_b, on="id", how="outer") \
        .fillna(0) \
        .withColumn("mutual_retweet_count",
                    F.col("mutual_retweet_count") + F.col("mutual_retweet_count_b")) \
        .drop("mutual_retweet_count_b")

    # ── Shared-target: count of distinct co-amplifiers per user ───────────────
    shared_raw = spark.read.parquet(str(PROC_DIR / "rt_motif_shared_target.parquet"))
    shared_a = shared_raw.groupBy("user_a").agg(F.countDistinct("user_b").alias("shared_target_count")) \
        .withColumnRenamed("user_a", "id")
    shared_b = shared_raw.groupBy("user_b").agg(F.countDistinct("user_a").alias("shared_target_count_b")) \
        .withColumnRenamed("user_b", "id")
    shared_count = shared_a.join(shared_b, on="id", how="outer") \
        .fillna(0) \
        .withColumn("shared_target_count",
                    F.col("shared_target_count") + F.col("shared_target_count_b")) \
        .drop("shared_target_count_b")

    # ── Connected components ───────────────────────────────────────────────────
    components = spark.read.parquet(str(PROC_DIR / "ca_components.parquet")) \
        .withColumnRenamed("component", "component_id")

    # ── LPA communities ────────────────────────────────────────────────────────
    communities = spark.read.parquet(str(PROC_DIR / "ca_communities.parquet")) \
        .withColumnRenamed("community", "community_id")

    # Community size
    comm_size = communities.groupBy("community_id") \
        .agg(F.count("*").alias("community_size"))

    # ── Assemble feature table ─────────────────────────────────────────────────
    feat = users \
        .join(degrees,      on="id", how="left") \
        .join(pagerank,     on="id", how="left") \
        .join(triangles,    on="id", how="left") \
        .join(mutual_count, on="id", how="left") \
        .join(shared_count, on="id", how="left") \
        .join(components,   on="id", how="left") \
        .join(communities,  on="id", how="left") \
        .join(comm_size,    on="community_id", how="left")

    feat = feat.fillna(0, subset=[
        "inDegree", "outDegree", "pagerank", "triangle_count",
        "mutual_retweet_count", "shared_target_count", "community_size",
    ])

    # ── Community bot ratio ────────────────────────────────────────────────────
    # For each community, what fraction of members are bots?
    comm_bot = (
        feat.groupBy("community_id")
        .agg(
            F.mean(F.when(F.col("label") == "bot", 1).otherwise(0))
             .alias("community_bot_ratio")
        )
    )
    feat = feat.join(comm_bot, on="community_id", how="left").fillna(0, subset=["community_bot_ratio"])

    # ── Binary label ──────────────────────────────────────────────────────────
    feat = feat.withColumn(
        "label_bin",
        F.when(F.col("label") == "bot", 1).otherwise(0)
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = str(PROC_DIR / "features.parquet")
    feat.write.mode("overwrite").parquet(out_path)

    # Summary
    print("\n-- Feature table summary ------------------------")
    feat.groupBy("label").count().show()
    print("Columns:", feat.columns)
    print(f"Total rows: {feat.count()}")
    print("\n-- Done: 04_feature_engineering.py --------------")

    spark.stop()


if __name__ == "__main__":
    main()
