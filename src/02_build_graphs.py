"""
02_build_graphs.py
------------------
Stage 2: Build two GraphFrames from the cleaned data.

Graph 1: Retweet Graph (directed)
  - Vertex: user account (id, label, screen_name, ...)
  - Edge  : u → v  (user u retweeted user v)
  - Weight: retweet count (aggregated)

Graph 2: Co-activity Graph (undirected, weighted)
  - Edge u—v: both users posted the same hashtag or URL within the same
              1-hour time bucket, at least MIN_CO_OCCUR times.
  - Built via self-join on (item, time_bucket).
  - Weak edges (weight < MIN_CO_OCCUR) are dropped to avoid combinatorial blowup.

Outputs (data/processed/):
  retweet_vertices.parquet
  retweet_edges.parquet
  coactivity_vertices.parquet
  coactivity_edges.parquet
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

import re
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import LongType

# ── Config ────────────────────────────────────────────────────────────────────
MIN_CO_OCCUR    = 2     # minimum shared items for a co-activity edge
TIME_BUCKET_H   = 1     # time bucket width in hours for co-activity
MIN_RT_COUNT    = 1     # minimum retweet count to keep an edge

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("BotDetection-GraphBuild")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .getOrCreate()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Graph 1: Retweet Graph
# ─────────────────────────────────────────────────────────────────────────────

def build_retweet_graph(spark, tweets_df, users_df):
    """
    Build a directed retweet graph.

    An edge (u → v) exists if user u retweeted a tweet whose author is v,
    identified by retweeted_status_id matching a tweet id in the dataset
    (or via a lookup of the original author).
    """
    print("[INFO] Building retweet graph...")

    # Tweets that are retweets (retweeted_status_id is set)
    retweets = tweets_df.filter(
        F.col("retweeted_status_id").isNotNull()
    ).select(
        F.col("user_id").alias("src"),
        F.col("retweeted_status_id").alias("rt_status_id"),
        F.col("created_at").alias("rt_ts"),
    )

    # Map retweeted_status_id → original author
    tweet_author = tweets_df.select(
        F.col("id").alias("tweet_id"),
        F.col("user_id").alias("dst"),
    )

    edges_raw = (
        retweets
        .join(tweet_author, retweets.rt_status_id == tweet_author.tweet_id, "inner")
        .filter(F.col("src") != F.col("dst"))          # no self-loops
        .select("src", "dst")
    )

    # Aggregate: count retweets per (src, dst) pair
    edges = (
        edges_raw
        .groupBy("src", "dst")
        .agg(F.count("*").alias("weight"))
        .filter(F.col("weight") >= MIN_RT_COUNT)
    )

    # Vertices: all users appearing in edges, enriched with profile data
    src_ids = edges.select(F.col("src").alias("id"))
    dst_ids = edges.select(F.col("dst").alias("id"))
    vertex_ids = src_ids.union(dst_ids).distinct()

    vertices = (
        vertex_ids
        .join(
            users_df.select("id", "label", "screen_name",
                            "followers_count", "friends_count", "statuses_count"),
            on="id",
            how="left",
        )
        .fillna({"label": "unknown", "screen_name": "unknown"})
    )

    print(f"       retweet edges    : {edges.count()}")
    print(f"       retweet vertices : {vertices.count()}")
    return vertices, edges


# ─────────────────────────────────────────────────────────────────────────────
# Graph 2: Co-activity Graph
# ─────────────────────────────────────────────────────────────────────────────

def extract_hashtags_urls(tweets_df):
    """
    Extract (user_id, item, time_bucket) rows from tweets.
    item = hashtag text or URL, normalised to lowercase.
    time_bucket = UNIX timestamp floored to TIME_BUCKET_H hours.
    """
    BUCKET_SECS = TIME_BUCKET_H * 3600

    # Try to parse created_at as timestamp (Twitter format: "Mon Jan 01 00:00:00 +0000 2020")
    df = tweets_df.withColumn(
        "ts_unix",
        F.coalesce(
            F.unix_timestamp("created_at", "EEE MMM dd HH:mm:ss Z yyyy"),
            F.unix_timestamp("created_at"),          # ISO fallback
        )
    ).withColumn(
        "time_bucket",
        (F.col("ts_unix") / BUCKET_SECS).cast("long") * BUCKET_SECS
    ).filter(F.col("time_bucket").isNotNull())

    # ── Hashtags ──────────────────────────────────────────────────────────────
    hashtags = (
        df.select(
            F.col("user_id"),
            F.col("time_bucket"),
            F.regexp_extract_all(
                F.lower(F.coalesce(F.col("text"), F.lit(""))),
                F.lit(r"#(\w+)"),
                1,
            ).alias("tags"),
        )
        .filter(F.size("tags") > 0)
        .select(
            F.col("user_id"),
            F.col("time_bucket"),
            F.explode("tags").alias("item"),
        )
    )

    # ── URLs ──────────────────────────────────────────────────────────────────
    urls = (
        df.select(
            F.col("user_id"),
            F.col("time_bucket"),
            F.regexp_extract_all(
                F.lower(F.coalesce(F.col("text"), F.lit(""))),
                F.lit(r"https?://[^\s\"',]+"),
                0,
            ).alias("url_list"),
        )
        .filter(F.size("url_list") > 0)
        .select(
            F.col("user_id"),
            F.col("time_bucket"),
            F.explode("url_list").alias("item"),
        )
    )

    items = hashtags.union(urls).distinct()
    return items


def build_coactivity_graph(spark, tweets_df, users_df):
    """
    Build an undirected, weighted co-activity graph.
    Edge (u, v) weight = number of (item, time_bucket) pairs they share.
    """
    print("[INFO] Building co-activity graph...")

    items = extract_hashtags_urls(tweets_df)

    # Self-join: find pairs (u, v) sharing (item, time_bucket)
    a = items.alias("a")
    b = items.alias("b")

    pairs = (
        a.join(
            b,
            (F.col("a.item")        == F.col("b.item")) &
            (F.col("a.time_bucket") == F.col("b.time_bucket")) &
            (F.col("a.user_id")     <  F.col("b.user_id")),   # avoid duplicates
        )
        .select(
            F.col("a.user_id").alias("src"),
            F.col("b.user_id").alias("dst"),
        )
    )

    edges = (
        pairs
        .groupBy("src", "dst")
        .agg(F.count("*").alias("weight"))
        .filter(F.col("weight") >= MIN_CO_OCCUR)
    )

    # Vertices
    src_ids = edges.select(F.col("src").alias("id"))
    dst_ids = edges.select(F.col("dst").alias("id"))
    vertex_ids = src_ids.union(dst_ids).distinct()

    vertices = (
        vertex_ids
        .join(
            users_df.select("id", "label", "screen_name",
                            "followers_count", "friends_count", "statuses_count"),
            on="id",
            how="left",
        )
        .fillna({"label": "unknown", "screen_name": "unknown"})
    )

    print(f"       co-activity edges    : {edges.count()}")
    print(f"       co-activity vertices : {vertices.count()}")
    return vertices, edges


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    users_df  = spark.read.parquet(str(PROC_DIR / "users_clean.parquet"))
    tweets_df = spark.read.parquet(str(PROC_DIR / "tweets_clean.parquet"))

    # ── Retweet graph ──────────────────────────────────────────────────────────
    rt_v, rt_e = build_retweet_graph(spark, tweets_df, users_df)
    rt_v.write.mode("overwrite").parquet(str(PROC_DIR / "retweet_vertices.parquet"))
    rt_e.write.mode("overwrite").parquet(str(PROC_DIR / "retweet_edges.parquet"))

    # ── Co-activity graph ──────────────────────────────────────────────────────
    ca_v, ca_e = build_coactivity_graph(spark, tweets_df, users_df)
    ca_v.write.mode("overwrite").parquet(str(PROC_DIR / "coactivity_vertices.parquet"))
    ca_e.write.mode("overwrite").parquet(str(PROC_DIR / "coactivity_edges.parquet"))

    print("\n-- Done: 02_build_graphs.py ---------------------")
    spark.stop()


if __name__ == "__main__":
    main()
