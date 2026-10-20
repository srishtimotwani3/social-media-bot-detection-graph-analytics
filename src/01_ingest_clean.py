"""
01_ingest_clean.py
------------------
Stage 1: Load Cresci-2017 CSV files into Spark DataFrames,
clean them, attach ground-truth labels, and save as Parquet.

Outputs (data/processed/):
  users_clean.parquet   — one row per account, with `label` column
  tweets_clean.parquet  — one row per tweet, with `user_id` and `label`
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
from pyspark.sql.types import LongType, StringType, IntegerType, TimestampType

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR      = PROJECT_ROOT / "data" / "raw"  / "cresci-2017"
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset categories and their labels ──────────────────────────────────────
CATEGORIES = {
    "genuine_accounts.csv"      : "genuine",
    "social_spambots_1.csv"     : "bot",
    "social_spambots_2.csv"     : "bot",
    "social_spambots_3.csv"     : "bot",
    "traditional_spambots_1.csv": "bot",
    "traditional_spambots_2.csv": "bot",
    "traditional_spambots_3.csv": "bot",
    "traditional_spambots_4.csv": "bot",
    "fake_followers.csv"        : "bot",
}

# ── Spark session ─────────────────────────────────────────────────────────────
def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("BotDetection-Ingest")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


# ── User schema (key columns only; rest inferred) ────────────────────────────
USER_KEEP = [
    "id", "screen_name", "name", "followers_count", "friends_count",
    "listed_count", "favourites_count", "statuses_count",
    "created_at", "verified", "default_profile", "default_profile_image",
    "geo_enabled", "lang",
]

TWEET_KEEP = [
    "id", "user_id", "text", "created_at",
    "retweet_count", "favorite_count",
    "in_reply_to_user_id", "retweeted_status_id",
    "entities_urls", "entities_hashtags",
]


def load_category(spark: SparkSession, cat_dir: pathlib.Path, label: str, kind: str):
    """Load users.csv or tweets.csv from a category directory."""
    csv_path = str(cat_dir / f"{kind}.csv")
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "false")       # we'll cast manually
        .option("multiLine", "true")
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .csv(csv_path)
    )
    df = df.withColumn("label", F.lit(label))
    return df


def clean_users(df):
    """Select, cast, and clean the users DataFrame."""
    available = set(df.columns)
    keep = [c for c in USER_KEEP if c in available] + ["label"]
    df = df.select(keep)

    # Cast numeric fields
    for col in ["followers_count", "friends_count", "listed_count",
                "favourites_count", "statuses_count"]:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast(LongType()))

    df = df.withColumn("id", F.col("id").cast(LongType()))

    # Boolean-ish flags
    for col in ["verified", "default_profile", "default_profile_image", "geo_enabled"]:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast("boolean"))

    # Drop rows with no user id
    df = df.filter(F.col("id").isNotNull())

    # Deduplicate on user id (keep first occurrence)
    from pyspark.sql.window import Window
    w = Window.partitionBy("id").orderBy(F.monotonically_increasing_id())
    df = df.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")

    return df


def clean_tweets(df):
    """Select, cast, and clean the tweets DataFrame."""
    available = set(df.columns)

    # Flexible column mapping — different Cresci files use different col names
    col_map = {}
    if "user_id" not in available and "user.id" in available:
        df = df.withColumnRenamed("user.id", "user_id")
    if "retweeted_status_id" not in available:
        # try common variants
        for alt in ["retweeted_status.id", "retweeted_status_id_str"]:
            if alt in available:
                df = df.withColumnRenamed(alt, "retweeted_status_id")
                break

    available = set(df.columns)
    keep = [c for c in TWEET_KEEP if c in available] + ["label"]
    df = df.select(keep)

    for col in ["id", "user_id", "retweeted_status_id", "in_reply_to_user_id"]:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast(LongType()))
    for col in ["retweet_count", "favorite_count"]:
        if col in df.columns:
            df = df.withColumn(col, F.col(col).cast(IntegerType()))

    df = df.filter(F.col("id").isNotNull() & F.col("user_id").isNotNull())
    return df


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    user_frames  = []
    tweet_frames = []

    for cat_subdir, label in CATEGORIES.items():
        cat_dir = RAW_DIR / cat_subdir
        if not cat_dir.exists():
            print(f"[WARN] Missing category dir: {cat_dir} - skipping")
            continue

        print(f"[INFO] Loading {cat_subdir}  (label={label})")

        # Users
        if (cat_dir / "users.csv").exists():
            u = load_category(spark, cat_dir, label, "users")
            u = clean_users(u)
            user_frames.append(u)
            print(f"       users  -> {u.count()} rows")

        # Tweets
        if (cat_dir / "tweets.csv").exists():
            t = load_category(spark, cat_dir, label, "tweets")
            t = clean_tweets(t)
            tweet_frames.append(t)
            print(f"       tweets -> {t.count()} rows")

    if not user_frames:
        raise RuntimeError(
            "No data loaded. Run 00_download_data.py first or place "
            "the Cresci-2017 files under data/raw/cresci-2017/"
        )

    # Union all categories
    from functools import reduce
    from pyspark.sql import DataFrame

    def safe_union(frames):
        """Union DataFrames, aligning columns."""
        all_cols = set()
        for f in frames:
            all_cols.update(f.columns)
        aligned = []
        for f in frames:
            missing = all_cols - set(f.columns)
            for mc in missing:
                f = f.withColumn(mc, F.lit(None))
            aligned.append(f.select(sorted(all_cols)))
        return reduce(DataFrame.union, aligned)

    users_df  = safe_union(user_frames)
    tweets_df = safe_union(tweet_frames)

    # Save as Parquet
    out_users  = str(PROC_DIR / "users_clean.parquet")
    out_tweets = str(PROC_DIR / "tweets_clean.parquet")

    print(f"\n[INFO] Writing {out_users}")
    users_df.write.mode("overwrite").parquet(out_users)

    print(f"[INFO] Writing {out_tweets}")
    tweets_df.write.mode("overwrite").parquet(out_tweets)

    # Summary
    print("\n-- Summary -------------------------------------")
    users_df.groupBy("label").count().show()
    tweets_df.groupBy("label").count().show()
    print("-- Done: 01_ingest_clean.py ---------------------")

    spark.stop()


if __name__ == "__main__":
    main()
