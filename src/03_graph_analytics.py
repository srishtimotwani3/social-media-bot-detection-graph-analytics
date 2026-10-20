"""
03_graph_analytics.py
---------------------
Stage 3: Run GraphFrames algorithms on the two graphs.

Algorithms applied:
  Retweet graph (directed):
    - inDegrees / outDegrees
    - PageRank (resetProbability=0.15, maxIter=10)
    - TriangleCount (undirected version via dropDuplicates)
    - Motif: mutual retweet   (a)→(b); (b)→(a)
    - Motif: shared-target    (a)→(c); (b)→(c)

  Co-activity graph (undirected):
    - ConnectedComponents
    - LabelPropagation (maxIter=5)

Outputs (data/processed/):
  rt_degrees.parquet
  rt_pagerank.parquet
  rt_triangles.parquet
  rt_motif_mutual.parquet
  rt_motif_shared_target.parquet
  ca_components.parquet
  ca_communities.parquet
"""

import sys
import pathlib
import os

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_hadoop_home = str(PROJECT_ROOT / "hadoop")
os.environ["HADOOP_HOME"] = _hadoop_home
_hadoop_bin = str(PROJECT_ROOT / "hadoop" / "bin")
if _hadoop_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _hadoop_bin + os.pathsep + os.environ.get("PATH", "")

# GraphFrames requires the JAR on the classpath via spark.jars.packages
os.environ.setdefault(
    "PYSPARK_SUBMIT_ARGS",
    "--packages graphframes:graphframes:0.8.3-spark3.5-s_2.12 pyspark-shell",
)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from graphframes import GraphFrame

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("BotDetection-GraphAnalytics")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config(
            "spark.jars.packages",
            "graphframes:graphframes:0.8.3-spark3.5-s_2.12",
        )
        .getOrCreate()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Retweet Graph Analytics
# ─────────────────────────────────────────────────────────────────────────────

def retweet_analytics(spark):
    print("[INFO] Retweet graph analytics...")

    v = spark.read.parquet(str(PROC_DIR / "retweet_vertices.parquet"))
    e = spark.read.parquet(str(PROC_DIR / "retweet_edges.parquet"))

    # GraphFrames requires 'id' on vertices and 'src'/'dst' on edges
    g = GraphFrame(v, e)

    # ── Degrees ────────────────────────────────────────────────────────────────
    in_deg  = g.inDegrees
    out_deg = g.outDegrees
    degrees = (
        in_deg.join(out_deg, on="id", how="outer")
        .fillna(0, subset=["inDegree", "outDegree"])
    )
    degrees.write.mode("overwrite").parquet(str(PROC_DIR / "rt_degrees.parquet"))
    print(f"       degrees saved ({degrees.count()} rows)")

    # ── PageRank ───────────────────────────────────────────────────────────────
    pr = g.pageRank(resetProbability=0.15, maxIter=10)
    pr.vertices.select("id", "pagerank") \
        .write.mode("overwrite").parquet(str(PROC_DIR / "rt_pagerank.parquet"))
    print("       pagerank saved")

    # ── Triangle Count (on symmetrised graph) ─────────────────────────────────
    # Make undirected by adding reverse edges then dedup
    e_sym = (
        e.select("src", "dst")
        .union(e.select(F.col("dst").alias("src"), F.col("src").alias("dst")))
        .distinct()
        .withColumn("weight", F.lit(1))
    )
    g_sym = GraphFrame(v, e_sym)
    tc = g_sym.triangleCount()
    tc.select("id", "count").withColumnRenamed("count", "triangle_count") \
        .write.mode("overwrite").parquet(str(PROC_DIR / "rt_triangles.parquet"))
    print("       triangle count saved")

    # ── Motif: mutual retweet  (a)-[e1]->(b); (b)-[e2]->(a) ──────────────────
    motif_mutual = g.find("(a)-[e1]->(b); (b)-[e2]->(a)")
    mutual = motif_mutual.select(
        F.col("a.id").alias("user_a"),
        F.col("b.id").alias("user_b"),
    ).distinct()
    mutual.write.mode("overwrite").parquet(str(PROC_DIR / "rt_motif_mutual.parquet"))
    print(f"       mutual retweet pairs: {mutual.count()}")

    # ── Motif: shared-target amplification  (a)-[e1]->(c); (b)-[e2]->(c) ─────
    motif_shared = g.find("(a)-[e1]->(c); (b)-[e2]->(c)") \
        .filter(F.col("a.id") != F.col("b.id"))
    shared = motif_shared.select(
        F.col("a.id").alias("user_a"),
        F.col("b.id").alias("user_b"),
        F.col("c.id").alias("target"),
    ).distinct()
    shared.write.mode("overwrite").parquet(str(PROC_DIR / "rt_motif_shared_target.parquet"))
    print(f"       shared-target triplets: {shared.count()}")


# ─────────────────────────────────────────────────────────────────────────────
# Co-activity Graph Analytics
# ─────────────────────────────────────────────────────────────────────────────

def coactivity_analytics(spark):
    print("[INFO] Co-activity graph analytics...")

    # Drop existing 'label' (ground-truth) to prevent column collision with LPA's 'label'
    v = spark.read.parquet(str(PROC_DIR / "coactivity_vertices.parquet")).drop("label")
    e = spark.read.parquet(str(PROC_DIR / "coactivity_edges.parquet"))

    g = GraphFrame(v, e)

    # ── Connected Components ───────────────────────────────────────────────────
    chk_dir = PROJECT_ROOT / "data" / "checkpoints"
    chk_dir.mkdir(parents=True, exist_ok=True)
    spark.sparkContext.setCheckpointDir(str(chk_dir))
    cc = g.connectedComponents()
    cc.select("id", "component") \
        .write.mode("overwrite").parquet(str(PROC_DIR / "ca_components.parquet"))
    print(f"       connected components saved")

    # ── Label Propagation ─────────────────────────────────────────────────────
    lpa = g.labelPropagation(maxIter=5)
    lpa.select("id", "label") \
        .withColumnRenamed("label", "community") \
        .write.mode("overwrite").parquet(str(PROC_DIR / "ca_communities.parquet"))
    print("       label propagation (communities) saved")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")

    retweet_analytics(spark)
    coactivity_analytics(spark)

    print("\n-- Done: 03_graph_analytics.py ------------------")
    spark.stop()


if __name__ == "__main__":
    main()
