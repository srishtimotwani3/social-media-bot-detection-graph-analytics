"""
spark_utils.py
--------------
Shared Spark session factory and common helpers used across pipeline scripts.
Import this instead of duplicating SparkSession config in every script.
"""

import os
import sys
import pathlib

# ── Fix for Windows: ensure Spark workers use the same Python executable ──────
# On Windows, 'python' may resolve to the Microsoft Store stub.
# PYSPARK_PYTHON / PYSPARK_DRIVER_PYTHON must point to the real interpreter.
_py = sys.executable
os.environ["PYSPARK_PYTHON"] = _py
os.environ["PYSPARK_DRIVER_PYTHON"] = _py

# ── Fix for Windows: HADOOP_HOME / winutils ──────────────────────────────────
_hadoop_home = str(pathlib.Path(__file__).resolve().parents[1] / "hadoop")
os.environ["HADOOP_HOME"] = _hadoop_home
_hadoop_bin = str(pathlib.Path(_hadoop_home) / "bin")
if _hadoop_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _hadoop_bin + os.pathsep + os.environ.get("PATH", "")

# Tell PySpark to pull the GraphFrames JAR from Maven when submitting
GRAPHFRAMES_PKG = "graphframes:graphframes:0.8.3-spark3.5-s_2.12"

os.environ.setdefault(
    "PYSPARK_SUBMIT_ARGS",
    f"--packages {GRAPHFRAMES_PKG} pyspark-shell",
)


def get_spark(app_name: str = "BotDetection", use_graphframes: bool = False,
              shuffle_partitions: int = 8, driver_memory: str = "4g"):
    """
    Create (or retrieve) a SparkSession.

    Parameters
    ----------
    app_name : str
    use_graphframes : bool
        If True, adds the GraphFrames package to spark.jars.packages.
    shuffle_partitions : int
        spark.sql.shuffle.partitions (lower = faster on single machine)
    driver_memory : str
        e.g. "4g", "8g"
    """
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        # Avoid Spark UI port conflicts when running multiple scripts
        .config("spark.ui.port", "4041")
    )

    if use_graphframes:
        builder = builder.config("spark.jars.packages", GRAPHFRAMES_PKG)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def proc_dir() -> pathlib.Path:
    return project_root() / "data" / "processed"


def results_dir() -> pathlib.Path:
    d = project_root() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d
