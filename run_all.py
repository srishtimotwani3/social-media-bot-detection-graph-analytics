"""
run_all.py
----------
End-to-end runner: executes all pipeline stages in order.
Run this after placing the Cresci-2017 data under data/raw/cresci-2017/.

Usage:
    python run_all.py [--skip-download]
"""

import argparse
import subprocess
import sys
import pathlib
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
SRC_DIR      = PROJECT_ROOT / "src"

STAGES = [
    ("00_download_data.py",       "Dataset download / verification"),
    ("01_ingest_clean.py",        "Ingest & clean CSVs → Parquet"),
    ("02_build_graphs.py",        "Build retweet + co-activity graphs"),
    ("03_graph_analytics.py",     "PageRank, LPA, CC, TriangleCount, Motifs"),
    ("04_feature_engineering.py", "Assemble per-account feature table"),
    ("05_classification.py",      "Train Random Forest (with/without graph features)"),
    ("06_evaluation.py",          "Detailed evaluation & community purity"),
    ("07_scalability.py",         "Scalability experiment (10%/50%/100%)"),
    ("08_visualization.py",       "Export subgraph plots + GEXF"),
]


def run_stage(script: str, desc: str) -> bool:
    path = str(SRC_DIR / script)
    print(f"\n{'='*60}")
    print(f"  STAGE: {desc}")
    print(f"  Script: {script}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, path], cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  [PASS] Completed in {elapsed:.1f}s")
        return True
    else:
        print(f"  [FAIL] FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the full bot-detection pipeline.")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip the dataset download step (data already present)")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Start from stage N (0-indexed), skipping earlier stages")
    args = parser.parse_args()

    stages = STAGES[:]
    if args.skip_download:
        stages = [s for s in stages if s[0] != "00_download_data.py"]

    start = args.start_from
    stages = stages[start:]

    print("\n[START] Social Media Bot Detection Pipeline")
    print(f"   Running {len(stages)} stage(s), starting from stage {start}")

    total_t0 = time.time()
    failed = []
    for script, desc in stages:
        ok = run_stage(script, desc)
        if not ok:
            failed.append(script)
            ans = input(f"\nStage {script} failed. Continue anyway? [y/N] ").strip().lower()
            if ans != "y":
                print("Aborting.")
                break

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"  Pipeline finished in {total_elapsed:.1f}s")
    if failed:
        print(f"  Failed stages: {failed}")
    else:
        print("  All stages completed successfully [PASS]")
    print(f"  Results in: {PROJECT_ROOT / 'results'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
