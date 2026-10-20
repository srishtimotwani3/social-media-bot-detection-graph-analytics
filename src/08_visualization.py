"""
08_visualization.py
-------------------
Stage 8: Export suspicious subgraphs for visual inspection.

1. Identify top suspicious accounts:
   - high PageRank + is a bot (ground truth)
   - OR member of a pure-bot community (bot_ratio >= 0.8)

2. Extract a neighbourhood subgraph around those accounts
   from the retweet graph (1-hop neighbours)

3. Export as:
   - PNG  (NetworkX matplotlib plot, up to MAX_NODES nodes)
   - GEXF (Gephi-importable graph file)

4. Also plot:
   - PageRank distribution (genuine vs. bot)
   - In-degree distribution (genuine vs. bot)
   - Feature importance bar chart (from saved CSV)

Outputs:
  results/plots/suspicious_subgraph.png
  results/plots/suspicious_subgraph.gexf
  results/plots/pagerank_dist.png
  results/plots/indegree_dist.png
  results/plots/feature_importance.png
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
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT    = pathlib.Path(__file__).resolve().parents[1]
PROC_DIR        = PROJECT_ROOT / "data" / "processed"
RESULTS_METRICS = PROJECT_ROOT / "results" / "metrics"
RESULTS_PLOTS   = PROJECT_ROOT / "results" / "plots"
RESULTS_PLOTS.mkdir(parents=True, exist_ok=True)

MAX_NODES         = 150    # cap for the subgraph visualisation
PAGERANK_TOP_PERC = 0.05   # top 5% by PageRank
PURITY_THRESHOLD  = 0.8


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data (pandas — small enough after filtering)
# ─────────────────────────────────────────────────────────────────────────────

def load_pandas():
    """Load processed Parquet files into pandas for small-graph operations."""
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .appName("BotDetection-Viz")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    rt_v  = spark.read.parquet(str(PROC_DIR / "retweet_vertices.parquet")).toPandas()
    rt_e  = spark.read.parquet(str(PROC_DIR / "retweet_edges.parquet")).toPandas()
    pr    = spark.read.parquet(str(PROC_DIR / "rt_pagerank.parquet")).toPandas()
    feat  = spark.read.parquet(str(PROC_DIR / "features.parquet")).toPandas()

    spark.stop()
    return rt_v, rt_e, pr, feat


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build NetworkX retweet graph
# ─────────────────────────────────────────────────────────────────────────────

def build_nx_graph(rt_v: pd.DataFrame, rt_e: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in rt_v.iterrows():
        G.add_node(row["id"],
                   label=row.get("label", "unknown"),
                   screen_name=row.get("screen_name", ""))
    for _, row in rt_e.iterrows():
        G.add_edge(row["src"], row["dst"], weight=int(row["weight"]))
    return G


# ─────────────────────────────────────────────────────────────────────────────
# 3. Extract suspicious subgraph
# ─────────────────────────────────────────────────────────────────────────────

def get_suspicious_seeds(feat: pd.DataFrame, pr: pd.DataFrame) -> list:
    if "pagerank" not in feat.columns:
        feat_pr = feat.merge(pr, on="id", how="left")
    else:
        feat_pr = feat
    pr_thresh = feat_pr["pagerank"].quantile(1 - PAGERANK_TOP_PERC)

    suspicious = feat_pr[
        ((feat_pr["pagerank"] >= pr_thresh) & (feat_pr["label_bin"] == 1)) |
        (feat_pr.get("community_bot_ratio", pd.Series(dtype=float)) >= PURITY_THRESHOLD)
    ]["id"].dropna().astype(int).tolist()

    return suspicious


def extract_subgraph(G: nx.DiGraph, seeds: list, max_nodes: int) -> nx.DiGraph:
    """1-hop neighbourhood around seed nodes, capped at max_nodes."""
    nodes = set(seeds)
    for s in seeds:
        if s in G:
            nodes.update(G.predecessors(s))
            nodes.update(G.successors(s))
        if len(nodes) >= max_nodes:
            break
    nodes = list(nodes)[:max_nodes]
    return G.subgraph(nodes).copy()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Plot subgraph
# ─────────────────────────────────────────────────────────────────────────────

def plot_subgraph(Gsub: nx.DiGraph, pr: pd.DataFrame):
    pr_map = dict(zip(pr["id"], pr["pagerank"]))

    color_map = {
        "bot":     "#e74c3c",   # red
        "genuine": "#2ecc71",   # green
        "unknown": "#95a5a6",   # grey
    }
    node_colors = [color_map.get(Gsub.nodes[n].get("label", "unknown"), "#95a5a6")
                   for n in Gsub.nodes()]

    node_sizes = [
        max(100, min(2000, pr_map.get(n, 0.01) * 3000))
        for n in Gsub.nodes()
    ]

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(Gsub, seed=42, k=0.3)
    nx.draw_networkx_nodes(Gsub, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(Gsub, pos, alpha=0.3, arrows=True,
                           arrowsize=10, edge_color="#555555", ax=ax)

    # Label only high-PageRank nodes
    top_pr = sorted(Gsub.nodes(), key=lambda n: pr_map.get(n, 0), reverse=True)[:20]
    labels = {n: Gsub.nodes[n].get("screen_name", str(n)) for n in top_pr}
    nx.draw_networkx_labels(Gsub, pos, labels=labels, font_size=6, ax=ax)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Bot"),
        Patch(facecolor="#2ecc71", label="Genuine"),
        Patch(facecolor="#95a5a6", label="Unknown"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    ax.set_title(f"Suspicious Subgraph (n={Gsub.number_of_nodes()} nodes, "
                 f"{Gsub.number_of_edges()} edges)\nNode size proportional to PageRank, Red = Bot")
    ax.axis("off")
    fig.tight_layout()
    out = str(RESULTS_PLOTS / "suspicious_subgraph.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] Subgraph plot saved -> {out}")


def export_gexf(Gsub: nx.DiGraph):
    """Export to GEXF for Gephi."""
    # Gephi needs string node IDs
    Gmapped = nx.relabel_nodes(Gsub, {n: str(n) for n in Gsub.nodes()})
    out = str(RESULTS_PLOTS / "suspicious_subgraph.gexf")
    nx.write_gexf(Gmapped, out)
    print(f"[INFO] GEXF exported -> {out}")


# -----------------------------------------------------------------------------
# 5. Distribution plots
# -----------------------------------------------------------------------------

def dist_plot(feat: pd.DataFrame, col: str, title: str, out_name: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    for lbl, color in [("genuine", "#2ecc71"), ("bot", "#e74c3c")]:
        sub = feat[feat["label"] == lbl][col].dropna()
        sub = sub[sub > 0]
        if not sub.empty:
            ax.hist(sub, bins=50, alpha=0.6, color=color, label=lbl, density=True)
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    out = str(RESULTS_PLOTS / out_name)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] Distribution plot saved -> {out}")


def feature_importance_plot():
    imp_path = RESULTS_METRICS / "feature_importances.csv"
    if not imp_path.exists():
        print("[WARN] feature_importances.csv not found -- run 05_classification.py first")
        return
    df = pd.read_csv(imp_path).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(df["feature"], df["importance"], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importances")
    fig.tight_layout()
    out = str(RESULTS_PLOTS / "feature_importance.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[INFO] Feature importance plot saved -> {out}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    print("[INFO] Loading data...")
    rt_v, rt_e, pr, feat = load_pandas()

    # Distribution plots (always possible)
    if "pagerank" in feat.columns:
        dist_plot(feat, "pagerank",  "PageRank Distribution: Bot vs. Genuine",
                  "pagerank_dist.png")
    if "inDegree" in feat.columns:
        dist_plot(feat, "inDegree", "In-Degree Distribution: Bot vs. Genuine",
                  "indegree_dist.png")

    feature_importance_plot()

    # Subgraph visualisation (requires retweet graph)
    if rt_e.empty or rt_v.empty:
        print("[WARN] Retweet graph is empty -- skipping subgraph visualisation")
        return

    print("[INFO] Building NetworkX graph...")
    G = build_nx_graph(rt_v, rt_e)

    print("[INFO] Finding suspicious seeds...")
    seeds = get_suspicious_seeds(feat, pr)
    if not seeds:
        print("[WARN] No suspicious seeds found -- using top-PageRank nodes instead")
        seeds = pr.nlargest(20, "pagerank")["id"].astype(int).tolist()

    Gsub = extract_subgraph(G, seeds, MAX_NODES)
    print(f"[INFO] Subgraph: {Gsub.number_of_nodes()} nodes, {Gsub.number_of_edges()} edges")

    plot_subgraph(Gsub, pr)
    export_gexf(Gsub)

    print("\n-- Done: 08_visualization.py --------------------")


if __name__ == "__main__":
    main()
