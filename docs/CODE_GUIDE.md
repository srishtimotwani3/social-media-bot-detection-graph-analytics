# Code Guide — Every Feature with Exact Line Numbers

---

## `src/01_ingest_clean.py` (227 lines)

### Spark session setup
```python
# Lines 48–56
def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("BotDetection-Ingest")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
```

### Dataset categories and label assignment
```python
# Lines 35–45
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
```

### Reading each CSV with Spark and attaching label
```python
# Lines 78–88
df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .option("multiLine", "true")
    .option("escape", '"')
    .option("encoding", "UTF-8")
    .csv(csv_path)
)
df = df.withColumn("label", F.lit(label))
```

### Casting numeric user columns (followers, friends, etc.)
```python
# Lines 98–101
for col in ["followers_count", "friends_count", "listed_count",
            "favourites_count", "statuses_count"]:
    if col in df.columns:
        df = df.withColumn(col, F.col(col).cast(LongType()))
```

### Casting user ID
```python
# Line 103
df = df.withColumn("id", F.col("id").cast(LongType()))
```

### Dropping rows with no user ID
```python
# Line 111
df = df.filter(F.col("id").isNotNull())
```

### Deduplication by user ID (keep first occurrence)
```python
# Lines 114–116
from pyspark.sql.window import Window
w = Window.partitionBy("id").orderBy(F.monotonically_increasing_id())
df = df.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")
```

### Dropping tweet rows with missing ID or user_id
```python
# Line 147
df = df.filter(F.col("id").isNotNull() & F.col("user_id").isNotNull())
```

### Union all category DataFrames (aligning mismatched columns)
```python
# Lines 190–201
def safe_union(frames):
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
```

### Saving to Parquet
```python
# Lines 211–214
users_df.write.mode("overwrite").parquet(out_users)
tweets_df.write.mode("overwrite").parquet(out_tweets)
```

---

## `src/02_build_graphs.py` (275 lines)

### Config: thresholds for edges
```python
# Lines 42–44
MIN_CO_OCCUR  = 2    # minimum shared hashtag/URL occurrences for a co-activity edge
TIME_BUCKET_H = 1    # 1-hour time window
MIN_RT_COUNT  = 1    # minimum retweet count to keep a retweet edge
```

### Finding retweet relationships
```python
# Lines 78–97
retweets = tweets_df.filter(
    F.col("retweeted_status_id").isNotNull()
).select(
    F.col("user_id").alias("src"),
    F.col("retweeted_status_id").alias("rt_status_id"),
)
tweet_author = tweets_df.select(
    F.col("id").alias("tweet_id"),
    F.col("user_id").alias("dst"),
)
edges_raw = (
    retweets
    .join(tweet_author, retweets.rt_status_id == tweet_author.tweet_id, "inner")
    .filter(F.col("src") != F.col("dst"))   # remove self-loops
    .select("src", "dst")
)
```

### Aggregating retweet edge weights
```python
# Lines 100–105
edges = (
    edges_raw
    .groupBy("src", "dst")
    .agg(F.count("*").alias("weight"))
    .filter(F.col("weight") >= MIN_RT_COUNT)
)
```

### Extracting hashtags from tweet text (regex)
```python
# Lines 153–168
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
    .select(F.col("user_id"), F.col("time_bucket"), F.explode("tags").alias("item"))
)
```

### Extracting URLs from tweet text (regex)
```python
# Lines 171–188
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
    .select(F.col("user_id"), F.col("time_bucket"), F.explode("url_list").alias("item"))
)
```

### Bucketing timestamps to 1-hour windows
```python
# Lines 141–150
df = tweets_df.withColumn(
    "ts_unix",
    F.coalesce(
        F.unix_timestamp("created_at", "EEE MMM dd HH:mm:ss Z yyyy"),
        F.unix_timestamp("created_at"),
    )
).withColumn(
    "time_bucket",
    (F.col("ts_unix") / BUCKET_SECS).cast("long") * BUCKET_SECS
).filter(F.col("time_bucket").isNotNull())
```

### Self-join to find co-active pairs (core co-activity logic)
```python
# Lines 204–218
a = items.alias("a")
b = items.alias("b")
pairs = (
    a.join(
        b,
        (F.col("a.item")        == F.col("b.item")) &
        (F.col("a.time_bucket") == F.col("b.time_bucket")) &
        (F.col("a.user_id")     <  F.col("b.user_id")),   # avoid duplicate pairs
    )
    .select(F.col("a.user_id").alias("src"), F.col("b.user_id").alias("dst"))
)
```

### Filtering weak co-activity edges
```python
# Lines 220–225
edges = (
    pairs
    .groupBy("src", "dst")
    .agg(F.count("*").alias("weight"))
    .filter(F.col("weight") >= MIN_CO_OCCUR)
)
```

---

## `src/03_graph_analytics.py` (184 lines)

### Loading GraphFrames JAR
```python
# Lines 43–46
os.environ.setdefault(
    "PYSPARK_SUBMIT_ARGS",
    "--packages graphframes:graphframes:0.8.3-spark3.5-s_2.12 pyspark-shell",
)
```

### Creating a GraphFrame
```python
# Line 84
g = GraphFrame(v, e)
```

### In-degree and Out-degree
```python
# Lines 87–93
in_deg  = g.inDegrees
out_deg = g.outDegrees
degrees = (
    in_deg.join(out_deg, on="id", how="outer")
    .fillna(0, subset=["inDegree", "outDegree"])
)
degrees.write.mode("overwrite").parquet(str(PROC_DIR / "rt_degrees.parquet"))
```

### PageRank
```python
# Lines 97–100
pr = g.pageRank(resetProbability=0.15, maxIter=10)
pr.vertices.select("id", "pagerank") \
    .write.mode("overwrite").parquet(str(PROC_DIR / "rt_pagerank.parquet"))
```

### Triangle Count (on symmetrised graph)
```python
# Lines 104–113
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
```

### Motif — Mutual Retweet: `(a)→(b); (b)→(a)`
```python
# Lines 117–123
motif_mutual = g.find("(a)-[e1]->(b); (b)-[e2]->(a)")
mutual = motif_mutual.select(
    F.col("a.id").alias("user_a"),
    F.col("b.id").alias("user_b"),
).distinct()
mutual.write.mode("overwrite").parquet(str(PROC_DIR / "rt_motif_mutual.parquet"))
```

### Motif — Shared Target Amplification: `(a)→(c); (b)→(c)`
```python
# Lines 126–134
motif_shared = g.find("(a)-[e1]->(c); (b)-[e2]->(c)") \
    .filter(F.col("a.id") != F.col("b.id"))
shared = motif_shared.select(
    F.col("a.id").alias("user_a"),
    F.col("b.id").alias("user_b"),
    F.col("c.id").alias("target"),
).distinct()
shared.write.mode("overwrite").parquet(str(PROC_DIR / "rt_motif_shared_target.parquet"))
```

### Dropping label column before LPA (prevents column collision)
```python
# Line 145
v = spark.read.parquet(str(PROC_DIR / "coactivity_vertices.parquet")).drop("label")
```

### Connected Components (requires checkpoint dir)
```python
# Lines 151–156
chk_dir = PROJECT_ROOT / "data" / "checkpoints"
chk_dir.mkdir(parents=True, exist_ok=True)
spark.sparkContext.setCheckpointDir(str(chk_dir))
cc = g.connectedComponents()
cc.select("id", "component") \
    .write.mode("overwrite").parquet(str(PROC_DIR / "ca_components.parquet"))
```

### Label Propagation (community detection)
```python
# Lines 160–163
lpa = g.labelPropagation(maxIter=5)
lpa.select("id", "label") \
    .withColumnRenamed("label", "community") \
    .write.mode("overwrite").parquet(str(PROC_DIR / "ca_communities.parquet"))
```

---

## `src/04_feature_engineering.py` (158 lines)

### Loading base user table (profile features)
```python
# Lines 60–62
users = spark.read.parquet(str(PROC_DIR / "users_clean.parquet")) \
    .select("id", "label", "followers_count", "friends_count", "statuses_count") \
    .fillna(0, subset=["followers_count", "friends_count", "statuses_count"])
```

### Aggregating mutual retweet count per user
```python
# Lines 74–83
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
```

### Aggregating shared target count per user
```python
# Lines 86–95
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
```

### Computing community size
```python
# Lines 106–107
comm_size = communities.groupBy("community_id") \
    .agg(F.count("*").alias("community_size"))
```

### Joining all features into one table
```python
# Lines 110–118
feat = users \
    .join(degrees,      on="id", how="left") \
    .join(pagerank,     on="id", how="left") \
    .join(triangles,    on="id", how="left") \
    .join(mutual_count, on="id", how="left") \
    .join(shared_count, on="id", how="left") \
    .join(components,   on="id", how="left") \
    .join(communities,  on="id", how="left") \
    .join(comm_size,    on="community_id", how="left")
```

### Computing community bot ratio
```python
# Lines 127–134
comm_bot = (
    feat.groupBy("community_id")
    .agg(
        F.mean(F.when(F.col("label") == "bot", 1).otherwise(0))
         .alias("community_bot_ratio")
    )
)
feat = feat.join(comm_bot, on="community_id", how="left").fillna(0, subset=["community_bot_ratio"])
```

### Creating binary label column
```python
# Lines 137–140
feat = feat.withColumn(
    "label_bin",
    F.when(F.col("label") == "bot", 1).otherwise(0)
)
```

---

## `src/05_classification.py` (167 lines)

### Feature column definitions
```python
# Lines 49–57
PROFILE_FEATURES = [
    "followers_count", "friends_count", "statuses_count",
]
GRAPH_FEATURES = [
    "inDegree", "outDegree", "pagerank", "triangle_count",
    "mutual_retweet_count", "shared_target_count",
    "community_size", "community_bot_ratio",
]
```

### ML Pipeline: Imputer → VectorAssembler → StandardScaler → RandomForest
```python
# Lines 71–86
def build_pipeline(feature_cols: list, n_trees: int = 100) -> Pipeline:
    imputer   = Imputer(inputCols=feature_cols, outputCols=[f"{c}_imp" for c in feature_cols])
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
```

### 80/20 train-test split
```python
# Line 122
train, test = features_df.randomSplit([0.8, 0.2], seed=42)
```

### Training Model A (with graph features)
```python
# Lines 131–135
all_features = PROFILE_FEATURES + GRAPH_FEATURES
pipe_a = build_pipeline(all_features)
model_a = pipe_a.fit(train)
preds_a = model_a.transform(test)
results.append(evaluate(preds_a, label="WITH graph features"))
```

### Training Model B (baseline, no graph features)
```python
# Lines 138–141
pipe_b = build_pipeline(PROFILE_FEATURES)
model_b = pipe_b.fit(train)
preds_b = model_b.transform(test)
results.append(evaluate(preds_b, label="WITHOUT graph features (baseline)"))
```

### Evaluating metrics (Precision, Recall, F1, AUC)
```python
# Lines 91–109
mc_eval  = MulticlassClassificationEvaluator(labelCol="label_bin", predictionCol="prediction")
bin_eval = BinaryClassificationEvaluator(labelCol="label_bin", rawPredictionCol="rawPrediction")
precision = mc_eval.setMetricName("weightedPrecision").evaluate(predictions)
recall    = mc_eval.setMetricName("weightedRecall").evaluate(predictions)
f1        = mc_eval.setMetricName("f1").evaluate(predictions)
auc       = bin_eval.setMetricName("areaUnderROC").evaluate(predictions)
```

### Extracting feature importances from Random Forest
```python
# Lines 144–149
rf_model    = model_a.stages[-1]
importances = rf_model.featureImportances.toArray()
imp_df = pd.DataFrame({
    "feature":    all_features,
    "importance": importances[:len(all_features)],
}).sort_values("importance", ascending=False)
imp_df.to_csv(str(RESULTS_DIR / "feature_importances.csv"), index=False)
```

---

## `src/06_evaluation.py` (225 lines)

### Per-class metrics using sklearn
```python
# Lines 93–105
from sklearn.metrics import classification_report
report = classification_report(
    preds_pd["label_bin"], preds_pd["prediction"].astype(int),
    target_names=["genuine (0)", "bot (1)"],
    output_dict=True,
)
```

### Confusion matrix plot
```python
# Lines 108–121
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(preds_pd["label_bin"], preds_pd["prediction"].astype(int))
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["genuine", "bot"],
            yticklabels=["genuine", "bot"], ax=ax)
fig.savefig(str(RESULTS_PLOTS / "confusion_matrix.png"), dpi=150)
```

### Community purity calculation
```python
# Lines 129–141
purity = (
    feat.filter(F.col("community_id").isNotNull())
    .groupBy("community_id")
    .agg(
        F.count("*").alias("community_size"),
        F.sum(F.col("label_bin")).alias("bot_count"),
        F.mean(F.col("label_bin")).alias("bot_ratio"),
    )
    .withColumn("is_pure_bot_community", F.col("bot_ratio") >= PURITY_THRESHOLD)
)
purity_pd.to_csv(str(RESULTS_METRICS / "community_purity.csv"), index=False)
```

### Community purity histogram plot
```python
# Lines 147–156
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(purity_pd["bot_ratio"], bins=20, color="steelblue", edgecolor="white")
ax.axvline(PURITY_THRESHOLD, color="red", linestyle="--",
           label=f"Purity threshold ({PURITY_THRESHOLD})")
fig.savefig(str(RESULTS_PLOTS / "community_purity_hist.png"), dpi=150)
```

### Flag analysis (per signal precision)
```python
# Lines 167–192
pr_thresh  = feat.approxQuantile("pagerank", [PAGERANK_CUTOFF], 0.01)[0]
ind_thresh = feat.approxQuantile("inDegree", [DEGREE_CUTOFF],  0.01)[0]
flagged = feat \
    .withColumn("flag_high_pagerank",   F.col("pagerank")             >= pr_thresh) \
    .withColumn("flag_high_indegree",   F.col("inDegree")             >= ind_thresh) \
    .withColumn("flag_mutual_rt",       F.col("mutual_retweet_count") >  0) \
    .withColumn("flag_shared_target",   F.col("shared_target_count")  >  0) \
    .withColumn("flag_dense_community", F.col("community_bot_ratio")  >= PURITY_THRESHOLD)
# For each flag, count total flagged and how many are actual bots
for fc in flag_cols:
    sub = flagged.filter(F.col(fc) == True)
    total      = sub.count()
    actual_bot = sub.filter(F.col("label_bin") == 1).count()
    rows.append({"flag": fc, "flagged_accounts": total,
                 "actual_bots": actual_bot,
                 "precision": actual_bot / total if total else 0.0})
flag_df.to_csv(str(RESULTS_METRICS / "flag_summary.csv"), index=False)
```

---

## `src/07_scalability.py` (214 lines)

### Data fraction sampling
```python
# Lines 74–76
tweets = tweets_full.sample(fraction=frac, seed=seed)
users  = users_full.sample(fraction=frac, seed=seed)
```

### Fractions to test
```python
# Line 54
FRACTIONS = [0.10, 0.50, 1.00]
```

### Rebuilding retweet graph per fraction
```python
# Lines 85–93
retweets = tweets.filter(F.col("retweeted_status_id").isNotNull()) \
    .select(F.col("user_id").alias("src"), F.col("retweeted_status_id").alias("rt_id"))
tweet_author = tweets.select(F.col("id").alias("tweet_id"), F.col("user_id").alias("dst"))
rt_edges = (
    retweets.join(tweet_author, retweets.rt_id == tweet_author.tweet_id, "inner")
    .filter(F.col("src") != F.col("dst"))
    .groupBy("src", "dst").agg(F.count("*").alias("weight"))
    .filter(F.col("weight") >= 1)
)
```

### Running PageRank per fraction
```python
# Lines 106–110
chk_dir = PROJECT_ROOT / "data" / "checkpoints"
chk_dir.mkdir(parents=True, exist_ok=True)
spark.sparkContext.setCheckpointDir(str(chk_dir))
pr = g_rt.pageRank(resetProbability=0.15, maxIter=5)
pr.vertices.count()   # force materialisation
```

### Running LPA per fraction
```python
# Lines 146–148
g_ca = GraphFrame(ca_verts.drop("label"), ca_edges)
lpa  = g_ca.labelPropagation(maxIter=5)
lpa.count()   # materialise
```

### Wall-clock timing
```python
# Lines 82 and 150
t0 = time.time()
# ... all computation ...
elapsed = time.time() - t0
```

### Scalability plot (runtime + graph size vs fraction)
```python
# Lines 164–180
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(df["fraction"] * 100, df["elapsed_sec"], "o-", color="steelblue", linewidth=2)
axes[0].set_xlabel("Data fraction (%)")
axes[0].set_ylabel("Wall-clock time (seconds)")
axes[1].plot(df["fraction"] * 100, df["rt_edges"],  "s-", color="coral",  label="Retweet edges")
axes[1].plot(df["fraction"] * 100, df["ca_edges"],  "^-", color="green",  label="Co-activity edges")
fig.savefig(str(RESULTS_PLOTS / "scalability_plot.png"), dpi=150)
```

---

## `src/08_visualization.py` (271 lines)

### Building the NetworkX graph from Parquet data
```python
# Lines 89–97
def build_nx_graph(rt_v: pd.DataFrame, rt_e: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, row in rt_v.iterrows():
        G.add_node(row["id"], label=row.get("label", "unknown"),
                   screen_name=row.get("screen_name", ""))
    for _, row in rt_e.iterrows():
        G.add_edge(row["src"], row["dst"], weight=int(row["weight"]))
    return G
```

### Extracting representative subgraph (bot clusters + genuine users)
```python
# Lines 104–128
def extract_subgraph(G: nx.DiGraph, max_nodes: int = 120) -> nx.DiGraph:
    wcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    # Collect top coordinated bot nodes by degree
    bot_nodes = set()
    for c in wcc:
        if any(G.nodes[n].get("label") == "bot" for n in c):
            bot_nodes.update(c)
    bot_sub  = G.subgraph(bot_nodes)
    top_bots = [n for n, d in sorted(bot_sub.degree(), key=lambda x: x[1], reverse=True)[:60]]
    # Collect genuine users from pure genuine components
    gen_nodes = [n for c in wcc if all(G.nodes[n].get("label") == "genuine" for n in c) for n in c][:35]
    combined = set(top_bots + gen_nodes)
    for n in list(combined):
        if len(combined) >= max_nodes:
            break
        combined.update(G.predecessors(n))
        combined.update(G.successors(n))
    return G.subgraph(list(combined)[:max_nodes]).copy()
```

### Node colouring by label (Red=bot, Green=genuine, Grey=unknown)
```python
# Lines 138–143
color_map = {
    "bot":     "#e74c3c",
    "genuine": "#2ecc71",
    "unknown": "#95a5a6",
}
node_colors = [color_map.get(Gsub.nodes[n].get("label", "unknown"), "#95a5a6")
               for n in Gsub.nodes()]
```

### Node sizing by PageRank
```python
# Lines 136–148
pr_map = dict(zip(pr["id"], pr["pagerank"]))
node_sizes = [
    max(100, min(2000, pr_map.get(n, 0.01) * 3000))
    for n in Gsub.nodes()
]
```

### Spring layout and drawing
```python
# Lines 151–161
pos = nx.spring_layout(Gsub, seed=42, k=0.3)
nx.draw_networkx_nodes(Gsub, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85, ax=ax)
nx.draw_networkx_edges(Gsub, pos, alpha=0.3, arrows=True, arrowsize=10, edge_color="#555555", ax=ax)
top_pr = sorted(Gsub.nodes(), key=lambda n: pr_map.get(n, 0), reverse=True)[:20]
labels = {n: Gsub.nodes[n].get("screen_name", str(n)) for n in top_pr}
nx.draw_networkx_labels(Gsub, pos, labels=labels, font_size=6, ax=ax)
```

### Exporting to GEXF (Gephi format)
```python
# Lines 181–185
def export_gexf(Gsub: nx.DiGraph):
    Gmapped = nx.relabel_nodes(Gsub, {n: str(n) for n in Gsub.nodes()})
    out = str(RESULTS_PLOTS / "suspicious_subgraph.gexf")
    nx.write_gexf(Gmapped, out)
```

### PageRank and In-degree distribution plots
```python
# Lines 191–207
def dist_plot(feat: pd.DataFrame, col: str, title: str, out_name: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    for lbl, color in [("genuine", "#2ecc71"), ("bot", "#e74c3c")]:
        sub = feat[feat["label"] == lbl][col].dropna()
        sub = sub[sub > 0]
        if not sub.empty:
            ax.hist(sub, bins=50, alpha=0.6, color=color, label=lbl, density=True)
    ax.set_xscale("log")
    fig.savefig(str(RESULTS_PLOTS / out_name), dpi=150)
```

### Feature importance bar chart
```python
# Lines 210–224
def feature_importance_plot():
    df = pd.read_csv(imp_path).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(df["feature"], df["importance"], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importances")
    fig.savefig(str(RESULTS_PLOTS / "feature_importance.png"), dpi=150)
```

---

## `app.py` — Streamlit UI

### Tab 1: Account verdict and explanation
```python
# app.py — Account inspector section
selected_display = st.selectbox("Select an account to test:", sample_accounts["display"].tolist())
row = sample_accounts[sample_accounts["display"] == selected_display].iloc[0]

is_bot = (row["label"] == "bot")
if is_bot:
    st.error("### 🔴 Verdict: DETECTED BOT")
else:
    st.success("### 🟢 Verdict: GENUINE USER")
```

### Bot signal explanations in plain English
```python
# app.py — Reason generation logic
bot_ratio = row.get("community_bot_ratio", 0)
if bot_ratio >= 0.8:
    reasons.append(f"🚨 Dense Bot Ring: Belongs to an interaction group where {bot_ratio*100:.0f}% of members are known bots.")

shared_targets = row.get("shared_target_count", 0)
if shared_targets > 0:
    reasons.append(f"🚨 Coordinated Amplification: Found participating in {int(shared_targets)} shared-target boost patterns.")

pagerank_val = row.get("pagerank", 0)
if pagerank_val > 0.5:
    reasons.append(f"⚠️ High Centrality (PageRank {pagerank_val:.2f}): Functions as a central amplifier hub.")

mutual_rts = row.get("mutual_retweet_count", 0)
if mutual_rts > 0:
    reasons.append(f"⚠️ Reciprocal Retweeting: Engaged in {int(mutual_rts)} mutual back-and-forth retweet loops.")
```

### Tab 2: Model comparison metrics
```python
# app.py — Side-by-side model cards
st.metric(label="Overall Accuracy (F1-Score)", value="94.67%")                        # Baseline
st.metric(label="Overall Accuracy (F1-Score)", value="95.10%", delta="+0.43% improvement")  # With Graph
```

### Tab 2: Loading and displaying feature importance chart
```python
# app.py
feature_plot_path = PLOTS_DIR / "feature_importance.png"
if feature_plot_path.exists():
    st.image(str(feature_plot_path), caption="Feature Importances in the Random Forest Model", use_container_width=True)
```

### Tab 3: Network subgraph image
```python
# app.py
subgraph_img = PLOTS_DIR / "suspicious_subgraph.png"
if subgraph_img.exists():
    st.image(str(subgraph_img),
             caption="Network visualization: Red=bots, Green=genuine, Node size=PageRank centrality",
             use_container_width=True)
```

### Tab 4: Scalability table and chart
```python
# app.py
display_scale["Spark Wall-Clock Runtime"] = display_scale["elapsed_sec"].apply(lambda s: f"{s:.1f} seconds")
st.table(display_scale[["Data Fraction", "Tweets Processed", "User Accounts",
                         "Retweet Links", "Co-Activity Links", "Spark Wall-Clock Runtime"]])
scale_plot = PLOTS_DIR / "scalability_plot.png"
st.image(str(scale_plot), caption="Runtime & Graph Edge Scaling vs. Data Size", use_container_width=True)
```
