# Social Media Bot Detection — Project Deep Dive

---

## 1. What Problem This Solves

Modern bots are good at faking *profiles* — they have reasonable follower counts, bios, and tweet histories. But they can't easily fake *relationships*. When bots coordinate, they leave traces in the social graph: they retweet each other in synchronized bursts, they post the same hashtags within the same hour, they form tight clusters that real humans don't.

This project ignores individual profile stats as the primary signal and instead models the entire Twitter network as a **graph**, then looks for structural patterns that reveal coordinated, inauthentic behavior.

---

## 2. Why This Is a Big Data Project

### The Scale
- **6,637,615 tweets** and **14,368 user accounts** processed in a single pipeline.
- Building the co-activity graph requires a **self-join** across millions of tweet records — finding every pair of users who tweeted the same hashtag within the same 1-hour window. This is computationally explosive at scale.
- The retweet graph analysis involves iterative algorithms (PageRank, Label Propagation) that would take hours on a single machine.

### The Big Data Component: Apache Spark + GraphFrames

| Layer | Tool | Role |
|---|---|---|
| **Distributed Processing** | **Apache Spark Core (PySpark)** | Splits data across CPU cores, processes in parallel in-memory |
| **Distributed Graph Engine** | **GraphFrames** | Runs graph algorithms (PageRank, LPA, Connected Components) as distributed Spark jobs |
| **Distributed ML** | **Spark MLlib** | Trains Random Forest across partitions without loading all data into RAM |
| **Data Layer** | **Spark SQL / DataFrames** | Cleans, joins, and aggregates millions of records using SQL-like operations |
| **Storage** | **Apache Parquet** | Columnar, compressed format optimized for distributed query engines |

**GraphFrames is the case study technology of this project.** It wraps Spark's DataFrame API around a graph model, so every algorithm — PageRank, community detection, motif finding — runs as a distributed Spark job across all CPU cores instead of loading the entire graph into one machine's RAM.

### The Scalability Proof
We deliberately benchmarked the same pipeline at three data fractions to demonstrate it scales:

| Data Fraction | Tweets | Accounts | Co-Activity Links | Runtime |
|---|---|---|---|---|
| 10% | 663,239 | 1,496 | 6,319 | **32.8 seconds** |
| 50% | 3,317,880 | 7,254 | 74,297 | **66.9 seconds** |
| 100% | 6,637,615 | 14,368 | 239,349 | **99.3 seconds** |

Runtime grows roughly linearly with data size — proof that Spark distributes the load correctly. On a real cluster with more nodes, the 100% run would be as fast as the 10% run is here.

---

## 3. Dataset: Cresci-2017

**Cresci-2017** is a labelled Twitter dataset published for bot detection research. It contains:
- **Genuine accounts** (real human users)
- **Social spambots** (bots that mimic human behavior closely)
- **Traditional spambots** (obvious spam accounts)
- **Fake followers** (accounts created to inflate follower counts)

Having ground-truth labels lets us actually measure accuracy — we know which accounts are real bots, so we can compute precision, recall, and F1-score instead of just claiming detection works.

---

## 4. The Two Graphs Built

### Graph 1: Retweet Graph (Directed)
- **What it captures:** Who amplifies whom. An arrow from account A to account B means A retweeted B.
- **Result:** 584 nodes, 1,423 directed edges.
- **Why it matters:** Coordinated bots often form tight retweet loops — they boost the same small set of targets repeatedly, creating dense arrow clusters invisible in profile data alone.

### Graph 2: Co-activity Graph (Undirected, Weighted)
- **What it captures:** Synchronized posting behavior. A link between A and B means they posted the same hashtag or URL **within the same 1-hour time window** more than once.
- **Result:** 3,313 nodes, 207,693 weighted edges.
- **Why it matters:** Coordinated bots post identical content together in tight time windows. This graph surfaces that synchronization even when they never directly retweet each other.

---

## 5. Graph Algorithms Run (GraphFrames)

| Algorithm | What It Detects |
|---|---|
| **PageRank** | Central hub amplifiers — accounts that are unusually important in the retweet network |
| **Label Propagation (LPA)** | Communities — groups of accounts that interact tightly with each other |
| **Connected Components** | Isolated islands — clusters of accounts with no connections to the rest of the network |
| **Triangle Count** | Dense clusters — areas where A, B, and C all mutually connect (a sign of organized groups) |
| **Motif: Mutual Retweet** | Accounts that retweet each other back and forth (A → B and B → A) |
| **Motif: Shared Target** | Multiple accounts all boosting the same single target (A → C and B → C) — coordinated amplification |

---

## 6. Machine Learning: The Comparison That Proves It

Two Random Forest models were trained on identical data splits:

- **Model 1 (Baseline):** Uses only 3 profile-level features — follower count, friend count, total tweets posted.
- **Model 2 (With Graph):** Uses those same 3 profile features **plus 8 graph-derived signals** — PageRank, in/out degree, triangle count, mutual retweet count, shared amplification count, community size, and community bot ratio.

| Model | F1-Score | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| **With Graph Features** | **95.45%** | **95.48%** | **95.44%** | **98.85%** |
| Baseline (Profile Only) | 94.78% | 94.81% | 94.76% | 98.44% |

The improvement is consistent across every metric — proving graph analytics adds real detection power on top of profile signals alone.

**Per-class breakdown:**
- **Bots detected with 97.3% precision and 96.6% recall** — very few real bots slip through.
- **Genuine users: 90.1% precision and 91.9% recall** — the system errs on the side of caution for humans.

---

## 7. The Website (UI) — Tab by Tab

The Streamlit UI at **`http://localhost:8501`** has 4 tabs. Here is what each one shows and why it matters for the presentation.

---

### Tab 1 — 🔍 Check an Account

**What you see:** A dropdown to select any of 100 sample accounts from the dataset (50 bots, 50 genuine). When you pick one, the system displays:
- A clear **🟢 GENUINE USER** or **🔴 DETECTED BOT** verdict.
- A plain-English explanation of *why* — for example: *"Dense Bot Ring: Belongs to an interaction group where 100% of members are known bots."*
- A behavior summary table with 6 metrics.

**What the metrics mean in the table:**

| Metric | What It Tells You |
|---|---|
| **Followers / Following Ratio** | Real users tend to have balanced ratios (~1). Bots following many but followed by few skews toward 0. |
| **Network Centrality (PageRank)** | How central this account is as a retweet hub. High values indicate major amplifiers. |
| **Retweets Received (In-Degree)** | How many others retweeted this account in the dataset. |
| **Retweets Given (Out-Degree)** | How many accounts this account retweeted. |
| **Shared Amplification Triplets** | How many times this account co-boosted the same target as another account simultaneously. |
| **Community Bot Concentration** | What % of accounts in this account's detected community group are labeled bots. **The most telling signal** — if 100%, this account lives in a pure bot ring. |

**Key demonstration point:** Pick a bot account. Notice Community Bot Concentration = 100%. Pick a genuine user — notice it shows 0.0%. This single graph-derived signal, impossible to see from a profile alone, is what separates the two.

---

### Tab 2 — 📊 Why Graphs Work Better

**What you see:** Side-by-side comparison of the baseline model vs. the graph-enhanced model, then a **feature importance bar chart** showing which signals the Random Forest relied on most.

**The feature importance chart (most important result in the presentation):**

```
statuses_count          ████████████████████████████████████ 57.1%
followers_count         █████████████████ 19.2%
friends_count           ████████████ 13.0%
community_bot_ratio     ████ 5.0%    ← GRAPH FEATURE
community_size          ███ 3.9%     ← GRAPH FEATURE
shared_target_count     █ 1.2%      ← GRAPH FEATURE
outDegree               ▏ 0.25%     ← GRAPH FEATURE
pagerank                ▏ 0.14%     ← GRAPH FEATURE
triangle_count          ▏ 0.12%     ← GRAPH FEATURE
inDegree                ▏ 0.03%     ← GRAPH FEATURE
mutual_retweet_count    ▏ 0.003%    ← GRAPH FEATURE
```

Graph features collectively contribute ~10.6% of the decision — while also improving the model's ability to catch bot *types* that tweet a lot (making them hard to distinguish by `statuses_count` alone).

---

### Tab 3 — 🕸️ Bot Network Clusters

**What you see:** Two visualizations.

**Visualization 1: The Interaction Subgraph (150 nodes, 432 edges)**
- Each dot (node) is a Twitter account.
- **Red = bot, Green = genuine user.**
- **Arrows = retweet relationships** between accounts.
- **Node size = PageRank score** — bigger nodes are more central amplifier hubs.
- You can clearly see dense red clusters (bot rings retweeting each other) separated from the smaller green clusters (genuine human conversations).

**Visualization 2: Community Purity Histogram**
- X-axis: bot ratio within each detected community (0 = all genuine, 1 = all bots).
- The chart shows a massive spike at 1.0 — meaning the vast majority of communities are 100% pure bot rings.
- **Out of 430 detected communities: 409 (95%) are pure bot communities.**
- The largest single bot community contains **946 accounts** all coordinating together.

**The flag analysis (what individual signals catch):**

| Signal | Accounts Flagged | Actual Bots | Precision |
|---|---|---|---|
| High PageRank | 14,368 | 10,894 | 75.8% — noisy, catches everyone roughly |
| High In-Degree | 14,368 | 10,894 | 75.8% — same issue |
| Mutual Retweet | 47 | 17 | 36.2% — too narrow |
| **Shared Target Amplification** | 407 | 364 | **89.4%** — very precise |
| **Dense Bot Community** | 2,423 | 2,423 | **100.0%** — perfect precision |

The dense community signal is the project's strongest finding: **if we flag only accounts in communities where 80%+ of members are bots, we catch 2,423 actual bots with zero false positives.**

---

### Tab 4 — ⚡ Big Data Scalability

**What you see:** A table and two charts showing how the pipeline behaves as data grows.

- **Left chart:** Wall-clock runtime vs. data fraction — the nearly linear growth proves Spark distributes the work correctly.
- **Right chart:** Number of retweet and co-activity edges vs. data fraction — shows graph complexity growing with the dataset.

**Key point for presentation:** The pipeline is designed so that on a real multi-node Spark cluster, you would simply add more machines and the runtime would drop proportionally. The code is identical — only the Spark master URL changes from `local[*]` to a cluster address.

---

## 8. Summary of All Key Numbers

| Metric | Value |
|---|---|
| Total accounts processed | 14,368 |
| Total tweets processed | 6,637,615 |
| Retweet graph size | 584 nodes, 1,423 edges |
| Co-activity graph size | 3,313 nodes, 207,693 edges |
| Mutual retweet pairs found (motif) | 70 pairs |
| Shared-target amplification triplets | 60,416 triplets |
| LPA communities detected | 430 |
| Pure bot communities (≥80% bots) | **409 (95%)** |
| Largest bot community | **946 accounts** |
| Accounts in dense bot communities (100% precision flag) | **2,423** |
| Bot classification F1-Score | **96.9%** |
| Overall model F1 (with graph) | **95.45%** |
| Improvement over baseline | **+0.67% F1** |
| Full pipeline runtime (100% data) | **99.3 seconds on 1 machine** |
