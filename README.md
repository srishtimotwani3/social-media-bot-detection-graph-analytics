# Social Media Bot and Coordinated Inauthentic Behavior Detection
### Using Graph Analytics (GraphFrames + PySpark)

---

## Overview
This project detects bots and coordinated inauthentic behavior on social media by modeling
accounts as a **graph** and applying distributed graph analytics via **Apache Spark + GraphFrames**.

- Dataset: **Cresci-2017** (labelled genuine users, social spambots, fake followers, traditional spambots)
- Graph Engine: **GraphFrames** (DataFrame-native, Python-friendly, runs on Spark)
- ML: **Spark MLlib Random Forest**
- Visualization: **NetworkX / Gephi**

---

## Setup

### 1. Prerequisites
- Python 3.10+
- Java 11 or 17 (required by Spark)  
  Download from: https://adoptium.net/

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
Run the downloader helper:
```bash
python src/00_download_data.py
```

This downloads and extracts Cresci-2017 into `data/raw/`.

**Manual alternative:** Download from https://botometer.osome.iu.edu/bot-repository/datasets.html
and place the extracted folders under `data/raw/cresci-2017/`.

Expected layout:
```
data/raw/cresci-2017/
├── genuine_accounts.csv/
│   ├── users.csv
│   └── tweets.csv
├── social_spambots_1.csv/
│   ├── users.csv
│   └── tweets.csv
├── social_spambots_2.csv/
│   ├── users.csv
│   └── tweets.csv
├── social_spambots_3.csv/
│   ├── users.csv
│   └── tweets.csv
├── traditional_spambots_1.csv/
│   ├── users.csv
│   └── tweets.csv
└── fake_followers.csv/
    ├── users.csv
    └── tweets.csv
```

---

## Pipeline (run in order)

| Script | What it does |
|--------|-------------|
| `src/00_download_data.py` | Downloads Cresci-2017 |
| `src/01_ingest_clean.py` | Loads CSVs into Spark, cleans, saves Parquet |
| `src/02_build_graphs.py` | Builds retweet graph + co-activity graph via GraphFrames |
| `src/03_graph_analytics.py` | PageRank, LPA, Connected Components, TriangleCount, Motifs |
| `src/04_feature_engineering.py` | Joins all signals into a per-account feature table |
| `src/05_classification.py` | Trains Random Forest (with & without graph features), reports metrics |
| `src/06_evaluation.py` | Community purity, detailed F1/Precision/Recall tables |
| `src/07_scalability.py` | Runtime experiment at 10%, 50%, 100% data fractions |
| `src/08_visualization.py` | Exports suspicious subgraphs as PNG + GEXF (Gephi) |

Run all steps end-to-end:
```bash
python src/01_ingest_clean.py
python src/02_build_graphs.py
python src/03_graph_analytics.py
python src/04_feature_engineering.py
python src/05_classification.py
python src/06_evaluation.py
python src/07_scalability.py
python src/08_visualization.py
```

---

## Technology Choices

| Layer | Choice | Reason |
|-------|--------|--------|
| Processing | Apache Spark (PySpark) | In-memory distributed; fast iterative graph algorithms |
| Graph | GraphFrames | DataFrame-native, Python API, distributed, motif finding |
| ML | Spark MLlib | Direct DataFrame input from GraphFrames, no conversion |
| Viz | NetworkX / Gephi | Rich algorithm support for small subgraph inspection |
| Streaming | **Not used** | Dataset is static; scoped as future work |

---

## Key Results

### 1. Classification (With vs. Without Graph Features)
| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|
| **WITH graph features** | **0.9516** | **0.9506** | **0.9510** | **0.9864** |
| WITHOUT graph features (baseline) | 0.9472 | 0.9463 | 0.9467 | 0.9846 |

### 2. Scalability Across Dataset Fractions
| Fraction | Tweets | Accounts | Retweet Edges | Co-activity Edges | Runtime (s) |
|---|---|---|---|---|---|
| **10%** | 663,239 | 1,496 | 11 | 6,319 | 32.81 s |
| **50%** | 3,317,880 | 7,254 | 408 | 74,297 | 66.88 s |
| **100%** | 6,637,615 | 14,368 | 1,423 | 239,349 | 99.28 s |

### 3. Structural Signals & Pure Bot Communities
- **Community Purity:** Of 430 detected communities on the co-activity graph, **409 are pure-bot communities** (≥80% bot concentration).
- **Dense Community Flag:** Accounts belonging to dense bot communities achieved **100.0% precision** (2,423 accounts flagged, 2,423 actual bots).
- **Shared Target Amplification:** Flagged 407 accounts with 364 actual bots (**89.4% precision**).

---

## Visualizations
All generated plots are saved in `results/plots/`:
- `confusion_matrix.png` — Confusion matrix for the Random Forest classifier
- `scalability_plot.png` — Runtime and graph edge scaling vs. data size
- `community_purity_hist.png` — Distribution of bot ratios across detected LPA communities
- `feature_importance.png` — Top features contributing to bot detection
- `suspicious_subgraph.png` & `suspicious_subgraph.gexf` — Suspicious subgraph network export (Gephi compatible)

