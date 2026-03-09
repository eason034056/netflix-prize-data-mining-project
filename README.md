# Netflix Prize Data Mining Project

A collaborative data mining and machine learning project based on the Netflix Prize dataset, exploring large-scale user rating data to understand user behavior, build recommendation systems, and perform user segmentation.

The project is organized into five analytical phases, each completed by a team member and integrated into a full end-to-end data mining pipeline.

The analysis includes:

- Data ingestion and preprocessing
- Exploratory data analysis (EDA)
- Recommendation system modeling
- Unsupervised clustering
- RFM-based user segmentation and business insights

---

## Table of Contents

- [Project Overview](#project-overview)
- [Data Source and Layout](#data-source-and-layout)
- [Repository Structure](#repository-structure)
- [Prerequisites and Installation](#prerequisites-and-installation)
- [How to Run the Project](#how-to-run)
- [Phase-by-Phase Summary](#phase-by-phase-summary-all-5-people)
- [Outputs and Artifacts](#outputs-and-artifacts)
- [Team Members](#team-members)
- [Presentation Structure](#presentation-and-deliverables)

---

## Project Overview

This project analyzes the Netflix Prize dataset, a large-scale movie rating dataset containing millions of user ratings across thousands of movies.

The project investigates several key data mining questions:
- What behavioral patterns exist in user rating activity?
- How can we build effective recommendation systems for users?
- Can we identify meaningful clusters of users and movies?
- How can we apply RFM segmentation to understand customer engagement?

The workflow is divided into five phases:

| Phase | Owner  | Focus |
|-------|--------|--------|
| **1** | Person A | Data loading and preprocessing → shared `ratings.parquet` and `probe.parquet` |
| **2** | Person B | Exploratory data analysis (EDA) on ratings, users, and movies |
| **3** | Person C | Recommendation engine: baselines, matrix factorization (SVD, NMF), hybrid model, RMSE evaluation |
| **4** | Person D | Clustering: user and movie segments (K-Means, hierarchical, DBSCAN, t-SNE) |
| **5** | Person E | RFM analysis: user segmentation, visualizations, business insights, cross-reference with clustering |

Each phase produces intermediate datasets and analytical outputs that feed into later stages of the pipeline.

---

## Data Source and Layout

The project uses the Netflix Prize dataset, originally released by Netflix as part of the Netflix Prize competition.
- **Netflix Prize**: [Kaggle – Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) or the official Netflix Prize release.
- **Expected layout under project root:**
  - `dataset/training_set/` — 17,770 files `mv_*.txt`, one per movie. Each file: first line `MovieID:`, then lines `CustomerID,Rating,Date`.
  - `dataset/movie_titles.txt` — `MovieID,YearOfRelease,Title` (Title may contain commas; parse with `split(',', 2)`).
  - `dataset/probe.txt` — Holdout set for recommendation evaluation: sections `MovieID:` followed by CustomerID lines (no ratings in file).

After **Phase 1**, the project uses:

- `data/ratings.parquet` — Main ratings table: **MovieID**, **CustomerID**, **Rating**, **Date**, **YearOfRelease**, **Title**, **Year**, **Month**, **DayOfWeek**.
- `data/probe.parquet` — Probe (MovieID, CustomerID) for evaluation; Person C adds ratings from the training set for RMSE computation.

---

## Repository Structure

```
netflix-prize-data-mining-project/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── 05_rfm_analysis.ipynb     # Phase 5 (Person E) — at project root
├── notebooks/
│   ├── 01_data_loading.ipynb # Phase 1 (Person A)
│   ├── 02_eda.ipynb          # Phase 2 (Person B)
│   ├── 03_recommendation.ipynb # Phase 3 (Person C)
│   └── 04_clustering.ipynb   # Phase 4 (Person D)
├── data/                     # Created by 01; used by 02–05
│   ├── ratings.parquet
│   ├── probe.parquet
│   ├── train.parquet         # Created by 03 (train set for recommendation)
│   └── probe_with_ratings.parquet  # Created by 03
├── outputs/
│   ├── 04_clustering/        # Person D outputs
│   │   ├── user_clusters.parquet
│   │   ├── movie_clusters.parquet
│   │   ├── algorithm_comparison.csv
│   │   └── *.png (elbow, PCA, dendrogram, t-SNE, DBSCAN, etc.)
│   └── 05_rfm_analysis/      # Person E outputs
│       ├── rfm_segments.parquet
│       ├── rfm_3d_scatter.html
│       ├── rfm_heatmap_RF.png
│       ├── rfm_boxplot_by_segment.png
│       ├── rfm_segment_sizes.png
│       ├── rfm_segment_treemap.html
│       ├── rfm_score_distribution.png
│       └── rfm_vs_clustering_heatmap.png
└── dataset/                  # You provide this (see Data Source)
    ├── training_set/
    ├── movie_titles.txt
    └── probe.txt
```

Notebooks resolve paths via a **PROJECT_ROOT** convention: if run from `notebooks/`, the parent directory is used as project root so that `data/` and `dataset/` are found correctly.

---

## Prerequisites and Installation

- **Python**: 3.8+ recommended.
- **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
- **Main packages:** pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, scikit-surprise, scipy, pyarrow, tqdm.

---

## How to Run

1. **Prepare data**  
   Place the Netflix dataset under `dataset/` as described in [Data Source and Layout](#data-source-and-layout).

2. **Run notebooks in order** (later phases depend on earlier outputs):
   - **01_data_loading.ipynb** — Produces `data/ratings.parquet` and `data/probe.parquet`. Optional: set `SAMPLE_FRACTION = 0.1` in the notebook for a quick test run.
   - **02_eda.ipynb** — Reads `data/ratings.parquet`; no new shared data files.
   - **03_recommendation.ipynb** — Builds train/probe split, fits models, evaluates RMSE; writes `data/train.parquet`, `data/probe_with_ratings.parquet`, and saved models (e.g. SVD, NMF).
   - **04_clustering.ipynb** — Reads `data/ratings.parquet`; writes `outputs/04_clustering/user_clusters.parquet`, `movie_clusters.parquet`, and figures.
   - **05_rfm_analysis.ipynb** — Reads `data/ratings.parquet` and (optional) `outputs/04_clustering/user_clusters.parquet`; writes `outputs/05_rfm_analysis/` (parquet, HTML, PNG).

3. **Where to run**  
   Run from project root or from `notebooks/`; path logic in each notebook detects the environment and sets `PROJECT_ROOT` accordingly.

---

## Phase-by-Phase Summary 

### Phase 1 — Data Loading and Preprocessing 

- **Notebook:** `notebooks/01_data_loading.ipynb`
- **Goal:** Build a shared data pipeline for the whole team.
- **Steps:**
  1. Import libraries and set paths (`PROJECT_ROOT`, `DATA_DIR`, `OUTPUT_DIR`).
  2. Parse `movie_titles.txt` → MovieID, YearOfRelease, Title.
  3. Parse all `mv_*.txt` in `training_set/` → MovieID, CustomerID, Rating, Date.
  4. Merge ratings with movie info; convert Date to datetime; add Year, Month, DayOfWeek.
  5. Save **ratings.parquet** to `data/`.
  6. Parse **probe.txt** → (MovieID, CustomerID) and save **probe.parquet** to `data/`.
  7. Validation and summary (row counts, columns, sample).
- **Outputs:** `data/ratings.parquet`, `data/probe.parquet`.
- **Note:** `SAMPLE_FRACTION` (e.g. 0.1) can be used to load a subset of movies for faster testing.

---

### Phase 2 — Exploratory Data Analysis

- **Notebook:** `notebooks/02_eda.ipynb`
- **Goal:** Understand distributions and patterns in ratings, users, and movies.
- **Main sections:**
  - **Load and setup:** Read `data/ratings.parquet`, ensure Date is datetime, derive Year/Month/DayOfWeek if missing.
  - **User behavior (2.2):** Rating volume per user (log-scale), mean rating per user, power-law / long-tail, temporal patterns (weekday, month, year).
  - **Movie popularity (2.3):** Ratings per movie, top-rated vs most-rated (with minimum ratings threshold), long-tail, scatter of count vs average rating.
  - **Correlation and statistical tests (2.5):** Movie age vs average rating, number of ratings vs average rating, summary statistics.
- **Outputs:** In-notebook figures and tables (no new shared parquet files). Findings feed into the presentation’s “EDA Highlights.”

---

### Phase 3 — Recommendation Engine

- **Notebook:** `notebooks/03_recommendation.ipynb`
- **Goal:** Train and evaluate recommendation models on the probe set; report RMSE.
- **Data prep:** Split ratings into train and probe (probe pairs removed from train); add actual ratings to probe for evaluation. Save `train.parquet` and `probe_with_ratings.parquet`.
- **Models:**
  - **Baselines:** Global mean; user + movie bias (prediction = global_mean + user_bias + movie_bias).
  - **Matrix factorization (Surprise):** **SVD** (best single model), **NMF**; GridSearchCV for hyperparameters on a tuning subset.
  - **Hybrid:** SVD + item-based collaborative filtering on **residuals** (actual − SVD prediction) for top popular movies; final prediction = SVD + α × KNN residual, clipped to [1, 5].
- **Evaluation:** RMSE on the probe set.
- **Typical results (probe RMSE):** Global mean ~1.13, bias ~0.99, SVD ~0.96, SVD + Item-CF residual correction ~0.95; NMF higher (~1.49 on probe with default usage).
- **Outputs:** `data/train.parquet`, `data/probe_with_ratings.parquet`, saved SVD/NMF models (e.g. under `data/` or a chosen path), and an RMSE comparison table for the presentation.

---

### Phase 4 — Clustering 

- **Notebook:** `notebooks/04_clustering.ipynb`
- **Goal:** Segment users and movies using clustering; compare methods and visualize.
- **Sections:**
  - **4.1 Feature engineering:**  
    **Users:** rating_count, rating_mean, rating_std, most_common_rating, five_star_pct, date_range_days.  
    **Movies:** rating_count, rating_mean, rating_std, rating_skewness, YearOfRelease.  
    StandardScaler + PCA (e.g. 2 and 3 components) for visualization and clustering.
  - **4.2 User clustering:** K-Means on PCA features; elbow and silhouette for k; cluster labels (e.g. Casual Users, Power Users, Harsh Critics, Generous Raters, Selective Viewers, Active Explorers).
  - **4.3 Movie clustering:** K-Means + hierarchical (dendrogram); cluster labels (e.g. Blockbusters, Niche Favorites, Polarizing Films, Forgotten Films, Classic Hits, Cult Favourites).
  - **4.4 Advanced:** DBSCAN for outliers; t-SNE visualization; algorithm comparison (K-Means vs hierarchical vs DBSCAN, silhouette).
- **Outputs:** `outputs/04_clustering/user_clusters.parquet`, `movie_clusters.parquet`, `algorithm_comparison.csv`, and figures (elbow, PCA scatter, dendrogram, t-SNE, DBSCAN). Person E uses `user_clusters.parquet` for RFM–clustering cross-tab.

---

### Phase 5 — RFM Analysis 

- **Notebook:** `05_rfm_analysis.ipynb` (at project root)
- **Goal:** Segment users with RFM (Recency, Frequency, Monetary); align with course framework; produce business insights and cross-reference with Person D’s clustering.
- **Sections:**
  - **5.1 RFM definitions:** R = days since last rating (benchmark 2005-12-31), F = total ratings, M = average rating (engagement proxy). Quintile scores 1–5 (higher = better).
  - **5.2 Quintile scoring and segments:** Composite RFM score; rule-based segments: Best Customers (444), Loyal Customers (X4X), Big Spenders (XX4), At Risk (213), Lost Customers (122), Lost Cheap Customers (111), plus Potential Loyal, Need Attention, Hibernating, Others.
  - **5.3 Visualizations:** 3D scatter (R, F, M by segment), R_score×F_score heatmap, R/F/M box plots by segment, segment-size bar chart and treemap, RFM score distribution.
  - **5.4 Business insights:** Marketing strategy table (course-aligned); retention context (e.g. 5× cost to acquire vs retain); cross-tab and heatmap of RFM segment vs Person D cluster; interpretation and key takeaways.
- **Outputs:** `outputs/05_rfm_analysis/rfm_segments.parquet`, HTML/PNG figures listed in [Repository Structure](#repository-structure). Cross-tab with clustering only if `outputs/04_clustering/user_clusters.parquet` exists.

---

## Outputs and Artifacts

| Path | Description |
|------|-------------|
| `data/ratings.parquet` | Main ratings + movie info + temporal features |
| `data/probe.parquet` | Probe (MovieID, CustomerID) |
| `data/train.parquet` | Training ratings (probe removed) |
| `data/probe_with_ratings.parquet` | Probe with actual ratings for RMSE |
| `outputs/04_clustering/user_clusters.parquet` | User features + cluster labels |
| `outputs/04_clustering/movie_clusters.parquet` | Movie features + cluster labels |
| `outputs/04_clustering/algorithm_comparison.csv` | Clustering method comparison |
| `outputs/05_rfm_analysis/rfm_segments.parquet` | User-level RFM + segment |
| `outputs/05_rfm_analysis/*.png`, `*.html` | RFM visualizations |

---

## Team Members

This project was completed collaboratively. Each team member contributed significantly to the development, analysis, and integration of the different phases of the pipeline.

| Role | Contribution |
|------|----------------|
| **Yu-Sen Wu** | Data loading and preprocessing; shared ratings and probe; Exploratory data analysis|
| **Eric Wu** | Exploratory data analysis; user/movie behavior and correlations. |
| **Xinqi Huang** | Recommendation engine; baselines, SVD, NMF, hybrid; RMSE evaluation. |
| **Kun-Yu Lee** | Clustering; user and movie features, K-Means, hierarchical, DBSCAN, t-SNE. |
| **Yuanyuan Xie** | RFM analysis ; segments, visualizations, business insights; cross-reference with clustering.|

---

## Presentation and Deliverables

The final presentation follows a **7-block structure**:

1. **Title + Team**
2. **Business Problem + Dataset Overview**
3. **EDA Highlights** — Best charts from Person A/B (e.g. 4–5 selected figures)
4. **Recommendation Engine** — Person C’s model comparison table and probe RMSE
5. **Clustering** — Person D’s visualizations and cluster interpretations
6. **RFM Analysis** — Person E’s segmentation and business insights
7. **Conclusions + Future Work**

**Project integration:** `requirements.txt`, this README, and ensuring all notebooks run from start to finish with a consistent markdown/style. Final artifacts include the five notebooks, the presentation, `requirements.txt`, and `README.md`.
