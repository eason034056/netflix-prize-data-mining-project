import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import mode as scipy_mode

cwd = Path(__file__).resolve().parent
PROJECT_ROOT = cwd
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "04_clustering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RATINGS_PATH = DATA_DIR / "ratings.parquet"
if not RATINGS_PATH.exists():
    raise FileNotFoundError(f"ratings.parquet not found at: {RATINGS_PATH}")

ratings = pd.read_parquet(RATINGS_PATH)

# Section 4.1: Feature Engineering

user_grp = ratings.groupby("CustomerID")

user_features = user_grp["Rating"].agg(
    rating_count="count",
    rating_mean="mean",
    rating_std="std",
).fillna(0)

freq = ratings.groupby(['CustomerID', 'Rating'], observed=True).size().reset_index(name='count')
most_common = freq.sort_values('count', ascending=False).drop_duplicates('CustomerID').set_index('CustomerID')['Rating'].rename("most_common_rating")
user_features = user_features.join(most_common)

five_star_cnt = ratings[ratings["Rating"] == 5].groupby("CustomerID").size().rename("_5s")
user_features = user_features.join(five_star_cnt, how="left").fillna({"_5s": 0})
user_features["five_star_pct"] = user_features["_5s"] / user_features["rating_count"]
user_features.drop(columns=["_5s"], inplace=True)

if "Date" in ratings.columns:
    dates = pd.to_datetime(ratings["Date"])
    dr = dates.groupby(ratings["CustomerID"]).agg(["min", "max"])
    user_features["date_range_days"] = (dr["max"] - dr["min"]).dt.days.fillna(0).astype(int)
else:
    user_features["date_range_days"] = 0

movie_grp = ratings.groupby("MovieID")

movie_features = movie_grp["Rating"].agg(
    rating_count="count",
    rating_mean="mean",
    rating_std="std",
    rating_skewness=lambda x: float(x.skew()),
).fillna(0)

if "YearOfRelease" in ratings.columns:
    year_map = ratings.groupby("MovieID")["YearOfRelease"].first()
    movie_features = movie_features.join(year_map, how="left")
    movie_features["YearOfRelease"] = movie_features["YearOfRelease"].fillna(
        movie_features["YearOfRelease"].median()
    )
else:
    movie_features["YearOfRelease"] = 2000.0

# StandardScaler + PCA
scaler_u = StandardScaler()
scaler_m = StandardScaler()

user_scaled  = scaler_u.fit_transform(user_features)
movie_scaled = scaler_m.fit_transform(movie_features)

pca3_u = PCA(n_components=3, random_state=42)
user_pca3 = pca3_u.fit_transform(user_scaled)

pca3_m = PCA(n_components=3, random_state=42)
movie_pca3 = pca3_m.fit_transform(movie_scaled)

user_pca2  = PCA(n_components=2, random_state=42).fit_transform(user_scaled)
movie_pca2 = PCA(n_components=2, random_state=42).fit_transform(movie_scaled)

# Section 4.2: User Clustering
K_RANGE = range(2, 10)
inertias_u, sil_u = [], []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(user_pca3)
    inertias_u.append(km.inertia_)
    s = silhouette_score(user_pca3, lbl, sample_size=10000, random_state=42)
    sil_u.append(s)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(list(K_RANGE), inertias_u, "o-", color="steelblue", lw=2)
axes[0].set_title("User Clustering", fontsize=13)
axes[0].grid(alpha=0.3)

axes[1].plot(list(K_RANGE), sil_u, "s-", color="coral", lw=2)
axes[1].set_title("User Silhouette", fontsize=13)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "user_elbow_silhouette.png", dpi=150)
plt.close()

best_k_user = list(K_RANGE)[int(np.argmax(sil_u))]

km_user = KMeans(n_clusters=best_k_user, random_state=42, n_init=10)
user_features["cluster"] = km_user.fit_predict(user_pca3)

CLUSTER_LABELS_USER = {
    0: "Casual Users",
    1: "Power Users",
    2: "Harsh Critics",
    3: "Generous Raters",
    4: "Selective Viewers",
    5: "Active Explorers",
}
user_features["cluster_name"] = user_features["cluster"].map(
    lambda c: CLUSTER_LABELS_USER.get(c, f"Cluster {c}")
)

COLORS = plt.cm.tab10.colors
fig, ax = plt.subplots(figsize=(9, 6))
for cid in sorted(user_features["cluster"].unique()):
    mask = user_features["cluster"].values == cid
    ax.scatter(user_pca2[mask, 0], user_pca2[mask, 1],
               s=5, alpha=0.35, color=COLORS[cid % len(COLORS)])
ax.set_title(f"User PCA 2D (k={best_k_user})", fontsize=13)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "user_kmeans_pca.png", dpi=150)
plt.close()

# Section 4.3: Movie Clustering
K_RANGE_M = range(2, 9)
inertias_m, sil_m = [], []

for k in K_RANGE_M:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(movie_pca3)
    inertias_m.append(km.inertia_)
    s = silhouette_score(movie_pca3, lbl, sample_size=None)
    sil_m.append(s)

best_k_movie = list(K_RANGE_M)[int(np.argmax(sil_m))]

km_movie = KMeans(n_clusters=best_k_movie, random_state=42, n_init=10)
movie_features["cluster"] = km_movie.fit_predict(movie_pca3)

CLUSTER_LABELS_MOVIE = {
    0: "Blockbusters",
    1: "Niche Favorites",
    2: "Polarizing Films",
    3: "Forgotten Films",
    4: "Classic Hits",
    5: "Cult Favourites",
}
movie_features["cluster_name"] = movie_features["cluster"].map(
    lambda c: CLUSTER_LABELS_MOVIE.get(c, f"Cluster {c}")
)

fig, ax = plt.subplots(figsize=(9, 6))
for cid in sorted(movie_features["cluster"].unique()):
    mask = movie_features["cluster"].values == cid
    ax.scatter(movie_pca2[mask, 0], movie_pca2[mask, 1],
               s=20, alpha=0.55, color=COLORS[cid % len(COLORS)])
ax.set_title(f"Movie PCA 2D (k={best_k_movie})", fontsize=13)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "movie_kmeans_pca.png", dpi=150)
plt.close()

# Hierarchical Clustering + Dendrogram
MAX_HIER = 500
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(movie_scaled), size=min(MAX_HIER, len(movie_scaled)), replace=False)
movie_sample = movie_scaled[sample_idx]

Z = linkage(movie_sample, method="ward")

fig, ax = plt.subplots(figsize=(15, 6))
dendrogram(Z, ax=ax,
           truncate_mode="lastp", p=30,
           leaf_rotation=45, leaf_font_size=8,
           color_threshold=0.7 * max(Z[:, 2]))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "movie_dendrogram.png", dpi=150)
plt.close()

agg_movie = AgglomerativeClustering(n_clusters=best_k_movie, linkage="ward")
movie_features["cluster_hierarchical"] = agg_movie.fit_predict(movie_scaled)

# Section 4.4: Advanced Analysis
# DBSCAN
MAX_DBSCAN_U = 50_000
db_idx = rng.choice(len(user_pca3), size=min(MAX_DBSCAN_U, len(user_pca3)), replace=False)
user_pca3_sub = user_pca3[db_idx]
user_pca2_sub = user_pca2[db_idx]

db_user = DBSCAN(eps=1.5, min_samples=5, n_jobs=-1)
db_labels = db_user.fit_predict(user_pca3_sub)
n_out_u = (db_labels == -1).sum()

is_out_u = db_labels == -1
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(user_pca2_sub[~is_out_u, 0], user_pca2_sub[~is_out_u, 1],
           s=5, alpha=0.25, color="steelblue")
ax.scatter(user_pca2_sub[is_out_u, 0], user_pca2_sub[is_out_u, 1],
           s=18, alpha=0.8, color="crimson")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "user_dbscan_outliers.png", dpi=150)
plt.close()

db_movie = DBSCAN(eps=1.2, min_samples=3, n_jobs=-1)
movie_features["cluster_dbscan"] = db_movie.fit_predict(movie_scaled)
n_out_m = (movie_features["cluster_dbscan"] == -1).sum()

# t-SNE
MAX_TSNE_U = 5_000
tsne_u_idx = rng.choice(len(user_scaled), size=min(MAX_TSNE_U, len(user_scaled)), replace=False)
user_tsne_labels = user_features["cluster"].values[tsne_u_idx]

user_tsne2 = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(
    user_scaled[tsne_u_idx]
)

fig, ax = plt.subplots(figsize=(9, 7))
for cid in np.unique(user_tsne_labels):
    mask = user_tsne_labels == cid
    ax.scatter(user_tsne2[mask, 0], user_tsne2[mask, 1],
               s=10, alpha=0.45, color=COLORS[cid % len(COLORS)])
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "user_tsne.png", dpi=150)
plt.close()

movie_tsne2 = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(movie_scaled)
movie_tsne_labels = movie_features["cluster"].values

fig, ax = plt.subplots(figsize=(9, 7))
for cid in np.unique(movie_tsne_labels):
    mask = movie_tsne_labels == cid
    ax.scatter(movie_tsne2[mask, 0], movie_tsne2[mask, 1],
               s=22, alpha=0.6, color=COLORS[cid % len(COLORS)])
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "movie_tsne.png", dpi=150)
plt.close()

# Algorithm Comparison
comparison = []
for method, col in [
    ("K-Means",      "cluster"),
    ("Hierarchical", "cluster_hierarchical"),
    ("DBSCAN",       "cluster_dbscan"),
]:
    lbl = movie_features[col].values
    core = lbl != -1
    n_cl = len(set(lbl) - {-1})
    if core.sum() > 1 and n_cl > 1:
        sil = silhouette_score(movie_scaled[core], lbl[core], sample_size=None)
    else:
        sil = float("nan")
    comparison.append({"Method": method, "n_clusters": n_cl, "Silhouette": round(sil, 4)})

comp_df = pd.DataFrame(comparison)

comp_df.to_csv(OUTPUT_DIR / "algorithm_comparison.csv", index=False)

user_features.to_parquet(OUTPUT_DIR / "user_clusters.parquet")
movie_features.to_parquet(OUTPUT_DIR / "movie_clusters.parquet")


print(f"Total Ratings  : {len(ratings):,}")
print(f"Total Users    : {len(user_features):,}  | Best k: {best_k_user} | Silhouette: {max(sil_u):.4f}")
print(f"Total Movies   : {len(movie_features):,}   | Best k: {best_k_movie} | Silhouette: {max(sil_m):.4f}")
print(f"User Outliers  : {n_out_u:,}")
print(f"Movie Outliers : {n_out_m:,}")
print("\nAlgorithm Comparison (Movies):")
print(comp_df.to_string(index=False))
print(f"\nOutputs saved to: {OUTPUT_DIR.name}/")
