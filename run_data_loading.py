#!/usr/bin/env python3
"""
Runs the logic of 01_data_loading to generate ratings.parquet and probe.parquet.
(No need to open Jupyter, just run: python run_data_loading.py)
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_FRACTION = 1.0  # 0.1 = 10% of movies (for development); 1.0 = load all
DATA_DIR = PROJECT_ROOT / "dataset"
TRAINING_DIR = DATA_DIR / "training_set"
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_movie_titles(filepath: Path) -> pd.DataFrame:
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            if len(parts) >= 3:
                movie_id = int(parts[0])
                year = int(parts[1]) if parts[1].isdigit() else None
                title = parts[2].strip()
                rows.append({"MovieID": movie_id, "YearOfRelease": year, "Title": title})
            elif len(parts) == 2:
                rows.append({"MovieID": int(parts[0]), "YearOfRelease": int(parts[1]) if parts[1].isdigit() else None, "Title": ""})
    return pd.DataFrame(rows)

def parse_single_movie_file(filepath: Path) -> list:
    rows = []
    movie_id = None
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(":"):
                movie_id = int(line[:-1])
                continue
            parts = line.split(",")
            if len(parts) >= 3 and movie_id is not None:
                try:
                    rows.append({
                        "MovieID": movie_id,
                        "CustomerID": int(parts[0]),
                        "Rating": int(parts[1]),
                        "Date": parts[2]
                    })
                except (ValueError, IndexError):
                    pass
    return rows

def load_all_ratings(training_dir: Path) -> pd.DataFrame:
    import numpy as np
    movie_files = sorted(training_dir.glob("mv_*.txt"))
    if SAMPLE_FRACTION < 1.0:
        np.random.seed(42)
        n_sample = max(1, int(len(movie_files) * SAMPLE_FRACTION))
        movie_files = np.random.choice(movie_files, n_sample, replace=False).tolist()
        print(f"Development mode: Loading only {n_sample} files ({SAMPLE_FRACTION*100:.0f}%)")
    print(f"Found {len(movie_files)} movie rating files")
    all_rows = []
    for fp in tqdm(movie_files, desc="Parsing rating files"):
        rows = parse_single_movie_file(fp)
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)

def load_probe(filepath: Path) -> pd.DataFrame:
    rows = []
    movie_id = None
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.endswith(":"):
                movie_id = int(line[:-1])
                continue
            if movie_id is not None:
                try:
                    rows.append({"MovieID": movie_id, "CustomerID": int(line)})
                except ValueError:
                    pass
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print("1. Loading movie_titles...")
    movie_titles = load_movie_titles(DATA_DIR / "movie_titles.txt")
    print(f"   Number of movies: {len(movie_titles)}")

    print("2. Parsing training_set...")
    ratings_raw = load_all_ratings(TRAINING_DIR)
    print(f"   Total number of ratings: {len(ratings_raw):,}")

    print("3. Merging movie information and extracting temporal features...")
    ratings = ratings_raw.merge(movie_titles, on="MovieID", how="left")
    ratings["Date"] = pd.to_datetime(ratings["Date"])
    ratings["Year"] = ratings["Date"].dt.year
    ratings["Month"] = ratings["Date"].dt.month
    ratings["DayOfWeek"] = ratings["Date"].dt.dayofweek

    print("4. Saving ratings.parquet...")
    ratings.to_parquet(OUTPUT_DIR / "ratings.parquet", index=False)
    p = OUTPUT_DIR / "ratings.parquet"
    print(f"   Saved: {p} ({p.stat().st_size / 1024 / 1024:.2f} MB)")

    print("5. Parsing probe.txt...")
    probe = load_probe(DATA_DIR / "probe.txt")
    probe.to_parquet(OUTPUT_DIR / "probe.parquet", index=False)
    print(f"   Number of probe records: {len(probe):,}")

    print("Done!")
