"""
Loads a real-data lookup index for the agent's tools. This public-demo repo
ships a pre-built, trimmed index (data/full_index.pkl.gz) instead of the raw
608K-review YelpZip files, so the repo stays small enough for GitHub/Streamlit
Cloud. The trim keeps every review in the demo sample plus each of those
reviewers'/businesses' most recent real history (up to 40 other reviews per
reviewer, 60 per business) — genuine Yelp data throughout, just capped in
volume rather than full-dataset scale. See dashboard/README.md for details.

review_id = row position in the original merged dataframe (stable, 0-indexed).
"""

import gzip
import pickle
from pathlib import Path

CACHE_PATH = Path(__file__).parent.parent / "data" / "full_index.pkl"
CACHE_PATH_GZ = Path(__file__).parent.parent / "data" / "full_index.pkl.gz"


def load_index():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    if CACHE_PATH_GZ.exists():
        with gzip.open(CACHE_PATH_GZ, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(
        f"No index found at {CACHE_PATH} or {CACHE_PATH_GZ}. "
        "This public repo ships the trimmed data/full_index.pkl.gz — check it wasn't excluded from the deploy."
    )


if __name__ == "__main__":
    idx = load_index()
    df = idx["df"]
    print(f"Indexed {len(df):,} reviews")
    print(f"Unique reviewers: {len(idx['by_user']):,}")
    print(f"Unique businesses: {len(idx['by_prod']):,}")
