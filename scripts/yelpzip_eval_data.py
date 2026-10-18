"""Prepare a local, Git-ignored YelpZip index and frozen evaluation manifests.

Raw YelpZip files remain outside the repository. The SQLite cache and manifests
are generated under data/yelpzip_eval/ (Git-ignored) and can be rebuilt from the
source directory. Labels are stored separately from the inference review data.
"""

import argparse
import hashlib
import csv
import json
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "yelpzip_eval"
DB_PATH = OUTPUT_DIR / "yelpzip.sqlite3"
SEED = 20261018


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(source_dir):
    source_dir = Path(source_dir).expanduser()
    required = ["metadata", "reviewContent", "reviewGraph"]
    missing = [name for name in required if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing YelpZip source files: {', '.join(missing)}")
    return source_dir


def build_index(source_dir, db_path=DB_PATH):
    """Build a local SQLite cache, validating alignment while streaming rows."""
    source_dir = _source_files(source_dir)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    try:
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.execute("PRAGMA temp_store=MEMORY")
        con.execute("""CREATE TABLE reviews (
            review_id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            prod_id INTEGER NOT NULL,
            rating REAL NOT NULL,
            label INTEGER NOT NULL,
            date TEXT NOT NULL,
            review_text TEXT NOT NULL
        )""")
        insert = "INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?)"
        batch = []
        labels = Counter()
        source_hashes = {name: hashlib.sha256() for name in ("metadata", "reviewContent", "reviewGraph")}
        row_count = 0
        paths = [source_dir / n for n in ("metadata", "reviewContent", "reviewGraph")]
        with paths[0].open(encoding="utf-8") as meta, \
             paths[1].open(encoding="utf-8") as content, \
             paths[2].open(encoding="utf-8") as graph:
            for row_id, (ml, cl, gl) in enumerate(zip(meta, content, graph)):
                for name, raw_line in zip(source_hashes, (ml, cl, gl)):
                    source_hashes[name].update(raw_line.encode("utf-8"))
                m = ml.rstrip("\r\n").split("\t")
                c = cl.rstrip("\r\n").split("\t", 3)
                g = gl.rstrip("\r\n").split("\t")
                if len(m) != 5 or len(c) != 4 or len(g) != 3:
                    raise ValueError(f"Malformed row at YelpZip source row {row_id}.")
                user_id, prod_id, rating, raw_label, date = m
                c_user, c_prod, c_date, review_text = c
                g_user, g_prod, g_rating = g
                if (user_id, prod_id, date) != (c_user, c_prod, c_date) or \
                        (user_id, prod_id, rating) != (g_user, g_prod, g_rating):
                    raise ValueError(f"Source files are misaligned at YelpZip row {row_id}.")
                label = int(raw_label)
                if label not in (-1, 1):
                    raise ValueError(f"Unexpected YelpZip label {raw_label!r} at row {row_id}.")
                filtered = int(label == -1)
                labels[filtered] += 1
                batch.append((row_id, int(user_id), int(prod_id), float(rating), filtered, date, review_text))
                row_count += 1
                if len(batch) >= 5000:
                    con.executemany(insert, batch)
                    batch.clear()
                if row_count % 100000 == 0:
                    print(f"Indexed {row_count:,} aligned rows")
        if batch:
            con.executemany(insert, batch)
        # Date-ordered composite indexes support point-in-time history lookups.
        con.execute("CREATE INDEX reviews_user_date ON reviews(user_id, date, review_id)")
        con.execute("CREATE INDEX reviews_prod_date ON reviews(prod_id, date, review_id)")
        con.commit()
        summary = {
            "rows": row_count,
            "filtered": labels[1],
            "recommended": labels[0],
            "filtered_rate": labels[1] / row_count if row_count else 0,
            "db_path": str(db_path),
            "source_dir": str(source_dir),
            "source_file_sha256": {name: digest.hexdigest() for name, digest in source_hashes.items()},
        }
        (db_path.parent / "source_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        con.close()


def _previously_used_ids():
    ids = set()
    for path in (ROOT / "data").glob("*.csv"):
        if path.name == "sample.csv" or "review_id" in pd.read_csv(path, nrows=0).columns:
            frame = pd.read_csv(path, usecols=lambda c: c == "review_id")
            if "review_id" in frame:
                ids.update(int(v) for v in frame["review_id"].dropna())
    # Also exclude the review IDs called out in the written tuning/process notes.
    doc_paths = [ROOT / "PROCESS.md", ROOT / "README.md"]
    doc_paths.extend((ROOT / "experiments").rglob("*"))
    for path in doc_paths:
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(errors="ignore")
        ids.update(int(m.group(1)) for m in re.finditer(r"review_id\s*[:=]?\s*(\d+)", text, re.I))
    return ids


def make_manifests(db_path=DB_PATH, output_dir=OUTPUT_DIR, tune_n=50, dev_n=500, holdout_n=3000):
    """Create disjoint stratified manifests, excluding known project examples."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded = _previously_used_ids()
    con = sqlite3.connect(db_path)
    try:
        available = {0: [], 1: []}
        for review_id, label in con.execute("SELECT review_id, label FROM reviews ORDER BY review_id"):
            if review_id not in excluded:
                available[label].append(review_id)
    finally:
        con.close()
    rng = random.Random(SEED)
    partitions = (("tuning", tune_n), ("development", dev_n), ("final_holdout", holdout_n))
    prevalence = len(available[1]) / (len(available[0]) + len(available[1]))
    positive_counts = {name: round(n * prevalence) for name, n in partitions}
    total_n = sum(n for _, n in partitions)
    positive_n = sum(positive_counts.values())
    requests = {1: positive_n, 0: total_n - positive_n}
    if any(requests[label] > len(available[label]) for label in (0, 1)):
        raise ValueError("Not enough unused YelpZip rows for the requested manifests.")
    selected = {}
    for label in (0, 1):
        rng.shuffle(available[label])
        selected[label] = available[label][:requests[label]]
    ordered = {name: [] for name, _ in partitions}
    for label, ids in selected.items():
        offset = 0
        for name, n in partitions:
            class_n = positive_counts[name] if label == 1 else n - positive_counts[name]
            ordered[name].extend((rid, label) for rid in ids[offset:offset + class_n])
            offset += class_n
    # Catch any rounding remainder in the last partition while preserving disjointness.
    assigned = {rid for rows in ordered.values() for rid, _ in rows}
    if len(assigned) != total_n:
        raise RuntimeError("Manifest construction did not produce the requested disjoint row counts.")
    for name, rows in ordered.items():
        rng.shuffle(rows)
        ids_path = output_dir / f"{name}_ids.csv"
        labels_path = output_dir / f"{name}_labels.csv"
        pd.DataFrame({"review_id": [rid for rid, _ in rows]}).to_csv(ids_path, index=False)
        pd.DataFrame({"review_id": [rid for rid, _ in rows],
                      "true_filtered": [label for _, label in rows]}).to_csv(labels_path, index=False)
    summary = {
        "seed": SEED,
        "excluded_known_ids": len(excluded),
        "excluded_ids_intersecting_corpus": None,
        "source_population_after_exclusion": sum(len(v) for v in available.values()),
        "source_filtered_rate_after_exclusion": prevalence,
        "partitions": {name: {"n": len(rows), "filtered": sum(label for _, label in rows)}
                       for name, rows in ordered.items()},
        "manifest_sha256": {
            name: {"ids": _sha256(output_dir / f"{name}_ids.csv"),
                   "labels": _sha256(output_dir / f"{name}_labels.csv")}
            for name in ordered
        },
        "note": "Keep *_labels.csv files in the scorer path only; never include labels in Jev payloads.",
    }
    source_summary_path = Path(db_path).parent / "source_summary.json"
    if source_summary_path.exists():
        source_summary = json.loads(source_summary_path.read_text())
        summary["source_file_sha256"] = source_summary.get("source_file_sha256", {})
    summary["excluded_ids_intersecting_corpus"] = len(excluded) - sum(
        1 for i in excluded if i < 0 or i >= 608598
    )
    (output_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-index")
    build.add_argument("--source-dir", required=True)
    build.add_argument("--db-path", default=str(DB_PATH))
    manifests = sub.add_parser("make-manifests")
    manifests.add_argument("--db-path", default=str(DB_PATH))
    manifests.add_argument("--output-dir", default=str(OUTPUT_DIR))
    manifests.add_argument("--tuning", type=int, default=50)
    manifests.add_argument("--development", type=int, default=500)
    manifests.add_argument("--holdout", type=int, default=3000)
    args = parser.parse_args()
    if args.command == "build-index":
        build_index(args.source_dir, args.db_path)
    else:
        make_manifests(args.db_path, args.output_dir, args.tuning, args.development, args.holdout)


if __name__ == "__main__":
    main()
