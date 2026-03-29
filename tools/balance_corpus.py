"""
Balance Corpus Generator
=========================
Builds two balanced versions of the master corpus by:
  1. Removing all off-topic Quora records
  2. Removing all positive Quora records
  3. Downsampling remaining positive records to achieve target ratios

Outputs:
  - data/final_corpus/corpus_balanced_1to1.csv   (Scenario D — 1:1 pos/neg)
  - data/final_corpus/corpus_balanced_1.5to1.csv (Scenario E — 1.5:1 pos/neg)

Usage:
    python tools/balance_corpus.py
"""

import csv
import os
import random

csv.field_size_limit(10 ** 7)
random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

MASTER_FILE = os.path.join(DATA_DIR, "analysis", "master_corpus.csv")
SENTIMENT_FILE = os.path.join(DATA_DIR, "analysis", "sentiment_distribution.csv")
OFF_TOPIC_FILE = os.path.join(DATA_DIR, "analysis", "off_topic.csv")

OUTPUT_D = os.path.join(DATA_DIR, "final_corpus", "corpus_balanced_1to1.csv")
OUTPUT_E = os.path.join(DATA_DIR, "final_corpus", "corpus_balanced_1.5to1.csv")

FIELDNAMES = ["id", "source", "url", "title", "text", "date"]


def load_sentiment(filepath: str) -> dict[str, str]:
    sentiment = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sentiment[row["id"]] = row["sentiment"]
    return sentiment


def load_off_topic_ids(filepath: str) -> set[str]:
    ids = set()
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            ids.add(row["id"])
    return ids


def write_corpus(filepath: str, rows: list[dict]):
    # Re-assign sequential IDs
    for i, row in enumerate(rows, start=1):
        row["id"] = i

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_stats(label: str, rows: list[dict], sentiment: dict[str, str], orig_ids: dict[int, str]):
    total = len(rows)
    words = sum(len(r["text"].split()) for r in rows)
    pos = sum(1 for r in rows if orig_ids.get(id(r)) == "positive")
    neg = sum(1 for r in rows if orig_ids.get(id(r)) == "negative")
    neu = sum(1 for r in rows if orig_ids.get(id(r)) == "neutral")
    ratio = pos / neg if neg else float("inf")

    print(f"\n  {label}")
    print(f"  Records: {total:,} | Words: {words:,}")
    print(f"  pos={pos:,} ({100*pos/total:.1f}%) neg={neg:,} ({100*neg/total:.1f}%) neu={neu:,} ({100*neu/total:.1f}%)")
    print(f"  Pos/Neg ratio: {ratio:.2f}")

    by_src: dict[str, int] = {}
    for r in rows:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print(f"  Per-source:")
    for src in sorted(by_src, key=lambda s: -by_src[s]):
        print(f"    {src:<12} {by_src[src]:>6,}")


def main():
    print("Loading data...")
    sentiment = load_sentiment(SENTIMENT_FILE)
    off_topic_ids = load_off_topic_ids(OFF_TOPIC_FILE)

    with open(MASTER_FILE, "r", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))

    # Map each row object to its original sentiment (using object id as key)
    orig_sentiment: dict[int, str] = {}
    for r in rows:
        orig_sentiment[id(r)] = sentiment.get(r["id"], "neutral")

    # Step 1: Remove off-topic Quora
    # Step 2: Remove all positive Quora
    base = []
    for r in rows:
        if r["source"] == "Quora":
            if r["id"] in off_topic_ids:
                continue
            if orig_sentiment[id(r)] == "positive":
                continue
        base.append(r)

    print(f"After removing off-topic + positive Quora: {len(base):,} records")

    pos_rows = [r for r in base if orig_sentiment[id(r)] == "positive"]
    neg_rows = [r for r in base if orig_sentiment[id(r)] == "negative"]
    neu_rows = [r for r in base if orig_sentiment[id(r)] == "neutral"]

    print(f"  pos={len(pos_rows):,}  neg={len(neg_rows):,}  neu={len(neu_rows):,}")

    # Scenario D: 1:1 ratio
    sampled_d = random.sample(pos_rows, len(neg_rows))
    kept_d = sampled_d + neg_rows + neu_rows
    random.shuffle(kept_d)
    write_corpus(OUTPUT_D, kept_d)
    print_stats(f"Scenario D (1:1) -> {OUTPUT_D}", kept_d, sentiment, orig_sentiment)

    # Scenario E: 1.5:1 ratio
    target_pos = int(len(neg_rows) * 1.5)
    sampled_e = random.sample(pos_rows, target_pos)
    kept_e = sampled_e + neg_rows + neu_rows
    random.shuffle(kept_e)
    write_corpus(OUTPUT_E, kept_e)
    print_stats(f"Scenario E (1.5:1) -> {OUTPUT_E}", kept_e, sentiment, orig_sentiment)

    print("\nDone.")


if __name__ == "__main__":
    main()
