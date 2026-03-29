"""
Sentiment Distribution Checker
===============================
Runs VADER sentiment analysis on the master corpus to show the
positive / negative / neutral distribution.

This is a quick health check to verify the dataset is reasonably
balanced — not the final classifier for the project.

Outputs:
  - Overall sentiment distribution
  - Per-source sentiment breakdown
  - Saves results to data/analysis/sentiment_distribution.csv

Usage:
    python tools/check_sentiment.py
"""

import csv
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

csv.field_size_limit(10 ** 7)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "analysis", "master_corpus.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "analysis", "sentiment_distribution.csv")

# VADER compound score thresholds (standard cutoffs)
POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05


def classify(compound: float) -> str:
    if compound >= POS_THRESHOLD:
        return "positive"
    elif compound <= NEG_THRESHOLD:
        return "negative"
    else:
        return "neutral"


def main():
    analyzer = SentimentIntensityAnalyzer()

    with open(INPUT_FILE, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    overall = {"positive": 0, "negative": 0, "neutral": 0}
    by_source: dict[str, dict[str, int]] = {}

    results = []
    for i, row in enumerate(rows):
        text = row.get("text", "")
        scores = analyzer.polarity_scores(text)
        label = classify(scores["compound"])

        overall[label] += 1
        src = row["source"]
        if src not in by_source:
            by_source[src] = {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
        by_source[src][label] += 1
        by_source[src]["total"] += 1

        results.append({
            "id": row["id"],
            "source": src,
            "sentiment": label,
            "compound": round(scores["compound"], 4),
            "pos": round(scores["pos"], 4),
            "neu": round(scores["neu"], 4),
            "neg": round(scores["neg"], 4),
        })

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1:,} / {total:,} records...")

    # Save per-record results
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "source", "sentiment", "compound", "pos", "neu", "neg"])
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    print()
    print("=" * 60)
    print("SENTIMENT DISTRIBUTION (VADER)")
    print("=" * 60)

    print(f"\nOverall ({total:,} records):")
    for label in ["positive", "negative", "neutral"]:
        count = overall[label]
        pct = 100 * count / total
        bar = "#" * int(pct / 2)
        print(f"  {label:<10} {count:>8,}  ({pct:5.1f}%)  {bar}")

    print(f"\nPer-source breakdown:")
    print(f"  {'Source':<12} {'Total':>7} {'Pos':>7} {'Neg':>7} {'Neu':>7}  {'Pos%':>6} {'Neg%':>6} {'Neu%':>6}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*7}  {'-'*6} {'-'*6} {'-'*6}")
    for src in sorted(by_source, key=lambda s: -by_source[s]["total"]):
        s = by_source[src]
        t = s["total"]
        print(
            f"  {src:<12} {t:>7,} {s['positive']:>7,} {s['negative']:>7,} {s['neutral']:>7,}"
            f"  {100*s['positive']/t:>5.1f}% {100*s['negative']/t:>5.1f}% {100*s['neutral']/t:>5.1f}%"
        )

    pos_neg_ratio = overall["positive"] / overall["negative"] if overall["negative"] else float("inf")
    print(f"\nPositive/Negative ratio: {pos_neg_ratio:.2f}")
    if 0.5 <= pos_neg_ratio <= 2.0:
        print("Balance: GOOD — dataset is reasonably balanced.")
    elif 0.33 <= pos_neg_ratio <= 3.0:
        print("Balance: FAIR — moderate skew, may be acceptable.")
    else:
        print("Balance: SKEWED — consider rebalancing the dataset.")

    print(f"\nPer-record results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
