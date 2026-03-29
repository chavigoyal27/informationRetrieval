"""
Index the balanced corpus into Apache Solr.

Reads corpus_balanced_1to1.csv, computes VADER sentiment for each record,
and sends documents to Solr in batches via pysolr.
"""

import csv
import sys
import time
from datetime import datetime

import pysolr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SOLR_URL = "http://localhost:8983/solr/opinions"
CORPUS_PATH = "data/final_corpus/corpus_balanced_1to1.csv"
BATCH_SIZE = 500


def parse_date(date_str):
    """Try to parse a date string into Solr-compatible ISO format."""
    if not date_str or date_str.strip() == "":
        return None
    date_str = date_str.strip()
    # Already ISO with timezone
    if date_str.endswith("Z"):
        return date_str
    # Try common formats
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    return None


def main():
    print("Connecting to Solr...")
    solr = pysolr.Solr(SOLR_URL, always_commit=False, timeout=30)

    # Verify connection
    try:
        solr.ping()
        print("Solr connection OK")
    except Exception as e:
        print(f"ERROR: Cannot connect to Solr at {SOLR_URL}")
        print(f"  {e}")
        print("Make sure Solr is running: docker compose up -d")
        sys.exit(1)

    # Clear existing data
    print("Clearing existing index...")
    solr.delete(q="*:*")
    solr.commit()

    # Initialize VADER
    analyzer = SentimentIntensityAnalyzer()

    print(f"Reading corpus from {CORPUS_PATH}...")
    docs = []
    total = 0
    indexed = 0
    skipped = 0

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            text = row.get("text", "").strip()
            if not text:
                skipped += 1
                continue

            # Compute VADER sentiment
            scores = analyzer.polarity_scores(text)
            compound = scores["compound"]
            if compound >= 0.05:
                sentiment = "positive"
            elif compound <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            doc = {
                "id": row["id"],
                "source": row.get("source", ""),
                "url": row.get("url", ""),
                "title": row.get("title", ""),
                "text": text,
                "sentiment": sentiment,
                "sentiment_score": round(compound, 4),
            }

            # Parse date if available
            parsed_date = parse_date(row.get("date", ""))
            if parsed_date:
                doc["date"] = parsed_date

            docs.append(doc)

            # Send batch
            if len(docs) >= BATCH_SIZE:
                solr.add(docs)
                indexed += len(docs)
                print(f"  Indexed {indexed}/{total} records...")
                docs = []

    # Final batch
    if docs:
        solr.add(docs)
        indexed += len(docs)

    solr.commit()
    print(f"\nDone! Indexed {indexed} documents ({skipped} skipped, {total} total rows)")

    # Verify
    results = solr.search("*:*", rows=0)
    print(f"Solr reports {results.hits} documents in the index")


if __name__ == "__main__":
    main()
