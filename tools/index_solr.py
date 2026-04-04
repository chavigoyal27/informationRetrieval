"""
Index the corpus into Apache Solr using pre-computed classification results.

Joins corpus_balanced_1to1.csv (url, title, date) with
classification_results.csv (subjectivity, polarity, emotion) on id,
and indexes the merged documents into Solr.
"""

import csv
import re
import sys
import time
from datetime import datetime

import pysolr

SOLR_URL = "http://localhost:8983/solr/opinions"
CORPUS_PATH = "data/final_corpus/corpus_balanced_1to1.csv"
CLASSIFICATION_PATH = "data/analysis/classification_results.csv"
BATCH_SIZE = 500


def parse_date(date_str):
    """Try to parse a date string into Solr-compatible ISO format."""
    if not date_str or date_str.strip() == "":
        return None
    date_str = date_str.strip()
    if date_str.endswith("Z"):
        return date_str
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    return None


def clean_for_dedup(t):
    t = t.lower()
    t = re.sub(r"#\w+", "", t)
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_title_like(text, title):
    if not text or not title:
        return False
    return clean_for_dedup(text) == clean_for_dedup(title)


def main():
    print("Connecting to Solr...")
    solr = pysolr.Solr(SOLR_URL, always_commit=False, timeout=30)

    try:
        solr.ping()
        print("Solr connection OK")
    except Exception as e:
        print(f"ERROR: Cannot connect to Solr at {SOLR_URL}")
        print(f"  {e}")
        print("Make sure Solr is running: docker compose up -d")
        sys.exit(1)

    print("Clearing existing index...")
    solr.delete(q="*:*")
    solr.commit()

    # Load classification results into a lookup by id
    print(f"Loading classification results from {CLASSIFICATION_PATH}...")
    classification = {}
    with open(CLASSIFICATION_PATH, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            classification[row["id"]] = row
    print(f"  Loaded {len(classification):,} classification records")

    print(f"Reading corpus from {CORPUS_PATH}...")
    docs = []
    total = 0
    indexed = 0
    skipped = 0
    seen_texts = set()

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            text = row.get("text", "").strip()
            title = row.get("title", "").strip()
            row_id = row.get("id", "")

            if not text:
                skipped += 1
                continue

            if is_title_like(text, title):
                skipped += 1
                continue

            if len(text.split()) < 4:
                skipped += 1
                continue

            clean_text = clean_for_dedup(text)
            if clean_text in seen_texts:
                skipped += 1
                continue
            seen_texts.add(clean_text)

            # Look up classification results for this id
            cls = classification.get(row_id, {})
            polarity = cls.get("polarity", "neutral")
            polarity_score = float(cls.get("polarity_score", 0))
            subjectivity = cls.get("subjectivity", "neutral")
            subjectivity_score = float(cls.get("subjectivity_score", 0))
            emotion = cls.get("emotion", "neutral")
            emotion_score = float(cls.get("emotion_score", 0))

            doc = {
                "id": row_id,
                "source": row.get("source", ""),
                "url": row.get("url", ""),
                "title": title,
                "text": text,
                "sentiment": polarity,
                "sentiment_score": round(polarity_score, 4),
                "subjectivity": subjectivity,
                "subjectivity_score": round(subjectivity_score, 4),
                "emotion": emotion,
                "emotion_score": round(emotion_score, 4),
            }

            parsed_date = parse_date(row.get("date", ""))
            if parsed_date:
                doc["date"] = parsed_date

            docs.append(doc)

            if len(docs) >= BATCH_SIZE:
                solr.add(docs)
                indexed += len(docs)
                print(f"  Indexed {indexed}/{total} records...")
                docs = []

    if docs:
        solr.add(docs)
        indexed += len(docs)

    solr.commit()
    print(f"\nDone! Indexed {indexed} documents ({skipped} skipped, {total} total rows)")

    results = solr.search("*:*", rows=0)
    print(f"Solr reports {results.hits} documents in the index")


if __name__ == "__main__":
    main()
