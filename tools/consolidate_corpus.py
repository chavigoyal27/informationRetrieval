"""
Consolidate all crawled CSVs into a single master corpus file.

Normalises every source to a unified schema:
    id, source, url, title, text, date

- Drops rows with empty/whitespace-only text
- Removes exact-duplicate texts (keeps first occurrence)
- Strips leading/trailing whitespace from all fields
- Writes to crawled_data/master_corpus.csv

Usage:
    python tools/consolidate_corpus.py
"""

import csv
import os
import re
from datetime import datetime

csv.field_size_limit(10 ** 7)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRAWLED_DIR = os.path.join(BASE_DIR, "crawled_data")
OUTPUT_FILE = os.path.join(CRAWLED_DIR, "master_corpus.csv")

FIELDNAMES = ["id", "source", "url", "title", "text", "date"]


def clean(text: str) -> str:
    """Collapse whitespace and strip."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def parse_quora(filepath: str) -> list[dict]:
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "source": "Quora",
                "url": clean(r.get("url", "")),
                "title": clean(r.get("question_title", "")),
                "text": clean(r.get("answer_text", "")),
                "date": clean(r.get("scraped_at", "")),
            })
    return rows


def parse_youtube(filepath: str) -> list[dict]:
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "source": "YouTube",
                "url": f"https://www.youtube.com/watch?v={r.get('video_id', '')}",
                "title": clean(r.get("video_title", "")),
                "text": clean(r.get("text", "")),
                "date": clean(r.get("published_at", "")),
            })
    return rows


def parse_twitter(filepath: str) -> list[dict]:
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "source": "Twitter",
                "url": clean(r.get("url", "")),
                "title": clean(r.get("question_title", "")),
                "text": clean(r.get("answer_text", "")),
                "date": "",
            })
    return rows


def parse_linkedin(filepath: str) -> list[dict]:
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "source": "LinkedIn",
                "url": clean(r.get("url", "")),
                "title": clean(r.get("question_title", "")),
                "text": clean(r.get("answer_text", "")),
                "date": "",
            })
    return rows


def parse_reddit(filepath: str) -> list[dict]:
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "source": "Reddit",
                "url": clean(r.get("url", "")),
                "title": clean(r.get("question_title", "")),
                "text": clean(r.get("answer_text", "")),
                "date": clean(r.get("scraped_at", "")),
            })
    return rows


SOURCES = [
    ("quoracrawl.csv", parse_quora),
    ("youtubecrawl.csv", parse_youtube),
    ("twitterxcrawl.csv", parse_twitter),
    ("linkedincrawl.csv", parse_linkedin),
    ("redditcrawl.csv", parse_reddit),
]


def main():
    all_rows: list[dict] = []

    for filename, parser in SOURCES:
        filepath = os.path.join(CRAWLED_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  SKIP  {filename} (not found)")
            continue
        rows = parser(filepath)
        print(f"  READ  {filename}: {len(rows)} records")
        all_rows.extend(rows)

    print(f"\nTotal raw records: {len(all_rows)}")

    # Drop rows with empty text
    before = len(all_rows)
    all_rows = [r for r in all_rows if r["text"]]
    dropped_empty = before - len(all_rows)
    print(f"Dropped {dropped_empty} empty-text rows")

    # Deduplicate by exact text match (keep first occurrence)
    seen_texts: set[str] = set()
    unique_rows: list[dict] = []
    for r in all_rows:
        text_key = r["text"].lower()
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            unique_rows.append(r)
    dropped_dupes = len(all_rows) - len(unique_rows)
    print(f"Dropped {dropped_dupes} duplicate rows")

    all_rows = unique_rows

    # Assign sequential IDs
    for i, row in enumerate(all_rows, start=1):
        row["id"] = i

    # Write output
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary stats
    total_words = sum(len(r["text"].split()) for r in all_rows)
    word_set: set[str] = set()
    for r in all_rows:
        for w in r["text"].split():
            word_set.add(w.lower())

    print(f"\n--- Master Corpus Summary ---")
    print(f"Records:      {len(all_rows):,}")
    print(f"Total words:  {total_words:,}")
    print(f"Unique types: {len(word_set):,}")
    print(f"Output:       {OUTPUT_FILE}")

    # Per-source breakdown
    source_counts: dict[str, int] = {}
    for r in all_rows:
        source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1
    print(f"\nPer-source:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:10s}  {count:,}")


if __name__ == "__main__":
    main()
