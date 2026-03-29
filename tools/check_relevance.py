"""
Relevance Checker — AI in Education
=====================================
Checks each record in the master corpus for topical relevance by verifying
it mentions at least one AI keyword AND at least one education keyword.

Outputs:
  - Per-source and overall relevance stats
  - Saves off-topic rows to crawled_data/off_topic.csv for review
  - Prints a sample of off-topic records for manual inspection

Usage:
    python tools/check_relevance.py
"""

import csv
import os
import re
import random

csv.field_size_limit(10 ** 7)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "crawled_data", "master_corpus.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "crawled_data", "off_topic.csv")

# ── Keywords ─────────────────────────────────────────────────────────────────

AI_KEYWORDS = [
    r"\bai\b",
    r"\bartificial intelligence\b",
    r"\bmachine learning\b",
    r"\bdeep learning\b",
    r"\bchatgpt\b",
    r"\bgpt[-\s]?\d*\b",
    r"\bllm\b",
    r"\blarge language model\b",
    r"\bgenai\b",
    r"\bgenerative ai\b",
    r"\bcopilot\b",
    r"\bgemini\b",
    r"\bclaude\b",
    r"\bnlp\b",
    r"\bneural network\b",
    r"\bautomated grading\b",
    r"\bai[ -]tutor\b",
    r"\bai[ -]powered\b",
    r"\bopenai\b",
    r"\btransformer\w*\b",
    r"\bchatbot\w*\b",
]

EDUCATION_KEYWORDS = [
    r"\beducat\w*\b",
    r"\bschool\w*\b",
    r"\buniversit\w*\b",
    r"\bcollege\w*\b",
    r"\bstudent\w*\b",
    r"\bteacher\w*\b",
    r"\bprofessor\w*\b",
    r"\bclassroom\w*\b",
    r"\bcurriculum\b",
    r"\bacademi\w*\b",
    r"\blearn\w*\b",
    r"\bteach\w*\b",
    r"\bhomework\b",
    r"\bassignment\w*\b",
    r"\bexam\w*\b",
    r"\blecture\w*\b",
    r"\btutor\w*\b",
    r"\bpedagog\w*\b",
    r"\be[ -]?learning\b",
    r"\bedtech\b",
    r"\bcourse\w*\b",
    r"\bsyllabus\b",
    r"\bgrading\b",
    r"\bplagiarism\b",
    r"\bacademic integrity\b",
    r"\bhigher ed\b",
    r"\bk[ -]?12\b",
    r"\btraining\b",
]

_ai_pattern = re.compile("|".join(AI_KEYWORDS), re.IGNORECASE)
_edu_pattern = re.compile("|".join(EDUCATION_KEYWORDS), re.IGNORECASE)


def is_relevant(row: dict) -> bool:
    text = (row.get("title", "") + " " + row.get("text", "")).lower()
    return bool(_ai_pattern.search(text)) and bool(_edu_pattern.search(text))


def main():
    with open(INPUT_FILE, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    relevant = []
    off_topic = []
    source_stats: dict[str, dict] = {}

    for row in rows:
        src = row["source"]
        if src not in source_stats:
            source_stats[src] = {"total": 0, "relevant": 0}
        source_stats[src]["total"] += 1

        if is_relevant(row):
            relevant.append(row)
            source_stats[src]["relevant"] += 1
        else:
            off_topic.append(row)

    # Save off-topic rows
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(off_topic)

    # Print results
    print("=" * 60)
    print("RELEVANCE CHECK — AI in Education")
    print("=" * 60)
    print(f"\nTotal records:    {len(rows):,}")
    print(f"On-topic:         {len(relevant):,} ({100 * len(relevant) / len(rows):.1f}%)")
    print(f"Off-topic:        {len(off_topic):,} ({100 * len(off_topic) / len(rows):.1f}%)")

    print(f"\nPer-source breakdown:")
    print(f"  {'Source':<12} {'Total':>8} {'On-topic':>10} {'Rate':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8}")
    for src in sorted(source_stats, key=lambda s: -source_stats[s]["total"]):
        s = source_stats[src]
        pct = 100 * s["relevant"] / s["total"] if s["total"] else 0
        print(f"  {src:<12} {s['total']:>8,} {s['relevant']:>10,} {pct:>7.1f}%")

    # Show sample off-topic records
    if off_topic:
        sample = random.sample(off_topic, min(10, len(off_topic)))
        print(f"\nSample off-topic records ({len(sample)} shown):")
        print("-" * 60)
        for row in sample:
            text_preview = row["text"][:120].replace("\n", " ")
            print(f"  [{row['source']}] {text_preview}...")
            print()

    print(f"Off-topic rows saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
