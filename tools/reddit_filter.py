"""
Reddit CSV Filter — AI in Education
====================================
Reads data/crawled/redditcrawl.csv and keeps only rows relevant to
"AI in Education" based on keyword matching.

Usage:
    python reddit_filter.py                # uses default input/output paths
    python reddit_filter.py -i input.csv   # custom input
    python reddit_filter.py -o output.csv  # custom output
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "..", "data", "crawled", "redditcrawl.csv")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "..", "data", "crawled", "redditcrawl_filtered.csv")

# ── Keywords ─────────────────────────────────────────────────────────────────
# A row is kept if it contains at least one AI term AND at least one Education term.

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
]

EDUCATION_KEYWORDS = [
    r"\beducat\w*\b",
    r"\bschool\b",
    r"\buniversit\w*\b",
    r"\bcollege\b",
    r"\bstudent\w*\b",
    r"\bteacher\w*\b",
    r"\bprofessor\w*\b",
    r"\bclassroom\b",
    r"\bcurriculum\b",
    r"\bacademi\w*\b",
    r"\blearn\w*\b",
    r"\bteach\w*\b",
    r"\bhomework\b",
    r"\bassignment\w*\b",
    r"\bexam\w*\b",
    r"\blecture\w*\b",
    r"\btutoring\b",
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
]

# Pre-compile patterns for speed
_ai_pattern = re.compile("|".join(AI_KEYWORDS), re.IGNORECASE)
_edu_pattern = re.compile("|".join(EDUCATION_KEYWORDS), re.IGNORECASE)


def is_relevant(row: dict) -> bool:
    """Return True if the row mentions both AI and Education."""
    text = (row.get("question_title", "") + " " + row.get("answer_text", "")).lower()
    return bool(_ai_pattern.search(text)) and bool(_edu_pattern.search(text))


def main():
    parser = argparse.ArgumentParser(description="Filter redditcrawl.csv for AI in Education relevance.")
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    filtered = [r for r in rows if is_relevant(r)]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered)

    print(f"Input:    {len(rows)} rows")
    print(f"Filtered: {len(filtered)} rows (kept)")
    print(f"Removed:  {len(rows) - len(filtered)} rows")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
