"""
Reddit CSV Formatter
=====================
Cleans up redditcrawl.csv so that each row is a single, tidy line:
  - Collapses newlines into spaces
  - Strips Reddit markdown (bold, italic, links, headers, bullets)
  - Normalises whitespace (double spaces, leading/trailing)
  - Removes URLs from answer_text
  - Trims overly short or empty rows

Usage:
    python reddit_format.py                # uses default paths
    python reddit_format.py -i in.csv -o out.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "..", "crawled_data", "redditcrawl.csv")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "..", "crawled_data", "redditcrawl.csv")

MIN_ANSWER_LENGTH = 15  # drop rows where cleaned answer_text is shorter than this


def clean_text(text: str) -> str:
    """Clean Reddit markdown and normalise whitespace."""
    # Replace newlines / carriage returns with a space
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")

    # Remove markdown links: [text](url) -> text
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)

    # Remove standalone URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove markdown bold/italic: **text** / *text* / __text__ / _text_
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)

    # Remove markdown headers (# ## ### etc.)
    text = re.sub(r"#+\s*", "", text)

    # Remove markdown bullet points / numbered lists
    text = re.sub(r"^\s*[-*]\s+", "", text)
    text = re.sub(r"^\s*\d+\.\s+", "", text)

    # Remove strikethrough ~~text~~
    text = re.sub(r"~~([^~]+)~~", r"\1", text)

    # Remove blockquotes >
    text = re.sub(r">\s?", "", text)

    # Remove code backticks
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    parser = argparse.ArgumentParser(description="Format redditcrawl.csv for clean, single-line rows.")
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

    cleaned = []
    dropped = 0
    for r in rows:
        r["question_title"] = clean_text(r.get("question_title", ""))
        r["answer_text"] = clean_text(r.get("answer_text", ""))

        if len(r["answer_text"]) < MIN_ANSWER_LENGTH:
            dropped += 1
            continue

        cleaned.append(r)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned)

    print(f"Input:    {len(rows)} rows")
    print(f"Cleaned:  {len(cleaned)} rows")
    print(f"Dropped:  {dropped} rows (answer_text < {MIN_ANSWER_LENGTH} chars)")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
