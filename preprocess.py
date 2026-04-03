"""
preprocess.py — Text Preprocessing for Classification
======================================================
Cleans and normalizes raw social media text before sentiment classification.

Steps:
  1. Convert emojis to text tokens
  2. Remove URLs
  3. Remove HTML tags
  4. Lowercase
  5. Collapse whitespace
  6. Truncate to max token length

Usage:
    from preprocess import preprocess

    clean_text = preprocess("Some raw text with 🔥 and https://example.com")
"""

import re

import emoji


def preprocess(text: str, max_tokens: int = 512) -> str:
    """
    Clean text for classification:
      - Convert emojis to text tokens (e.g. 🔥 → ':fire:')
      - Remove URLs
      - Remove HTML tags
      - Lowercase
      - Collapse whitespace
      - Truncate to max_tokens words (RoBERTa hard limit)
    """
    if not text:
        return ""

    # Convert emojis to text
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Lowercase
    text = text.lower()

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate to max_tokens words (rough approximation for RoBERTa's 512 token limit)
    words = text.split()
    if len(words) > max_tokens:
        text = " ".join(words[:max_tokens])

    return text
