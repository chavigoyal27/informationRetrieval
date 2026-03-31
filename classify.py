"""
classify.py — Sentiment Classification Pipeline
=================================================
Three-stage classification pipeline for the AI in Education opinion corpus.

Stage 1: Subjectivity Detection  — neutral vs opinionated
Stage 2: Polarity Detection      — positive vs negative (opinionated only)
Stage 3: Emotion Detection       — fear / anger / excitement / concern (opinionated only)

Models used:
  - Subjectivity: TextBlob subjectivity score (rule-based baseline)
  - Polarity:     cardiffnlp/twitter-roberta-base-sentiment-latest (HuggingFace)
  - Emotion:      j-hartmann/emotion-english-distilroberta-base (HuggingFace)

Usage:
    # Install dependencies first:
    pip install transformers torch pandas emoji textblob xlrd openpyxl

    # Run on full corpus:
    python classify.py

    # Run on eval set only (faster, for evaluation):
    python classify.py --eval-only

Output:
    data/analysis/classification_results.csv
"""

import argparse
import csv
import os
import re
import time

import emoji
import pandas as pd
from textblob import TextBlob
from transformers import pipeline

csv.field_size_limit(10 ** 7)

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_FILE = os.path.join(BASE_DIR, "data", "final_corpus", "corpus_balanced_1to1.csv")
EVAL_FILE   = os.path.join(BASE_DIR, "data", "final_corpus", "eval.xls")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "analysis", "classification_results.csv")

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(text: str, max_tokens: int = 512) -> str:
    """
    Clean text for classification:
      - Convert emojis to text tokens (e.g. 🔥 → ':fire:')
      - Remove URLs
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


# ── Stage 1: Subjectivity Detection ──────────────────────────────────────────

SUBJECTIVITY_THRESHOLD = 0.15  # texts below this are considered neutral/objective

def detect_subjectivity(text: str) -> tuple[str, float]:
    """
    Classify text as 'opinionated' or 'neutral' using TextBlob subjectivity score.
    Score ranges from 0.0 (very objective) to 1.0 (very subjective).

    We use a low threshold (0.15) because social media text tends to be less
    formally opinionated than traditional reviews, and we want to retain more
    potentially opinionated content for polarity analysis.

    Returns:
        (label, score) where label is 'opinionated' or 'neutral'
    """
    blob = TextBlob(text)
    score = blob.sentiment.subjectivity
    label = "opinionated" if score >= SUBJECTIVITY_THRESHOLD else "neutral"
    return label, round(score, 4)


# ── Stage 2: Polarity Detection ───────────────────────────────────────────────

def load_polarity_model():
    """
    Load the Cardiff NLP RoBERTa sentiment model.
    Pre-trained on ~124M tweets — ideal for social media opinion data.
    Labels: positive, negative, neutral
    """
    print("Loading polarity model (cardiffnlp/twitter-roberta-base-sentiment-latest)...")
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        max_length=512,
        truncation=True,
        device=-1  # CPU; change to 0 for GPU
    )


def detect_polarity(texts: list[str], polarity_pipe, batch_size: int = 16) -> list[tuple[str, float]]:
    """
    Run polarity detection in batches.
    Returns list of (label, score) tuples.
    Labels: 'positive', 'negative', 'neutral'
    """
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        outputs = polarity_pipe(batch)
        for out in outputs:
            label = out["label"].lower()
            score = round(out["score"], 4)
            results.append((label, score))
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Polarity: processed {min(i + batch_size, len(texts))}/{len(texts)}")
    return results


# ── Stage 3: Emotion Detection ────────────────────────────────────────────────

def load_emotion_model():
    """
    Load the Hartmann emotion classification model.
    Labels: anger, disgust, fear, joy, neutral, sadness, surprise
    We map these to education-relevant categories:
      joy      → excitement
      fear     → concern  
      anger    → anger
      sadness  → concern
      disgust  → anger
      surprise → excitement
      neutral  → neutral
    """
    print("Loading emotion model (j-hartmann/emotion-english-distilroberta-base)...")
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        max_length=512,
        truncation=True,
        device=-1  # CPU; change to 0 for GPU
    )


EMOTION_MAP = {
    "joy":      "excitement",
    "fear":     "concern",
    "anger":    "anger",
    "sadness":  "concern",
    "disgust":  "anger",
    "surprise": "excitement",
    "neutral":  "neutral",
}


def detect_emotion(texts: list[str], emotion_pipe, batch_size: int = 16) -> list[tuple[str, float]]:
    """
    Run emotion detection in batches.
    Returns list of (mapped_emotion, score) tuples.
    """
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        outputs = emotion_pipe(batch)
        for out in outputs:
            raw_label = out["label"].lower()
            mapped = EMOTION_MAP.get(raw_label, "neutral")
            score = round(out["score"], 4)
            results.append((mapped, score))
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Emotion: processed {min(i + batch_size, len(texts))}/{len(texts)}")
    return results


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(records: list[dict], polarity_pipe, emotion_pipe) -> list[dict]:
    """
    Run all three classification stages on a list of records.
    Each record must have at least an 'id', 'source', and 'text' field.
    """
    total = len(records)
    print(f"\nProcessing {total:,} records through 3-stage pipeline...\n")

    # Preprocess all texts
    print("Preprocessing texts...")
    texts_clean = [preprocess(r.get("text", "")) for r in records]

    # ── Stage 1: Subjectivity ────────────────────────────────────────────────
    print("\n── Stage 1: Subjectivity Detection ──")
    subjectivity_results = [detect_subjectivity(t) for t in texts_clean]
    subj_labels  = [r[0] for r in subjectivity_results]
    subj_scores  = [r[1] for r in subjectivity_results]

    opinionated_mask = [l == "opinionated" for l in subj_labels]
    opinionated_texts = [t for t, m in zip(texts_clean, opinionated_mask) if m]
    n_opinionated = sum(opinionated_mask)
    print(f"  Opinionated: {n_opinionated:,} ({100*n_opinionated/total:.1f}%)")
    print(f"  Neutral:     {total - n_opinionated:,} ({100*(total-n_opinionated)/total:.1f}%)")

    # ── Stage 2: Polarity ────────────────────────────────────────────────────
    print("\n── Stage 2: Polarity Detection ──")
    polarity_results_opinionated = detect_polarity(opinionated_texts, polarity_pipe)

    # Map back to full list
    polarity_labels = []
    polarity_scores = []
    op_iter = iter(polarity_results_opinionated)
    for is_op in opinionated_mask:
        if is_op:
            label, score = next(op_iter)
            polarity_labels.append(label)
            polarity_scores.append(score)
        else:
            polarity_labels.append("neutral")
            polarity_scores.append(0.0)

    # ── Stage 3: Emotion ─────────────────────────────────────────────────────
    print("\n── Stage 3: Emotion Detection ──")
    emotion_results_opinionated = detect_emotion(opinionated_texts, emotion_pipe)

    emotion_labels = []
    emotion_scores = []
    em_iter = iter(emotion_results_opinionated)
    for is_op in opinionated_mask:
        if is_op:
            label, score = next(em_iter)
            emotion_labels.append(label)
            emotion_scores.append(score)
        else:
            emotion_labels.append("neutral")
            emotion_scores.append(0.0)

    # ── Assemble results ─────────────────────────────────────────────────────
    output = []
    for i, record in enumerate(records):
        output.append({
            "id":               record.get("id", i + 1),
            "source":           record.get("source", ""),
            "text":             record.get("text", ""),
            "subjectivity":     subj_labels[i],
            "subjectivity_score": subj_scores[i],
            "polarity":         polarity_labels[i],
            "polarity_score":   polarity_scores[i],
            "emotion":          emotion_labels[i],
            "emotion_score":    emotion_scores[i],
        })

    return output


# ── Load Data ─────────────────────────────────────────────────────────────────

def load_corpus(filepath: str) -> list[dict]:
    """Load the balanced corpus CSV."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def load_eval(filepath: str) -> list[dict]:
    """Load the eval.xls file as a list of dicts with text and ground truth label."""
    df = pd.read_excel(filepath)
    return df.to_dict(orient="records")


# ── Save Output ───────────────────────────────────────────────────────────────

def save_results(results: list[dict], filepath: str):
    """Save classification results to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = ["id", "source", "text", "subjectivity", "subjectivity_score",
                  "polarity", "polarity_score", "emotion", "emotion_score"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results):,} results to: {filepath}")


# ── Print Summary ─────────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    """Print a summary of classification results."""
    total = len(results)

    subj_counts = {}
    pol_counts  = {}
    emo_counts  = {}

    for r in results:
        subj_counts[r["subjectivity"]] = subj_counts.get(r["subjectivity"], 0) + 1
        pol_counts[r["polarity"]]      = pol_counts.get(r["polarity"], 0) + 1
        emo_counts[r["emotion"]]       = emo_counts.get(r["emotion"], 0) + 1

    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"\nTotal records: {total:,}")

    print("\nSubjectivity:")
    for label, count in sorted(subj_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<15} {count:>6,}  ({100*count/total:.1f}%)")

    print("\nPolarity (opinionated records only):")
    for label, count in sorted(pol_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<15} {count:>6,}  ({100*count/total:.1f}%)")

    print("\nEmotion (opinionated records only):")
    for label, count in sorted(emo_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:<15} {count:>6,}  ({100*count/total:.1f}%)")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="3-stage sentiment classification pipeline")
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Run on eval.xls only (1,000 records) instead of full corpus"
    )
    args = parser.parse_args()

    # Load data
    if args.eval_only:
        print(f"Loading eval dataset from {EVAL_FILE}...")
        raw = load_eval(EVAL_FILE)
        # Add id/source fields if missing
        records = [{"id": i+1, "source": "", "text": r.get("text", ""),
                    "ground_truth": r.get("sentiment_label", "")}
                   for i, r in enumerate(raw)]
        out_file = OUTPUT_FILE.replace(".csv", "_eval.csv")
    else:
        print(f"Loading corpus from {CORPUS_FILE}...")
        records = load_corpus(CORPUS_FILE)
        out_file = OUTPUT_FILE

    print(f"Loaded {len(records):,} records")

    # Load models
    polarity_pipe = load_polarity_model()
    emotion_pipe  = load_emotion_model()

    # Run pipeline
    t0 = time.time()
    results = run_pipeline(records, polarity_pipe, emotion_pipe)
    elapsed = time.time() - t0

    # Save
    save_results(results, out_file)

    # Summary
    print_summary(results)

    # Performance metrics
    rps = len(records) / elapsed
    print(f"\nPerformance:")
    print(f"  Total time:  {elapsed:.1f}s ({elapsed/60:.1f} mins)")
    print(f"  Records/sec: {rps:.1f}")
    print(f"  Est. time for full corpus (28,664 records): {28664/rps/60:.0f} mins")


if __name__ == "__main__":
    main()
