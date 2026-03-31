import random
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import classify


def load_eval_ground_truth():
    """
    Load ground-truth sentiment labels from eval.xls.
    Returns a list of normalized labels in row order.
    """
    df = pd.read_excel(classify.EVAL_FILE)
    gold = df["sentiment_label"].astype(str).str.strip().str.lower().tolist()
    return gold


def load_eval_predictions():
    """
    Load predicted polarity labels from the CSV produced by:
        python classify.py --eval-only
    """
    pred_file = classify.OUTPUT_FILE.replace(".csv", "_eval.csv")
    df = pd.read_csv(pred_file)
    pred = df["polarity"].astype(str).str.strip().str.lower().tolist()
    return pred


def evaluate_eval_set():
    gold = load_eval_ground_truth()
    pred = load_eval_predictions()

    print("\n" + "=" * 60)
    print("EVALUATION METRICS ON eval.xls")
    print("=" * 60)
    print(f"Total evaluated records: {len(gold)}")
    print(f"Accuracy: {accuracy_score(gold, pred):.4f}\n")

    print("Classification Report:")
    print(classification_report(
        gold,
        pred,
        labels=["positive", "negative", "neutral"],
        digits=4,
        zero_division=0
    ))

    print("Confusion Matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(
        gold,
        pred,
        labels=["positive", "negative", "neutral"]
    ))


def random_accuracy_sample(sample_size=30, seed=42):
    """
    Print a random sample from the full corpus predictions for manual checking.
    """
    full_file = classify.OUTPUT_FILE
    df = pd.read_csv(full_file)

    sample_n = min(sample_size, len(df))
    sampled_rows = df.sample(n=sample_n, random_state=seed)

    print("\n" + "=" * 60)
    print(f"RANDOM ACCURACY TEST SAMPLE (n={sample_n})")
    print("=" * 60)

    for i, (_, row) in enumerate(sampled_rows.iterrows(), 1):
        text = str(row["text"]).replace("\n", " ")
        preview = text[:200] + ("..." if len(text) > 200 else "")

        print(f"\n[{i}]")
        print(f"Text: {preview}")
        print(f"Subjectivity: {row['subjectivity']} ({row['subjectivity_score']})")
        print(f"Polarity: {row['polarity']} ({row['polarity_score']})")
        print(f"Emotion: {row['emotion']} ({row['emotion_score']})")


if __name__ == "__main__":
    evaluate_eval_set()
    random_accuracy_sample(sample_size=30)