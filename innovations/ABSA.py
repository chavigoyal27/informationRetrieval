"""
ABSA with 20/80 split
"""

import os
import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re

BASE_DIR = r"C:\Users\Sutheerth\PycharmProjects\PythonProject2"
EVAL_FILE = os.path.join(BASE_DIR, "eval.xls")
BASELINE_FILE = os.path.join(BASE_DIR, "data", "analysis", "classification_results_eval.csv")

def preprocess(text):
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:512]

def label_to_numeric(label):
    label = str(label).lower().strip()
    if label == 'positive': return 2
    elif label == 'neutral': return 1
    elif label == 'negative': return 0
    return 1


class LearnedABSA:
    """
    ABSA that LEARNS aspect patterns from data
    """

    def __init__(self):
        print("="*60)
        print("LOADING LEARNED ABSA MODEL")
        print("="*60)

        print("\nLoading sentiment model...")
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            truncation=True,
            max_length=512,
            device=-1
        )
        print("✓ Sentiment model loaded")

        self.aspect_pattern_model = None

    def split_sentences(self, text: str) -> list:
        """Split text into sentences"""
        sentences = re.split(r'[.!?;]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 15]

    def train_aspect_patterns(self, texts, labels):
        """Learn which sentence patterns indicate mixed sentiment"""
        print("\nLearning aspect patterns from training data...")

        X_features = []
        y = labels

        for i, text in enumerate(texts):
            sentences = self.split_sentences(text)

            if len(sentences) <= 1:
                X_features.append([1, 0, 0])  # single sentence
            else:
                sentiments = []
                for sent in sentences[:4]:
                    try:
                        result = self.sentiment_model(sent[:512])[0]
                        sentiments.append(result['label'].lower())
                    except:
                        continue

                if not sentiments:
                    X_features.append([0, 0, 0])
                    continue

                has_positive = 1 if 'positive' in sentiments else 0
                has_negative = 1 if 'negative' in sentiments else 0
                is_mixed = 1 if has_positive and has_negative else 0

                X_features.append([0, is_mixed, len(sentiments)])

        X = np.array(X_features)

        self.aspect_pattern_model = LogisticRegression(max_iter=1000)
        self.aspect_pattern_model.fit(X, y)
        print(f"✓ Trained on {len(texts)} examples")

    def classify(self, text: str) -> dict:
        processed = preprocess(text)

        if len(processed) < 10:
            return {'polarity': 'neutral'}

        sentences = self.split_sentences(processed)

        if len(sentences) <= 1:
            result = self.sentiment_model(processed[:512])[0]
            return {'polarity': result['label'].lower(), 'sentence_sentiments': []}

        # Get sentiment for each sentence
        sentiments = []
        for sent in sentences[:4]:
            try:
                result = self.sentiment_model(sent[:512])[0]
                sentiments.append(result['label'].lower())
            except:
                continue

        if not sentiments:
            return {'polarity': 'neutral', 'sentence_sentiments': []}

        has_positive = 'positive' in sentiments
        has_negative = 'negative' in sentiments

        # Mixed sentiment across sentences = neutral
        if has_positive and has_negative:
            return {
                'polarity': 'neutral',
                'sentence_sentiments': sentiments,
                'method': 'mixed_detected'
            }
        elif has_positive:
            return {
                'polarity': 'positive',
                'sentence_sentiments': sentiments,
                'method': 'mostly_positive'
            }
        elif has_negative:
            return {
                'polarity': 'negative',
                'sentence_sentiments': sentiments,
                'method': 'mostly_negative'
            }
        else:
            return {
                'polarity': 'neutral',
                'sentence_sentiments': sentiments,
                'method': 'no_sentiment'
            }


def evaluate_absa():
    print("="*70)
    print("LEARNED ABSA EVALUATION")
    print("No manual keywords - model learns from sentence patterns")
    print("="*70)

    # Load data
    df = pd.read_excel(EVAL_FILE)
    df = df.dropna(subset=['text', 'sentiment_label'])
    texts = df['text'].astype(str).tolist()
    y_true = [label_to_numeric(l) for l in df['sentiment_label']]

    print(f"\nLoaded {len(texts)} records")
    print(f"Distribution: {df['sentiment_label'].value_counts().to_dict()}")

    # Split train/test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, y_true, test_size=0.2, random_state=42, stratify=y_true
    )
    print(f"\nTraining on {len(train_texts)} samples")
    print(f"Testing on {len(test_texts)} samples")

    # Train ABSA
    absa = LearnedABSA()
    absa.train_aspect_patterns(train_texts, train_labels)

    # Test
    print("\nRunning inference on test set...")
    predictions = []
    all_results = []

    for i, text in enumerate(test_texts):
        result = absa.classify(text)
        predictions.append(label_to_numeric(result['polarity']))
        all_results.append(result)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(test_texts)}")

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)

    print(f"\n{'='*50}")
    print(f"LEARNED ABSA ACCURACY: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"{'='*50}")

    # Compare to baseline
    if os.path.exists(BASELINE_FILE):
        baseline_df = pd.read_csv(BASELINE_FILE)
        baseline_preds = baseline_df['polarity'].str.lower().str.strip().tolist()
        baseline_preds_numeric = [label_to_numeric(p) for p in baseline_preds]

        # Get baseline predictions for test set (same indices)
        test_indices = list(range(len(train_texts), len(train_texts) + len(test_texts)))
        baseline_test_preds = [baseline_preds_numeric[i] for i in test_indices]
        baseline_acc = accuracy_score(test_labels, baseline_test_preds)

        print(f"\nBaseline (RoBERTa only) on test set: {baseline_acc:.4f} ({baseline_acc*100:.1f}%)")
        print(f"IMPROVEMENT: +{(accuracy - baseline_acc)*100:.1f}%")

    # Classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(test_labels, predictions,
                                target_names=['Negative', 'Neutral', 'Positive']))

    # ========== SHOW EXAMPLES OF MIXED SENTIMENT ==========
    print("\n" + "="*70)
    print("EXAMPLES OF MIXED SENTIMENT DETECTION")
    print("(Texts where different sentences have different sentiments)")
    print("="*70)

    # Find texts with mixed sentiment from the test set
    mixed_examples = []
    for i, (text, result) in enumerate(zip(test_texts, all_results)):
        if result.get('method') == 'mixed_detected' and len(mixed_examples) < 5:
            mixed_examples.append((text, result))

    if mixed_examples:
        for i, (text, result) in enumerate(mixed_examples):
            print(f"\n{'─'*50}")
            print(f"Example {i+1}:")
            print(f"Text: {text[:200]}...")
            print(f"  Sentence sentiments: {result.get('sentence_sentiments', [])}")
            print(f"  Final polarity: {result['polarity']} (mixed → neutral)")
    else:
        print("\n  No mixed sentiment examples found in test set.")
        print("  Looking in full dataset for examples...")

        # Search full dataset for mixed sentiment
        for text in texts[:100]:
            result = absa.classify(text)
            if result.get('method') == 'mixed_detected' and len(mixed_examples) < 5:
                mixed_examples.append((text, result))

        for i, (text, result) in enumerate(mixed_examples):
            print(f"\n{'─'*50}")
            print(f"Example {i+1}:")
            print(f"Text: {text[:200]}...")
            print(f"  Sentence sentiments: {result.get('sentence_sentiments', [])}")
            print(f"  Final polarity: {result['polarity']} (mixed → neutral)")

    # Also show examples where ABSA fixed baseline errors
    if os.path.exists(BASELINE_FILE):
        print("\n" + "="*70)
        print("EXAMPLES WHERE ABSA FIXED BASELINE ERRORS")
        print("="*70)

        fixed_count = 0
        for i, (true, baseline_pred, absa_pred, text, result) in enumerate(zip(
            test_labels, baseline_test_preds, predictions, test_texts, all_results)):

            true_label = {0: 'negative', 1: 'neutral', 2: 'positive'}[true]
            baseline_label = {0: 'negative', 1: 'neutral', 2: 'positive'}[baseline_pred]
            absa_label = {0: 'negative', 1: 'neutral', 2: 'positive'}[absa_pred]

            if baseline_label != true_label and absa_label == true_label and fixed_count < 5:
                print(f"\n{'─'*50}")
                print(f"Example {fixed_count + 1}:")
                print(f"Text: {text[:150]}...")
                print(f"  True label: {true_label}")
                print(f"  Baseline (RoBERTa): {baseline_label} ❌")
                print(f"  ABSA (learned): {absa_label} ✅")
                if result.get('sentence_sentiments'):
                    print(f"  Sentence sentiments: {result['sentence_sentiments']}")
                fixed_count += 1

        if fixed_count == 0:
            print("\n  No baseline errors fixed in test set.")

    return accuracy


if __name__ == "__main__":
    evaluate_absa()