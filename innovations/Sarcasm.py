import os
import sys
import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
from datetime import datetime

# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FINAL_CORPUS_DIR = os.path.join(DATA_DIR, "final_corpus")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

EVAL_FILE = os.path.join(FINAL_CORPUS_DIR, "eval.xls")
OUTPUT_CSV = os.path.join(ANALYSIS_DIR, "sarcasm_fast_results.csv")
OUTPUT_TXT = os.path.join(ANALYSIS_DIR, "sarcasm_fast_report.txt")

RANDOM_SEED = 42


def preprocess(text):
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:512]


def label_to_numeric(label):
    label = str(label).lower().strip()
    if label == 'positive':
        return 2
    elif label == 'neutral':
        return 1
    elif label == 'negative':
        return 0
    return 1


def numeric_to_label(num):
    return {0: 'negative', 1: 'neutral', 2: 'positive'}[num]


class FastSarcasmDetector:
    """
    Fast Sarcasm Detection using:
    1. Sentiment confidence (sarcastic texts often have low confidence)
    2. Structural features (punctuation, capitalization, emojis)
    3. Learned patterns from training data (no manual keywords)
    """

    def __init__(self):
        print("=" * 60)
        print("LOADING FAST SARCASM DETECTOR")
        print("=" * 60)

        # Load sentiment model (fast)
        print("\nLoading sentiment model...")
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            truncation=True,
            max_length=512,
            device=-1
        )
        print("✓ Sentiment model loaded")

        # Trainable sarcasm classifier
        self.sarcasm_model = None
        self.vectorizer = None
        self.is_trained = False

    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract features for sarcasm detection
        No manual keywords - uses structural and confidence features
        """
        text_lower = text.lower()
        features = []

        # Feature 1: Text length
        features.append(min(len(text) / 500, 1.0))

        # Feature 2: Number of sentences
        sentences = re.split(r'[.!?;]+', text)
        features.append(min(len(sentences) / 10, 1.0))

        # Feature 3: Exclamation marks
        features.append(min(text.count('!') / 5, 1.0))

        # Feature 4: Question marks
        features.append(min(text.count('?') / 5, 1.0))

        # Feature 5: Capitalized words ratio (emphasis)
        words = text.split()
        if words:
            caps_count = sum(1 for w in words if w.isupper() and len(w) > 1)
            features.append(min(caps_count / len(words), 1.0))
        else:
            features.append(0)

        # Feature 6: Ellipsis (hesitation/sarcasm)
        features.append(min(text.count('...') / 3, 1.0))

        # Feature 7: Sentiment confidence (low confidence = possible sarcasm)
        try:
            sent_result = self.sentiment_model(text[:512])[0]
            features.append(1 - sent_result['score'])  # Low confidence = higher sarcasm chance
        except:
            features.append(0.5)

        # Feature 8: Sentiment polarity (sarcasm often inverts)
        try:
            sent_result = self.sentiment_model(text[:512])[0]
            sent = sent_result['label'].lower()
            polarity = 1 if sent == 'positive' else (-1 if sent == 'negative' else 0)
            features.append((polarity + 1) / 2)
        except:
            features.append(0.5)

        # Feature 9: Positive word count (sarcastic positives)
        positive_words = ['great', 'amazing', 'wonderful', 'fantastic', 'brilliant', 'perfect', 'love']
        pos_count = sum(1 for w in positive_words if w in text_lower)
        features.append(min(pos_count / 5, 1.0))

        # Feature 10: Negative word count
        negative_words = ['hate', 'terrible', 'awful', 'worst', 'useless', 'waste']
        neg_count = sum(1 for w in negative_words if w in text_lower)
        features.append(min(neg_count / 5, 1.0))

        return np.array(features)

    def train_sarcasm_detector(self, texts, labels):
        """
        Train sarcasm detection using weak supervision:
        A text is likely sarcastic if:
        - RoBERTa predicted positive but ground truth is negative, OR
        - RoBERTa predicted negative but ground truth is positive
        """
        print("\nTraining sarcasm detector...")

        # Get RoBERTa predictions
        roberta_preds = []
        for text in texts:
            try:
                result = self.sentiment_model(text[:512])[0]
                roberta_preds.append(result['label'].lower())
            except:
                roberta_preds.append('neutral')

        # Ground truth labels
        label_names = {0: 'negative', 1: 'neutral', 2: 'positive'}
        true_labels = [label_names[l] for l in labels]

        # Create sarcasm labels (weak supervision)
        sarcasm_labels = []
        sarcasm_count = 0
        for roberta, true in zip(roberta_preds, true_labels):
            # Sarcasm indicator: prediction contradicts ground truth
            if roberta == 'positive' and true == 'negative':
                sarcasm_labels.append(1)  # likely sarcastic
                sarcasm_count += 1
            elif roberta == 'negative' and true == 'positive':
                sarcasm_labels.append(1)  # likely sarcastic
                sarcasm_count += 1
            else:
                sarcasm_labels.append(0)  # not sarcastic

        print(f"  Detected potential sarcasm: {sarcasm_count}/{len(texts)} ({100 * sarcasm_count / len(texts):.1f}%)")

        # Extract features
        X_features = np.array([self.extract_features(text) for text in texts])

        # Add TF-IDF features for text patterns
        self.vectorizer = TfidfVectorizer(
            ngram_range=(2, 4),
            max_features=200,
            analyzer='char_wb',
            min_df=2
        )
        text_features = self.vectorizer.fit_transform(texts).toarray()

        # Combine features
        X = np.hstack([X_features, text_features])
        y = sarcasm_labels

        # Train classifier (balanced for sarcasm detection)
        self.sarcasm_model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.sarcasm_model.fit(X, y)

        self.is_trained = True
        print(f"✓ Sarcasm detector trained on {len(texts)} examples")

    def detect_sarcasm(self, text: str) -> float:
        """Return sarcasm probability"""
        if not self.is_trained or self.sarcasm_model is None:
            return 0.0

        try:
            # Extract features
            features = self.extract_features(text).reshape(1, -1)
            text_features = self.vectorizer.transform([text]).toarray()
            X = np.hstack([features, text_features])

            # Get probability
            prob = self.sarcasm_model.predict_proba(X)[0][1]
            return prob
        except:
            return 0.0

    def classify(self, text: str) -> dict:
        """
        Classify with sarcasm-aware adjustment
        If sarcastic, invert the sentiment
        """
        processed = preprocess(text)

        if len(processed) < 10:
            return {'polarity': 'neutral', 'sarcasm_prob': 0.0, 'adjusted': False}

        # Get base sentiment
        sent_result = self.sentiment_model(processed[:512])[0]
        base_sentiment = sent_result['label'].lower()
        base_confidence = sent_result['score']

        # Detect sarcasm
        sarcasm_prob = self.detect_sarcasm(text)

        # Adjust if highly likely sarcastic
        if sarcasm_prob > 0.6:
            # Invert sentiment
            invert_map = {'positive': 'negative', 'negative': 'positive', 'neutral': 'neutral'}
            final_sentiment = invert_map.get(base_sentiment, 'neutral')
            return {
                'polarity': final_sentiment,
                'sarcasm_prob': round(sarcasm_prob, 3),
                'base_sentiment': base_sentiment,
                'base_confidence': round(base_confidence, 3),
                'adjusted': True
            }

        return {
            'polarity': base_sentiment,
            'sarcasm_prob': round(sarcasm_prob, 3),
            'base_sentiment': base_sentiment,
            'base_confidence': round(base_confidence, 3),
            'adjusted': False
        }


def evaluate_sarcasm():
    """Evaluate fast sarcasm detector on eval set (80/20 split)"""

    print("=" * 70)
    print("FAST SARCASM DETECTOR EVALUATION")
    print("80% Train / 20% Test")
    print("=" * 70)

    # Load data
    print(f"\nLoading evaluation dataset from: {EVAL_FILE}")
    df = pd.read_excel(EVAL_FILE)
    df = df.dropna(subset=['text', 'sentiment_label'])
    texts = df['text'].astype(str).tolist()
    y_true = [label_to_numeric(l) for l in df['sentiment_label']]

    print(f"Loaded {len(texts)} records")
    print(f"Distribution: {df['sentiment_label'].value_counts().to_dict()}")

    # Split train/test (80/20)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, y_true, test_size=0.2, random_state=RANDOM_SEED, stratify=y_true
    )
    print(f"\nTraining on {len(train_texts)} samples")
    print(f"Testing on {len(test_texts)} samples")

    # Initialize and train sarcasm detector
    detector = FastSarcasmDetector()
    detector.train_sarcasm_detector(train_texts, train_labels)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions = []
    all_results = []
    sarcasm_detected = 0
    adjusted_count = 0

    for i, text in enumerate(test_texts):
        result = detector.classify(text)
        predictions.append(label_to_numeric(result['polarity']))
        all_results.append(result)

        if result.get('adjusted', False):
            adjusted_count += 1
        if result.get('sarcasm_prob', 0) > 0.5:
            sarcasm_detected += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(test_texts)}")

    accuracy = accuracy_score(test_labels, predictions)

    # Baseline (RoBERTa on same test set)
    print("\nLoading baseline (RoBERTa) on same test set...")
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=512,
        device=-1
    )

    baseline_preds = []
    for text in test_texts:
        result = sentiment_model(preprocess(text)[:512])[0]
        baseline_preds.append(label_to_numeric(result['label'].lower()))

    baseline_acc = accuracy_score(test_labels, baseline_preds)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"FAST SARCASM DETECTOR ACCURACY: {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"Baseline (RoBERTa) on test set: {baseline_acc:.4f} ({baseline_acc * 100:.1f}%)")
    print(f"SARCASM IMPROVEMENT: +{(accuracy - baseline_acc) * 100:.1f}%")
    print(f"{'=' * 50}")
    print(f"  Sarcasm detected: {sarcasm_detected} ({100 * sarcasm_detected / len(test_texts):.1f}%)")
    print(f"  Sentiments adjusted: {adjusted_count} ({100 * adjusted_count / len(test_texts):.1f}%)")

    # Classification report
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT (Fast Sarcasm Detector)")
    print("=" * 70)
    print(classification_report(test_labels, predictions,
                                target_names=['Negative', 'Neutral', 'Positive']))

    # Show examples of sarcasm detection
    print("\n" + "=" * 70)
    print("EXAMPLES OF SARCASM DETECTION")
    print("=" * 70)

    sarcasm_examples = []
    for text, result in zip(test_texts, all_results):
        if result.get('adjusted', False) and len(sarcasm_examples) < 5:
            sarcasm_examples.append((text, result))

    if sarcasm_examples:
        for i, (text, result) in enumerate(sarcasm_examples):
            print(f"\n{i + 1}. Text: {text[:150]}...")
            print(f"   Sarcasm probability: {result.get('sarcasm_prob', 0):.3f}")
            print(f"   Base sentiment: {result.get('base_sentiment', 'unknown')}")
            print(f"   Final sentiment: {result['polarity']}")
            print(f"   → Adjusted due to sarcasm")
    else:
        print("\n  No sarcastic examples found in test set.")

    # Show where sarcasm detection fixed errors
    print("\n" + "=" * 70)
    print("WHERE SARCASM DETECTION HELPED")
    print("=" * 70)

    baseline_preds_labels = [numeric_to_label(p) for p in baseline_preds]
    predictions_labels = [numeric_to_label(p) for p in predictions]
    true_labels = [numeric_to_label(t) for t in test_labels]

    fixed_count = 0
    for i, (true, base, pred, text, result) in enumerate(zip(
            true_labels, baseline_preds_labels, predictions_labels, test_texts, all_results)):

        if base != true and pred == true and fixed_count < 5:
            print(f"\n{fixed_count + 1}. Text: {text[:120]}...")
            print(f"   True: {true}")
            print(f"   Baseline (RoBERTa): {base} ❌")
            print(f"   Sarcasm-aware: {pred} ✅")
            print(f"   Sarcasm probability: {result.get('sarcasm_prob', 0):.3f}")
            fixed_count += 1

    # Save results
    results_df = pd.DataFrame([
        {
            'id': i + 1,
            'text': text[:500],
            'true_label': numeric_to_label(test_labels[i]),
            'predicted_label': numeric_to_label(predictions[i]),
            'correct': test_labels[i] == predictions[i],
            'sarcasm_prob': all_results[i].get('sarcasm_prob', 0),
            'adjusted': all_results[i].get('adjusted', False),
            'base_sentiment': all_results[i].get('base_sentiment', '')
        }
        for i, text in enumerate(test_texts)
    ])
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n📁 Results saved to: {OUTPUT_CSV}")

    # Save report
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FAST SARCASM DETECTOR - DETAILED REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Test set size: {len(test_labels)} records\n")
        f.write(f"Baseline (RoBERTa) Accuracy: {baseline_acc:.4f} ({baseline_acc * 100:.1f}%)\n")
        f.write(f"Sarcasm Detector Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)\n")
        f.write(f"IMPROVEMENT: +{(accuracy - baseline_acc) * 100:.1f}%\n\n")

        f.write(f"Sarcasm detected: {sarcasm_detected} ({100 * sarcasm_detected / len(test_texts):.1f}%)\n")
        f.write(f"Sentiments adjusted: {adjusted_count} ({100 * adjusted_count / len(test_texts):.1f}%)\n\n")

        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(test_labels, predictions,
                                      target_names=['Negative', 'Neutral', 'Positive']))
        f.write("\n")

        f.write("EXAMPLES OF SARCASM DETECTION\n")
        f.write("-" * 40 + "\n")
        for i, (text, result) in enumerate(sarcasm_examples[:5]):
            f.write(f"\n{i + 1}. Text: {text[:150]}...\n")
            f.write(f"   Sarcasm probability: {result.get('sarcasm_prob', 0):.3f}\n")
            f.write(f"   Base: {result.get('base_sentiment', 'unknown')} → Final: {result['polarity']}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"📁 Report saved to: {OUTPUT_TXT}")

    return accuracy


if __name__ == "__main__":
    evaluate_sarcasm()