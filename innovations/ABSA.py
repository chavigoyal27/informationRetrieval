import os
import sys
import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report
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
OUTPUT_CSV = os.path.join(ANALYSIS_DIR, "absa_fast_results.csv")
OUTPUT_TXT = os.path.join(ANALYSIS_DIR, "absa_fast_report.txt")

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


class FastABSAClassifier:
    """
    Fast Aspect-Based Sentiment Analysis
    Uses sentence splitting + pattern learning (no slow zero-shot)
    """

    def __init__(self):
        print("="*60)
        print("LOADING FAST ABSA CLASSIFIER")
        print("="*60)

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

        # Trainable components
        self.aspect_pattern_model = None
        self.vectorizer = None
        self.is_trained = False

    def split_sentences(self, text: str) -> list:
        """Split text into sentences"""
        sentences = re.split(r'[.!?;]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 15]

    def extract_sentence_features(self, text: str) -> np.ndarray:
        """
        Extract features for aspect detection without keywords
        Uses sentence-level sentiment patterns
        """
        sentences = self.split_sentences(text)

        if len(sentences) <= 1:
            # Single sentence: use overall sentiment
            result = self.sentiment_model(text[:512])[0]
            sent = result['label'].lower()
            sent_val = 1 if sent == 'positive' else (-1 if sent == 'negative' else 0)
            return np.array([1, sent_val, 0, 0, 0])

        # Multiple sentences: get sentiment for each
        sentiments = []
        for sent in sentences[:5]:  # Limit to 5 sentences
            try:
                result = self.sentiment_model(sent[:512])[0]
                sentiments.append(result['label'].lower())
            except:
                continue

        if not sentiments:
            return np.array([0, 0, 0, 0, 0])

        # Extract features
        has_positive = 1 if 'positive' in sentiments else 0
        has_negative = 1 if 'negative' in sentiments else 0
        is_mixed = 1 if has_positive and has_negative else 0
        num_sentences = len(sentiments)

        # Check for contrast words (indicates aspect switching)
        has_contrast = 1 if 'but' in text.lower() or 'however' in text.lower() else 0

        return np.array([0, is_mixed, has_contrast, num_sentences, has_positive + has_negative])

    def train_aspect_detection(self, texts, labels):
        """
        Train model to detect when multiple sentences have different sentiments
        This is the key: learns that different sentences = different aspects
        """
        print("\nTraining aspect detection model...")

        # Extract features for all training texts
        X_features = []
        for text in texts:
            features = self.extract_sentence_features(text)
            X_features.append(features)

        X = np.array(X_features)
        y = labels

        # Train classifier to predict sentiment from sentence patterns
        self.aspect_pattern_model = LogisticRegression(max_iter=1000)
        self.aspect_pattern_model.fit(X, y)

        print(f"  Learned coefficients: {self.aspect_pattern_model.coef_}")
        self.is_trained = True
        print(f"✓ Aspect detection trained on {len(texts)} examples")

    def classify(self, text: str) -> dict:
        """Fast ABSA classification"""
        processed = preprocess(text)

        if len(processed) < 10:
            return {'polarity': 'neutral', 'method': 'short_text', 'aspects': []}

        # Get base sentiment (fallback)
        base_result = self.sentiment_model(processed[:512])[0]
        base_sentiment = base_result['label'].lower()

        # Split into sentences for aspect analysis
        sentences = self.split_sentences(processed)

        if len(sentences) <= 1:
            # Single sentence - no aspect mixing possible
            return {
                'polarity': base_sentiment,
                'method': 'single_sentence',
                'aspects': []
            }

        # Get sentiment for each sentence
        sentence_sentiments = []
        for sent in sentences[:5]:
            try:
                result = self.sentiment_model(sent[:512])[0]
                sentence_sentiments.append(result['label'].lower())
            except:
                continue

        if not sentence_sentiments:
            return {'polarity': base_sentiment, 'method': 'no_sentences', 'aspects': []}

        # KEY ABSA LOGIC: Different sentences with different sentiments = mixed aspects
        has_positive = 'positive' in sentence_sentiments
        has_negative = 'negative' in sentence_sentiments

        # Check for contrast words (strong signal for aspect switching)
        has_contrast = 'but' in processed.lower() or 'however' in processed.lower()

        if has_positive and has_negative:
            # Mixed sentiment across sentences = neutral
            return {
                'polarity': 'neutral',
                'method': 'mixed_aspects',
                'aspects': ['positive_sentence', 'negative_sentence'],
                'sentence_sentiments': sentence_sentiments
            }
        elif has_contrast and has_positive:
            # "but" with positive may indicate mixed
            return {
                'polarity': 'neutral',
                'method': 'contrast_mixed',
                'aspects': ['contrast_pattern'],
                'sentence_sentiments': sentence_sentiments
            }
        elif has_positive:
            return {
                'polarity': 'positive',
                'method': 'unanimous_positive',
                'aspects': [],
                'sentence_sentiments': sentence_sentiments
            }
        elif has_negative:
            return {
                'polarity': 'negative',
                'method': 'unanimous_negative',
                'aspects': [],
                'sentence_sentiments': sentence_sentiments
            }
        else:
            return {
                'polarity': base_sentiment,
                'method': 'fallback',
                'aspects': [],
                'sentence_sentiments': sentence_sentiments
            }


def evaluate_absa():
    """Evaluate fast ABSA on eval set (80/20 split)"""

    print("="*70)
    print("FAST ABSA (ASPECT-BASED SENTIMENT ANALYSIS)")
    print("80% Train / 20% Test")
    print("="*70)

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

    # Initialize and train ABSA
    absa = FastABSAClassifier()
    absa.train_aspect_detection(train_texts, train_labels)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions = []
    all_results = []
    method_counts = {}

    for i, text in enumerate(test_texts):
        result = absa.classify(text)
        predictions.append(label_to_numeric(result['polarity']))
        all_results.append(result)

        method = result.get('method', 'unknown')
        method_counts[method] = method_counts.get(method, 0) + 1

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
    print(f"\n{'='*50}")
    print(f"FAST ABSA ACCURACY: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Baseline (RoBERTa) on test set: {baseline_acc:.4f} ({baseline_acc*100:.1f}%)")
    print(f"ABSA IMPROVEMENT: +{(accuracy - baseline_acc)*100:.1f}%")
    print(f"{'='*50}")

    # Method breakdown
    print("\n" + "="*70)
    print("METHOD BREAKDOWN")
    print("="*70)
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"  {method}: {count} ({100*count/len(test_texts):.1f}%)")

    # Classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT (Fast ABSA)")
    print("="*70)
    print(classification_report(test_labels, predictions,
                                target_names=['Negative', 'Neutral', 'Positive']))

    # Show examples of mixed aspect detection
    print("\n" + "="*70)
    print("EXAMPLES OF MIXED ASPECT DETECTION")
    print("="*70)

    mixed_examples = []
    for text, result in zip(test_texts, all_results):
        if result.get('method') in ['mixed_aspects', 'contrast_mixed'] and len(mixed_examples) < 5:
            mixed_examples.append((text, result))

    if mixed_examples:
        for i, (text, result) in enumerate(mixed_examples):
            print(f"\n{i+1}. Text: {text[:150]}...")
            print(f"   Sentiment: {result['polarity']} (mixed aspects → neutral)")
            print(f"   Sentence sentiments: {result.get('sentence_sentiments', [])}")
            print(f"   Method: {result['method']}")
    else:
        print("\n  No mixed aspect examples found in test set.")

    # Save results
    results_df = pd.DataFrame([
        {
            'id': i + 1,
            'text': text[:500],
            'true_label': numeric_to_label(test_labels[i]),
            'predicted_label': numeric_to_label(predictions[i]),
            'correct': test_labels[i] == predictions[i],
            'method': all_results[i].get('method', 'unknown'),
            'sentence_sentiments': str(all_results[i].get('sentence_sentiments', []))
        }
        for i, text in enumerate(test_texts)
    ])
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n📁 Results saved to: {OUTPUT_CSV}")

    # Save report
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("FAST ABSA - DETAILED REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        f.write("SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Test set size: {len(test_labels)} records\n")
        f.write(f"Baseline (RoBERTa) Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.1f}%)\n")
        f.write(f"Fast ABSA Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)\n")
        f.write(f"IMPROVEMENT: +{(accuracy - baseline_acc)*100:.1f}%\n\n")

        f.write("METHOD BREAKDOWN\n")
        f.write("-"*40 + "\n")
        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {method}: {count} ({100*count/len(test_texts):.1f}%)\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"📁 Report saved to: {OUTPUT_TXT}")

    return accuracy


if __name__ == "__main__":
    evaluate_absa()