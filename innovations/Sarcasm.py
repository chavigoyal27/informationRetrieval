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
    text = str(text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:512]

def label_to_numeric(label):
    label = str(label).lower().strip()
    if label == 'positive': return 2
    elif label == 'neutral': return 1
    elif label == 'negative': return 0
    return 1


class LearnedSarcasmDetector:
    """
    Sarcasm detection that LEARNS patterns from data
    No manual keywords - uses TF-IDF + logistic regression
    """

    def __init__(self):
        print("="*60)
        print("LOADING LEARNED SARCASM DETECTOR")
        print("="*60)

        # Load sentiment model (for base sentiment)
        print("\nLoading sentiment model...")
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            truncation=True,
            max_length=512,
            device=-1
        )
        print("✓ Sentiment model loaded")

        # Models for learning sarcasm patterns
        self.sarcasm_model = None
        self.vectorizer = None

    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract features that might indicate sarcasm
        These are structural features, not keyword-based
        """
        text_lower = text.lower()
        features = []

        # Feature 1: Length of text
        features.append(min(len(text) / 500, 1.0))

        # Feature 2: Number of sentences
        sentences = re.split(r'[.!?;]+', text)
        features.append(min(len(sentences) / 10, 1.0))

        # Feature 3: Exclamation marks count
        features.append(min(text.count('!') / 5, 1.0))

        # Feature 4: Question marks count
        features.append(min(text.count('?') / 5, 1.0))

        # Feature 5: Capitalized words ratio (potential emphasis)
        words = text.split()
        if words:
            caps_count = sum(1 for w in words if w.isupper() and len(w) > 1)
            features.append(min(caps_count / len(words), 1.0))
        else:
            features.append(0)

        # Feature 6: Ellipsis count (hesitation/sarcasm)
        features.append(min(text.count('...') / 3, 1.0))

        # Feature 7: Sentiment confidence (low confidence might indicate sarcasm)
        try:
            sent_result = self.sentiment_model(text[:512])[0]
            sent_conf = sent_result['score']
            features.append(1 - sent_conf)  # Low confidence = higher sarcasm chance
        except:
            features.append(0.5)

        # Feature 8: Sentiment polarity (sarcasm often inverts expected polarity)
        try:
            sent_result = self.sentiment_model(text[:512])[0]
            sent = sent_result['label'].lower()
            polarity = 1 if sent == 'positive' else (-1 if sent == 'negative' else 0)
            features.append((polarity + 1) / 2)  # Normalize to 0-1
        except:
            features.append(0.5)

        return np.array(features)

    def extract_text_features(self, texts):
        """Extract TF-IDF features from text (learns patterns automatically)"""
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(2, 4),  # Character n-grams
                max_features=200,
                analyzer='char_wb',
                min_df=2
            )
            X = self.vectorizer.fit_transform(texts)
        else:
            X = self.vectorizer.transform(texts)
        return X

    def train_sarcasm_model(self, texts, labels):
        """
        Train a model to detect sarcasm patterns
        Uses both structural features and learned text patterns
        """
        print("\nLearning sarcasm patterns from training data...")

        # First, we need to identify which texts are sarcastic
        # Since eval.xls doesn't have sarcasm labels, we'll use a heuristic:
        # Texts where RoBERTa prediction is different from ground truth might indicate sarcasm
        # This is a form of weak supervision

        print("  Generating pseudo-sarcasm labels from data...")

        # Get RoBERTa predictions
        roberta_preds = []
        for text in texts:
            try:
                result = self.sentiment_model(text[:512])[0]
                roberta_preds.append(result['label'].lower())
            except:
                roberta_preds.append('neutral')

        # Ground truth labels (from eval.xls)
        label_names = {0: 'negative', 1: 'neutral', 2: 'positive'}
        true_labels = [label_names[l] for l in labels]

        # A text might be sarcastic if RoBERTa got it wrong
        # (especially if it predicted positive but true is negative)
        sarcasm_labels = []
        for roberta, true in zip(roberta_preds, true_labels):
            # Strong signal: positive prediction but negative ground truth
            if roberta == 'positive' and true == 'negative':
                sarcasm_labels.append(1)  # likely sarcastic
            elif roberta == 'negative' and true == 'positive':
                sarcasm_labels.append(1)  # likely sarcastic
            else:
                sarcasm_labels.append(0)  # not sarcastic

        print(f"  Detected potential sarcasm: {sum(sarcasm_labels)}/{len(texts)} ({100*sum(sarcasm_labels)/len(texts):.1f}%)")

        # Extract structural features
        print("  Extracting structural features...")
        structural_features = np.array([self.extract_features(text) for text in texts])

        # Extract text patterns using TF-IDF
        print("  Extracting text patterns...")
        text_features = self.extract_text_features(texts).toarray()

        # Combine features
        X = np.hstack([structural_features, text_features])
        y = sarcasm_labels

        # Train classifier
        print("  Training classifier...")
        self.sarcasm_model = LogisticRegression(max_iter=1000, class_weight='balanced')
        self.sarcasm_model.fit(X, y)

        print(f"✓ Sarcasm model trained on {len(texts)} examples")

    def detect_sarcasm(self, text: str) -> float:
        """Detect sarcasm probability"""
        if self.sarcasm_model is None:
            return 0.0

        structural = self.extract_features(text).reshape(1, -1)

        try:
            text_features = self.vectorizer.transform([text]).toarray()
            X = np.hstack([structural, text_features])
            prob = self.sarcasm_model.predict_proba(X)[0][1]
            return prob
        except:
            return 0.0

    def classify(self, text: str) -> dict:
        """Classify with sarcasm-aware adjustment"""
        processed = preprocess(text)

        if len(processed) < 10:
            return {'polarity': 'neutral', 'sarcasm_prob': 0.0}

        # Get base sentiment
        sent_result = self.sentiment_model(processed[:512])[0]
        base_sentiment = sent_result['label'].lower()
        base_confidence = sent_result['score']

        # Detect sarcasm
        sarcasm_prob = self.detect_sarcasm(text)

        # Adjust sentiment if sarcastic
        if sarcasm_prob > 0.6:
            # Invert sentiment
            invert_map = {'positive': 'negative', 'negative': 'positive', 'neutral': 'neutral'}
            final_sentiment = invert_map.get(base_sentiment, 'neutral')
            return {
                'polarity': final_sentiment,
                'sarcasm_prob': sarcasm_prob,
                'base_sentiment': base_sentiment,
                'adjusted': True
            }

        return {
            'polarity': base_sentiment,
            'sarcasm_prob': sarcasm_prob,
            'base_sentiment': base_sentiment,
            'adjusted': False
        }


def evaluate_sarcasm():
    """Evaluate learned sarcasm detection"""

    print("="*70)
    print("LEARNED SARCASM DETECTION EVALUATION")
    print("No manual keywords - model learns sarcasm patterns from data")
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

    # Train sarcasm detector
    detector = LearnedSarcasmDetector()
    detector.train_sarcasm_model(train_texts, train_labels)

    # Test
    print("\nRunning inference on test set...")
    predictions = []
    all_results = []
    sarcastic_count = 0

    for i, text in enumerate(test_texts):
        result = detector.classify(text)
        predictions.append(label_to_numeric(result['polarity']))
        all_results.append(result)
        if result['sarcasm_prob'] > 0.5:
            sarcastic_count += 1
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(test_texts)}")

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)

    print(f"\n{'='*50}")
    print(f"LEARNED SARCASM ACCURACY: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Texts flagged as sarcastic: {sarcastic_count} ({100*sarcastic_count/len(test_texts):.1f}%)")
    print(f"{'='*50}")

    # Compare to baseline
    if os.path.exists(BASELINE_FILE):
        baseline_df = pd.read_csv(BASELINE_FILE)
        baseline_preds = baseline_df['polarity'].str.lower().str.strip().tolist()
        baseline_preds_numeric = [label_to_numeric(p) for p in baseline_preds]

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

    # Show examples of sarcasm detection
    print("\n" + "="*70)
    print("EXAMPLES OF SARCASM DETECTION")
    print("="*70)

    # Find texts with high sarcasm probability
    sarcasm_examples = []
    for i, (text, result) in enumerate(zip(test_texts, all_results)):
        if result['sarcasm_prob'] > 0.6 and len(sarcasm_examples) < 5:
            sarcasm_examples.append((text, result))

    if sarcasm_examples:
        for i, (text, result) in enumerate(sarcasm_examples):
            print(f"\n{'─'*50}")
            print(f"Example {i+1}:")
            print(f"Text: {text[:200]}...")
            print(f"  Sarcasm probability: {result['sarcasm_prob']:.3f}")
            print(f"  Base sentiment: {result['base_sentiment']}")
            print(f"  Final sentiment: {result['polarity']}")
            if result['adjusted']:
                print(f"  → Adjusted due to sarcasm detection")
    else:
        print("\n  No high-confidence sarcasm examples found in test set.")
        print("  Looking in full dataset...")

        for text in texts[:100]:
            result = detector.classify(text)
            if result['sarcasm_prob'] > 0.6 and len(sarcasm_examples) < 5:
                sarcasm_examples.append((text, result))

        for i, (text, result) in enumerate(sarcasm_examples):
            print(f"\n{'─'*50}")
            print(f"Example {i+1}:")
            print(f"Text: {text[:200]}...")
            print(f"  Sarcasm probability: {result['sarcasm_prob']:.3f}")
            print(f"  Base sentiment: {result['base_sentiment']}")
            print(f"  Final sentiment: {result['polarity']}")

    # Show where sarcasm detection fixed errors
    print("\n" + "="*70)
    print("EXAMPLES WHERE SARCASM DETECTION FIXED ERRORS")
    print("="*70)

    if os.path.exists(BASELINE_FILE):
        fixed_count = 0
        for i, (true, baseline_pred, sarcasm_pred, text, result) in enumerate(zip(
            test_labels, baseline_test_preds, predictions, test_texts, all_results)):

            true_label = {0: 'negative', 1: 'neutral', 2: 'positive'}[true]
            baseline_label = {0: 'negative', 1: 'neutral', 2: 'positive'}[baseline_pred]
            sarcasm_label = {0: 'negative', 1: 'neutral', 2: 'positive'}[sarcasm_pred]

            if baseline_label != true_label and sarcasm_label == true_label and fixed_count < 5:
                print(f"\n{'─'*50}")
                print(f"Example {fixed_count + 1}:")
                print(f"Text: {text[:150]}...")
                print(f"  True label: {true_label}")
                print(f"  Baseline (RoBERTa): {baseline_label} ❌")
                print(f"  Sarcasm-aware: {sarcasm_label} ✅")
                print(f"  Sarcasm probability: {result['sarcasm_prob']:.3f}")
                fixed_count += 1

        if fixed_count == 0:
            print("\n  No sarcasm-related error fixes found in test set.")

    return accuracy


if __name__ == "__main__":
    evaluate_sarcasm()