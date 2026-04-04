"""
hybrid_learning_fixed.py — Fixed learned hybrid classifier
No manual keywords - model learns patterns from data
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


class LearnedHybridClassifier:
    """
    Hybrid classifier that LEARNS patterns instead of using hardcoded keywords
    """

    def __init__(self, train_texts=None, train_labels=None):
        print("="*60)
        print("LOADING LEARNED HYBRID CLASSIFIER")
        print("="*60)

        # Neural component
        print("\nLoading neural model (RoBERTa)...")
        self.neural_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            truncation=True,
            max_length=512,
            device=-1
        )
        print("✓ Neural model loaded")

        # Learn symbolic patterns from training data if provided
        self.symbolic_model = None
        self.vectorizer = None

        if train_texts and train_labels:
            print("\nLearning symbolic patterns from training data...")
            self._train_symbolic_model(train_texts, train_labels)

    def _train_symbolic_model(self, texts, labels):
        """
        Train a model to learn which patterns indicate sentiment
        No manual keywords - the model learns from your data!
        """
        # Extract n-gram features (character patterns)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(2, 4),  # Character n-grams (2-4 chars)
            max_features=500,
            analyzer='char_wb',
            min_df=3
        )

        # Transform texts to feature vectors
        X = self.vectorizer.fit_transform(texts)
        y = labels

        # Train a simple classifier to learn pattern weights
        # Fixed: removed multi_class parameter (it's auto-detected)
        self.symbolic_model = LogisticRegression(max_iter=1000)
        self.symbolic_model.fit(X, y)

        # Get the most important patterns
        feature_names = self.vectorizer.get_feature_names_out()
        if hasattr(self.symbolic_model, 'coef_'):
            # For each class, get top patterns
            top_patterns = {}
            class_names = ['negative', 'neutral', 'positive']
            for i, class_name in enumerate(class_names):
                if i < len(self.symbolic_model.coef_):
                    coef = self.symbolic_model.coef_[i]
                    top_indices = np.argsort(coef)[-10:]
                    top_patterns[class_name] = [feature_names[idx] for idx in top_indices if coef[idx] > 0]

            print(f"  Learned patterns for 'positive': {top_patterns.get('positive', [])[:5]}")
            print(f"  Learned patterns for 'negative': {top_patterns.get('negative', [])[:5]}")

        print(f"✓ Symbolic model trained on {len(texts)} examples")

    def _get_learned_symbolic_score(self, text: str) -> np.ndarray:
        """Get symbolic prediction from learned model"""
        if self.symbolic_model is None:
            return None

        try:
            X = self.vectorizer.transform([text])
            probs = self.symbolic_model.predict_proba(X)[0]
            return probs
        except:
            return None

    def _detect_patterns(self, text: str) -> dict:
        """
        Detect structural patterns (minimal, just for obvious cases)
        """
        text_lower = text.lower()
        patterns = {}

        # Only these basic patterns (no keyword lists!)
        patterns['is_question'] = '?' in text
        patterns['has_contrast'] = 'but' in text_lower or 'however' in text_lower
        patterns['has_sarcasm_emoji'] = any(emoji in text for emoji in ['🙄', '😏', '😒', '😂', '💀'])

        return patterns

    def classify(self, text: str) -> dict:
        """Hybrid classification using learned patterns"""
        processed = preprocess(text)

        if len(processed) < 10:
            return {'polarity': 'neutral', 'method': 'short_text'}

        # Neural prediction
        neural_result = self.neural_model(processed)[0]
        neural_sentiment = neural_result['label'].lower()
        neural_confidence = neural_result['score']

        # Convert neural to probabilities
        neural_probs = np.zeros(3)
        if neural_sentiment == 'positive':
            neural_probs[2] = neural_confidence
            neural_probs[1] = (1 - neural_confidence) / 2
            neural_probs[0] = (1 - neural_confidence) / 2
        elif neural_sentiment == 'negative':
            neural_probs[0] = neural_confidence
            neural_probs[1] = (1 - neural_confidence) / 2
            neural_probs[2] = (1 - neural_confidence) / 2
        else:
            neural_probs[1] = neural_confidence
            neural_probs[0] = (1 - neural_confidence) / 2
            neural_probs[2] = (1 - neural_confidence) / 2

        # Get learned symbolic predictions
        symbolic_probs = self._get_learned_symbolic_score(processed)

        # Get pattern detections
        patterns = self._detect_patterns(text)

        # Combine predictions
        if symbolic_probs is not None:
            # Weight: 70% neural, 30% learned symbolic
            final_probs = 0.7 * neural_probs + 0.3 * symbolic_probs
        else:
            final_probs = neural_probs

        # Apply minimal structural rules (only for clear cases)
        if patterns['is_question'] and len(text.split()) > 3:
            final_probs = np.array([0.2, 0.7, 0.1])

        if patterns['has_sarcasm_emoji']:
            final_probs = np.array([final_probs[2], final_probs[1], final_probs[0]])

        # Get final prediction
        pred_class = np.argmax(final_probs)
        polarity = {0: 'negative', 1: 'neutral', 2: 'positive'}[pred_class]

        return {
            'polarity': polarity,
            'neural_sentiment': neural_sentiment,
            'neural_confidence': neural_confidence,
            'patterns': {k: v for k, v in patterns.items() if v},
            'method': 'learned_hybrid'
        }


def evaluate_hybrid():
    """Evaluate learned hybrid classifier"""

    print("="*80)
    print("LEARNED HYBRID CLASSIFIER EVALUATION")
    print("No manual keywords - model learns patterns from data")
    print("="*80)

    # Load data
    df = pd.read_excel(EVAL_FILE)
    df = df.dropna(subset=['text', 'sentiment_label'])
    texts = df['text'].astype(str).tolist()
    y_true = [label_to_numeric(l) for l in df['sentiment_label']]

    print(f"\nLoaded {len(texts)} records")
    print(f"Distribution: {df['sentiment_label'].value_counts().to_dict()}")

    # Split into train and test
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, y_true, test_size=0.2, random_state=42, stratify=y_true
    )
    print(f"\nTraining symbolic model on {len(train_texts)} samples")
    print(f"Testing on {len(test_texts)} samples")

    # Initialize classifier with training data
    classifier = LearnedHybridClassifier(train_texts, train_labels)

    # Test on unseen data
    print("\nRunning inference on test set...")
    predictions = []

    for i, text in enumerate(test_texts):
        result = classifier.classify(text)
        predictions.append(result['polarity'])
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_texts)}")

    # Calculate accuracy
    pred_numeric = [{'positive': 2, 'neutral': 1, 'negative': 0}[p] for p in predictions]
    accuracy = accuracy_score(test_labels, pred_numeric)

    print(f"\n{'='*50}")
    print(f"LEARNED HYBRID ACCURACY: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"{'='*50}")

    # Compare to baseline
    if os.path.exists(BASELINE_FILE):
        baseline_df = pd.read_csv(BASELINE_FILE)
        baseline_preds = baseline_df['polarity'].str.lower().str.strip().tolist()
        baseline_preds_numeric = [label_to_numeric(p) for p in baseline_preds]

        # Align with test set
        baseline_acc = accuracy_score(test_labels, [baseline_preds_numeric[i] for i in range(len(test_texts))])
        print(f"\nBaseline (RoBERTa only) on test set: {baseline_acc:.4f} ({baseline_acc*100:.1f}%)")
        print(f"IMPROVEMENT: +{(accuracy - baseline_acc)*100:.1f}%")

    # Classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(test_labels, pred_numeric,
                                target_names=['Negative', 'Neutral', 'Positive']))

    return accuracy


if __name__ == "__main__":
    evaluate_hybrid()