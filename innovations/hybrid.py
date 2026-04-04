"""
hybrid_classifier_train_test.py — Hybrid Classifier (Neural + Symbolic)
Trains on 80% of eval set, evaluates on 20% test set
Saves detailed results to CSV and TXT files
"""

import os
import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re
from datetime import datetime

BASE_DIR = r"C:\Users\Sutheerth\PycharmProjects\PythonProject2"
EVAL_FILE = os.path.join(BASE_DIR, "eval.xls")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    if label == 'positive': return 2
    elif label == 'neutral': return 1
    elif label == 'negative': return 0
    return 1

def numeric_to_label(num):
    return {0: 'negative', 1: 'neutral', 2: 'positive'}[num]


class HybridClassifier:
    """
    Hybrid classifier combining:
    1. Neural: RoBERTa for deep learning sentiment
    2. Symbolic: Learned patterns from training data
    3. Rule-based: Question detection, contrast detection
    """

    def __init__(self):
        print("="*60)
        print("LOADING HYBRID CLASSIFIER")
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

        # Symbolic component (will be trained on training data)
        self.symbolic_model = None
        self.vectorizer = None
        self.is_trained = False

        # Simple rule patterns (these are universal, not domain-specific)
        self.question_markers = ['?', 'what', 'why', 'how', 'does', 'is it', 'can you']
        self.contrast_markers = ['but', 'however', 'although', 'though', 'yet']
        self.positive_boost = ['love', 'excellent', 'amazing', 'perfect', 'best', 'wonderful']
        self.negative_boost = ['hate', 'terrible', 'awful', 'worst', 'useless', 'waste']

    def train_symbolic(self, texts, labels):
        """
        Train symbolic component on training data
        Learns which text patterns predict sentiment
        """
        print("\nTraining symbolic component...")

        # Convert labels to strings for vectorizer
        label_names = ['negative', 'neutral', 'positive']
        label_str = [label_names[l] for l in labels]

        # Learn character n-gram patterns from training data
        self.vectorizer = TfidfVectorizer(
            ngram_range=(2, 4),
            max_features=500,
            analyzer='char_wb',
            min_df=3
        )

        X = self.vectorizer.fit_transform(texts)
        self.symbolic_model = LogisticRegression(max_iter=1000)
        self.symbolic_model.fit(X, label_str)

        # Get top learned patterns
        feature_names = self.vectorizer.get_feature_names_out()
        if hasattr(self.symbolic_model, 'coef_'):
            top_patterns = {}
            for i, name in enumerate(label_names):
                if i < len(self.symbolic_model.coef_):
                    coef = self.symbolic_model.coef_[i]
                    top_indices = np.argsort(coef)[-10:]
                    top_patterns[name] = [feature_names[idx] for idx in top_indices if coef[idx] > 0]

            print(f"  Learned patterns for 'positive': {top_patterns.get('positive', [])[:5]}")
            print(f"  Learned patterns for 'negative': {top_patterns.get('negative', [])[:5]}")

        self.is_trained = True
        print(f"✓ Symbolic model trained on {len(texts)} examples")

    def _get_symbolic_prediction(self, text: str) -> tuple:
        """Get prediction from learned symbolic model"""
        if not self.is_trained:
            return 'neutral', 0.33

        try:
            X = self.vectorizer.transform([text])
            probs = self.symbolic_model.predict_proba(X)[0]
            classes = self.symbolic_model.classes_

            max_idx = np.argmax(probs)
            return classes[max_idx], probs[max_idx]
        except:
            return 'neutral', 0.33

    def _get_rule_based_signals(self, text: str) -> dict:
        """Extract rule-based signals"""
        text_lower = text.lower()

        signals = {
            'is_question': False,
            'has_contrast': False,
            'has_positive_boost': False,
            'has_negative_boost': False
        }

        if '?' in text:
            signals['is_question'] = True
        else:
            for marker in self.question_markers:
                if text_lower.startswith(marker):
                    signals['is_question'] = True
                    break

        signals['has_contrast'] = any(marker in text_lower for marker in self.contrast_markers)
        signals['has_positive_boost'] = any(word in text_lower for word in self.positive_boost)
        signals['has_negative_boost'] = any(word in text_lower for word in self.negative_boost)

        return signals

    def classify(self, text: str) -> dict:
        """Hybrid classification"""
        processed = preprocess(text)

        if len(processed) < 10:
            return {'polarity': 'neutral', 'method': 'short_text'}

        neural_result = self.neural_model(processed)[0]
        neural_sentiment = neural_result['label'].lower()
        neural_confidence = neural_result['score']

        symbolic_sentiment, symbolic_confidence = self._get_symbolic_prediction(processed)
        rules = self._get_rule_based_signals(processed)

        final_sentiment = neural_sentiment

        if rules['is_question']:
            return {'polarity': 'neutral', 'method': 'question_override', 'rules': rules}

        if rules['has_contrast']:
            if neural_sentiment != symbolic_sentiment:
                return {'polarity': 'neutral', 'method': 'mixed_sentiment', 'rules': rules}

        if rules['has_positive_boost'] and neural_confidence < 0.7:
            return {'polarity': 'positive', 'method': 'positive_boost', 'rules': rules}

        if rules['has_negative_boost'] and neural_confidence < 0.7:
            return {'polarity': 'negative', 'method': 'negative_boost', 'rules': rules}

        if neural_confidence < 0.55 and symbolic_confidence > 0.6:
            final_sentiment = symbolic_sentiment

        if neural_sentiment != symbolic_sentiment and symbolic_confidence > 0.7:
            final_sentiment = symbolic_sentiment

        return {
            'polarity': final_sentiment,
            'method': 'hybrid',
            'neural_sentiment': neural_sentiment,
            'neural_confidence': neural_confidence,
            'symbolic_sentiment': symbolic_sentiment,
            'symbolic_confidence': symbolic_confidence,
            'rules': rules
        }


def save_results_to_files(results_df, accuracy, baseline_acc, test_labels, predictions, all_results, test_texts):
    """Save detailed results to CSV and TXT files"""

    # CSV file with all predictions
    csv_path = os.path.join(OUTPUT_DIR, "hybrid_classifier_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n📁 CSV results saved to: {csv_path}")

    # TXT file with detailed breakdown
    txt_path = os.path.join(OUTPUT_DIR, "hybrid_classifier_report.txt")

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("HYBRID CLASSIFIER - DETAILED REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        # Summary
        f.write("SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Test set size: {len(test_labels)} records\n")
        f.write(f"Baseline (RoBERTa) Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.1f}%)\n")
        f.write(f"Hybrid Classifier Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)\n")
        f.write(f"IMPROVEMENT: +{(accuracy - baseline_acc)*100:.1f}%\n\n")

        # Classification Report
        f.write("CLASSIFICATION REPORT\n")
        f.write("-"*40 + "\n")
        f.write(classification_report(test_labels, predictions,
                                      target_names=['Negative', 'Neutral', 'Positive']))
        f.write("\n")

        # Confusion Matrix
        f.write("CONFUSION MATRIX\n")
        f.write("-"*40 + "\n")
        cm = confusion_matrix(test_labels, predictions)
        f.write("              Predicted\n")
        f.write("              Neg  Neu  Pos\n")
        f.write(f"Actual Neg    {cm[0,0]:3}   {cm[0,1]:3}   {cm[0,2]:3}\n")
        f.write(f"       Neu    {cm[1,0]:3}   {cm[1,1]:3}   {cm[1,2]:3}\n")
        f.write(f"       Pos    {cm[2,0]:3}   {cm[2,1]:3}   {cm[2,2]:3}\n\n")

        # Method Breakdown
        f.write("METHOD BREAKDOWN\n")
        f.write("-"*40 + "\n")
        method_counts = {}
        for result in all_results:
            method = result.get('method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1

        for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {method}: {count} ({100*count/len(test_texts):.1f}%)\n")
        f.write("\n")

        # Learned Patterns
        f.write("LEARNED SYMBOLIC PATTERNS\n")
        f.write("-"*40 + "\n")
        f.write("These patterns were automatically learned from training data:\n\n")

        # Show examples where each rule fired
        f.write("EXAMPLE CORRECTIONS BY METHOD\n")
        f.write("-"*40 + "\n")

        method_examples = {}
        for text, result in zip(test_texts, all_results):
            method = result.get('method', 'unknown')
            if method not in method_examples and method != 'hybrid':
                method_examples[method] = (text, result)

        for method, (text, result) in method_examples.items():
            f.write(f"\n[{method.upper()}]\n")
            f.write(f"  Text: {text[:150]}...\n")
            f.write(f"  Prediction: {result['polarity']}\n")
            if 'neural_sentiment' in result:
                f.write(f"  Neural (RoBERTa) said: {result['neural_sentiment']} (conf: {result['neural_confidence']:.2f})\n")
            if 'symbolic_sentiment' in result:
                f.write(f"  Symbolic model said: {result['symbolic_sentiment']} (conf: {result['symbolic_confidence']:.2f})\n")
            if 'rules' in result:
                active_rules = [k for k, v in result['rules'].items() if v]
                if active_rules:
                    f.write(f"  Active rules: {active_rules}\n")

        # Error Analysis
        f.write("\n" + "="*80 + "\n")
        f.write("ERROR ANALYSIS\n")
        f.write("="*80 + "\n")

        # Find misclassifications
        misclassified = []
        for i, (true, pred, text, result) in enumerate(zip(test_labels, predictions, test_texts, all_results)):
            if true != pred:
                misclassified.append((true, pred, text, result))

        f.write(f"\nTotal misclassifications: {len(misclassified)}/{len(test_labels)} ({100*len(misclassified)/len(test_labels):.1f}%)\n\n")

        # Show first 10 misclassifications
        f.write("FIRST 10 MISCLASSIFICATIONS:\n")
        f.write("-"*40 + "\n")
        for i, (true, pred, text, result) in enumerate(misclassified[:10]):
            true_label = numeric_to_label(true)
            pred_label = numeric_to_label(pred)
            f.write(f"\n{i+1}. True: {true_label} | Predicted: {pred_label}\n")
            f.write(f"   Text: {text[:150]}...\n")
            f.write(f"   Method: {result.get('method', 'unknown')}\n")
            if 'neural_sentiment' in result:
                f.write(f"   Neural said: {result['neural_sentiment']} (conf: {result['neural_confidence']:.2f})\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"📁 TXT report saved to: {txt_path}")


def evaluate_hybrid():
    """Evaluate hybrid classifier on eval set (80/20 split)"""

    print("="*70)
    print("HYBRID CLASSIFIER EVALUATION")
    print("80% Train / 20% Test (same methodology as ensemble)")
    print("="*70)

    # Load data
    df = pd.read_excel(EVAL_FILE)
    df = df.dropna(subset=['text', 'sentiment_label'])
    texts = df['text'].astype(str).tolist()
    y_true = [label_to_numeric(l) for l in df['sentiment_label']]

    print(f"\nLoaded {len(texts)} records")
    print(f"Distribution: {df['sentiment_label'].value_counts().to_dict()}")

    # Split train/test (80/20)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, y_true, test_size=0.2, random_state=RANDOM_SEED, stratify=y_true
    )
    print(f"\nTraining on {len(train_texts)} samples")
    print(f"Testing on {len(test_texts)} samples")

    # Initialize and train hybrid classifier
    hybrid = HybridClassifier()
    hybrid.train_symbolic(train_texts, train_labels)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions = []
    all_results = []

    for i, text in enumerate(test_texts):
        result = hybrid.classify(text)
        predictions.append(label_to_numeric(result['polarity']))
        all_results.append(result)
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
    print(f"HYBRID CLASSIFIER ACCURACY: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Baseline (RoBERTa) on test set: {baseline_acc:.4f} ({baseline_acc*100:.1f}%)")
    print(f"HYBRID IMPROVEMENT: +{(accuracy - baseline_acc)*100:.1f}%")
    print(f"{'='*50}")

    # Create results DataFrame
    results_df = pd.DataFrame([
        {
            'text': text[:500],
            'true_label': numeric_to_label(test_labels[i]),
            'predicted_label': numeric_to_label(predictions[i]),
            'correct': test_labels[i] == predictions[i],
            'method': all_results[i].get('method', 'unknown'),
            'neural_sentiment': all_results[i].get('neural_sentiment', ''),
            'neural_confidence': all_results[i].get('neural_confidence', 0),
            'symbolic_sentiment': all_results[i].get('symbolic_sentiment', ''),
            'symbolic_confidence': all_results[i].get('symbolic_confidence', 0),
            'is_question': all_results[i].get('rules', {}).get('is_question', False),
            'has_contrast': all_results[i].get('rules', {}).get('has_contrast', False),
            'has_positive_boost': all_results[i].get('rules', {}).get('has_positive_boost', False),
            'has_negative_boost': all_results[i].get('rules', {}).get('has_negative_boost', False)
        }
        for i in range(len(test_texts))
    ])

    # Save results
    save_results_to_files(results_df, accuracy, baseline_acc, test_labels, predictions, all_results, test_texts)

    # Print method breakdown
    print("\n" + "="*70)
    print("METHOD BREAKDOWN")
    print("="*70)
    method_counts = {}
    for result in all_results:
        method = result.get('method', 'unknown')
        method_counts[method] = method_counts.get(method, 0) + 1

    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"  {method}: {count} ({100*count/len(test_texts):.1f}%)")

    return accuracy


if __name__ == "__main__":
    evaluate_hybrid()