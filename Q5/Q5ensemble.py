"""
pip install vaderSentiment textblob transformers scikit-learn pandas
python -m textblob.download_corpora
"""

import pandas as pd
import re
from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter


class EnsembleClassifier:
    def __init__(self):
        print("Loading ensemble models...")
        # Model 1: RoBERTa (your original)
        self.roberta = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            truncation=True,
            max_length=512,
            device=-1
        )

        # Model 2: VADER (good for social media)
        self.vader = SentimentIntensityAnalyzer()

        print("Ensemble models loaded.")

    def preprocess(self, text: str) -> str:
        if not text or pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:512]

    def get_roberta_sentiment(self, text: str) -> tuple:
        """Get RoBERTa sentiment (positive/negative/neutral)"""
        try:
            if len(text) < 10:
                return 'neutral', 0.5
            result = self.roberta(text)[0]
            return result['label'].lower(), result['score']
        except Exception as e:
            return 'neutral', 0.5

    def get_vader_sentiment(self, text: str) -> tuple:
        """Get VADER sentiment (compound score -> label)"""
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']

        if compound >= 0.05:
            return 'positive', abs(compound)
        elif compound <= -0.05:
            return 'negative', abs(compound)
        else:
            return 'neutral', 1 - abs(compound)

    def get_textblob_sentiment(self, text: str) -> tuple:
        """Get TextBlob sentiment"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity > 0.1:
            return 'positive', polarity
        elif polarity < -0.1:
            return 'negative', abs(polarity)
        else:
            return 'neutral', 1 - abs(polarity)

    def rule_based_fixes(self, text: str, ensemble_vote: str) -> tuple:
        """Apply rule-based overrides to fix common errors"""
        text_lower = text.lower()

        # Fix 1: Questions should be neutral
        if '?' in text and len(text.split()) > 3:
            return 'neutral', 'question_override'

        # Fix 2: Off-topic (betting/sports)
        off_topic = [r'bets?', r'odds?', r'ML', r'ADD', r'CBB', r'🏀', r'⚽']
        if any(re.search(p, text_lower) for p in off_topic):
            return 'neutral', 'offtopic_override'

        # Fix 3: Hashtag-only posts
        words = text.split()
        if words and sum(1 for w in words if w.startswith('#')) / len(words) > 0.5:
            return 'neutral', 'hashtag_override'

        # Fix 4: Explicit positive indicators
        strong_positive = [r'i love', r'i enjoy', r'this is great', r'excellent', r'amazing', r'perfect']
        for pattern in strong_positive:
            if re.search(pattern, text_lower):
                return 'positive', 'positive_override'

        # Fix 5: Explicit negative indicators
        strong_negative = [r'i hate', r'terrible', r'awful', r'useless', r'waste of', r'cooked']
        for pattern in strong_negative:
            if re.search(pattern, text_lower):
                return 'negative', 'negative_override'

        return ensemble_vote, 'no_override'

    def ensemble_vote(self, predictions: list) -> str:
        """Weighted voting for ensemble predictions"""
        weights = {'positive': 0, 'negative': 0, 'neutral': 0}

        # RoBERTa (highest weight - best model)
        roberta_label, roberta_score = predictions[0]
        weights[roberta_label] += 0.5 * roberta_score

        # VADER (good for social media)
        vader_label, vader_score = predictions[1]
        weights[vader_label] += 0.3 * vader_score

        # TextBlob (baseline)
        blob_label, blob_score = predictions[2]
        weights[blob_label] += 0.2 * blob_score

        # Get highest weighted label
        return max(weights, key=weights.get)

    def classify(self, text: str) -> dict:
        """Full ensemble classification with rule overrides"""
        processed = self.preprocess(text)

        # Get predictions from all models
        roberta_label, roberta_score = self.get_roberta_sentiment(processed)
        vader_label, vader_score = self.get_vader_sentiment(processed)
        blob_label, blob_score = self.get_textblob_sentiment(processed)

        predictions = [
            (roberta_label, roberta_score),
            (vader_label, vader_score),
            (blob_label, blob_score)
        ]

        # Ensemble voting
        ensemble_label = self.ensemble_vote(predictions)

        # Apply rule-based overrides
        final_label, override = self.rule_based_fixes(text, ensemble_label)

        return {
            'polarity': final_label,  # Main output for ablation study
            'roberta': roberta_label,
            'vader': vader_label,
            'textblob': blob_label,
            'ensemble': ensemble_label,
            'override': override
        }


def evaluate_ensemble():
    """Test ensemble classifier on eval.xls"""

    BASE_DIR = r"C:\Users\Sutheerth\PycharmProjects\PythonProject2"
    EVAL_FILE = f"{BASE_DIR}/eval.xls"

    # Load data
    df = pd.read_excel(EVAL_FILE)
    y_true = df['sentiment_label'].str.lower().str.strip().tolist()
    texts = df['text'].tolist()

    print("="*70)
    print("ENSEMBLE CLASSIFIER EVALUATION")
    print("="*70)
    print(f"Total samples: {len(texts)}")
    print(f"Distribution: {pd.Series(y_true).value_counts().to_dict()}")

    # Run ensemble
    classifier = EnsembleClassifier()

    results = {
        'roberta': [],
        'vader': [],
        'textblob': [],
        'ensemble': [],
        'final': [],
        'overrides': []
    }

    for i, text in enumerate(texts):
        result = classifier.classify(text)
        results['roberta'].append(result['roberta'])
        results['vader'].append(result['vader'])
        results['textblob'].append(result['textblob'])
        results['ensemble'].append(result['ensemble'])
        results['final'].append(result['polarity'])
        results['overrides'].append(result['override'])

        if (i+1) % 200 == 0:
            print(f"  Processed {i+1}/{len(texts)}")

    # Calculate accuracies
    from sklearn.metrics import accuracy_score, classification_report

    print("\n" + "="*70)
    print("ACCURACY COMPARISON")
    print("="*70)

    roberta_acc = accuracy_score(y_true, results['roberta'])
    vader_acc = accuracy_score(y_true, results['vader'])
    blob_acc = accuracy_score(y_true, results['textblob'])
    ensemble_acc = accuracy_score(y_true, results['ensemble'])
    final_acc = accuracy_score(y_true, results['final'])

    print(f"\n{'Model':<15} {'Accuracy':<12} {'Improvement':<12}")
    print("-"*40)
    print(f"{'RoBERTa (baseline)':<15} {roberta_acc:.4f}       {'—':<12}")
    print(f"{'VADER':<15} {vader_acc:.4f}       {vader_acc - roberta_acc:+.4f}")
    print(f"{'TextBlob':<15} {blob_acc:.4f}       {blob_acc - roberta_acc:+.4f}")
    print(f"{'Ensemble (vote)':<15} {ensemble_acc:.4f}       {ensemble_acc - roberta_acc:+.4f}")
    print(f"{'Ensemble + Rules':<15} {final_acc:.4f}       {final_acc - roberta_acc:+.4f}")

    print("\n" + "="*70)
    print("FINAL CLASSIFICATION REPORT (Ensemble + Rules)")
    print("="*70)
    print(classification_report(y_true, results['final'],
                                labels=['positive', 'negative', 'neutral'],
                                target_names=['Positive', 'Negative', 'Neutral']))

    # Show override statistics
    print("\n" + "="*70)
    print("RULE OVERRIDES")
    print("="*70)
    override_counts = Counter(results['overrides'])
    for override, count in override_counts.most_common():
        if override != 'no_override':
            print(f"  {override}: {count} ({100*count/len(texts):.1f}%)")

    # Show examples where ensemble corrected errors
    print("\n" + "="*70)
    print("EXAMPLE CORRECTIONS (Ensemble fixed what RoBERTa got wrong)")
    print("="*70)

    corrections_shown = 0
    for i, (true, roberta_pred, final_pred, text) in enumerate(zip(y_true, results['roberta'], results['final'], texts)):
        if roberta_pred != true and final_pred == true and corrections_shown < 5:
            print(f"\nText: {str(text)[:150]}...")
            print(f"  True: {true}")
            print(f"  RoBERTa: {roberta_pred} ❌")
            print(f"  Ensemble+Rules: {final_pred} ✅")
            print(f"  Override: {results['overrides'][i]}")
            corrections_shown += 1

    return final_acc


if __name__ == "__main__":
    evaluate_ensemble()