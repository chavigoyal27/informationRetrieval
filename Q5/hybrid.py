"""
hybrid_classifier.py — Hybrid classification with proper token handling
"""

import re
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report


class HybridClassifier:
    """
    Hybrid approach combining:
    - Neural network (RoBERTa)
    - Symbolic rules (pattern-based)
    - Knowledge base (education-specific lexicon)
    """

    def __init__(self):
        # Neural network component
        print("Loading neural model...")
        self.neural_model = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            truncation=True,  # Truncate long texts
            max_length=512,   # Maximum token length
            device=-1
        )

        # Knowledge base: Education-specific word lists
        self.education_lexicon = {
            'positive': ['engaging', 'insightful', 'thought-provoking', 'well-structured', 'comprehensive',
                        'helpful', 'useful', 'clear', 'effective', 'valuable'],
            'negative': ['boring', 'outdated', 'confusing', 'irrelevant', 'overwhelming', 'useless',
                        'waste', 'terrible', 'awful', 'difficult', 'hard'],
            'neutral_indicators': ['according to', 'refers to', 'defines', 'states that', 'the research shows',
                                  'in summary', 'for example', 'such as']
        }

    def preprocess(self, text: str) -> str:
        """Basic text preprocessing - safe length"""
        if not text or pd.isna(text):
            return ""
        text = str(text)
        # Don't lowercase here - keep original for emoji detection
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Limit to ~4000 characters to be safe (will be tokenized to 512 max)
        if len(text) > 4000:
            text = text[:4000]
        return text

    def _symbolic_analysis(self, text: str) -> dict:
        """Rule-based symbolic analysis"""
        text_lower = text.lower()

        # Check knowledge base
        has_positive_edu = any(word in text_lower for word in self.education_lexicon['positive'])
        has_negative_edu = any(word in text_lower for word in self.education_lexicon['negative'])
        is_neutral_indicator = any(word in text_lower for word in self.education_lexicon['neutral_indicators'])

        # Pattern detection
        is_question = '?' in text
        has_sarcasm_emoji = any(emoji in text for emoji in ['🙄', '😏', '😒', '😂', '💀'])
        has_contrast = 'but' in text_lower or 'however' in text_lower
        has_both_sentiments = has_positive_edu and has_negative_edu

        return {
            'has_positive_edu': has_positive_edu,
            'has_negative_edu': has_negative_edu,
            'is_neutral_indicator': is_neutral_indicator,
            'is_question': is_question,
            'has_sarcasm_emoji': has_sarcasm_emoji,
            'has_contrast': has_contrast,
            'has_both_sentiments': has_both_sentiments
        }

    def _resolve_conflicts(self, neural_sentiment: str, neural_confidence: float, symbolic: dict) -> str:
        """
        Resolve conflicts between neural and symbolic systems.
        Priority order: Questions > Neutral indicators > Mixed sentiment > Sarcasm > Low confidence
        """

        # Rule 1: Questions override to neutral (highest priority)
        if symbolic['is_question']:
            return 'neutral'

        # Rule 2: Neutral indicators force neutral
        if symbolic['is_neutral_indicator']:
            return 'neutral'

        # Rule 3: Mixed sentiment (both positive and negative with contrast) → neutral
        if symbolic['has_both_sentiments'] and symbolic['has_contrast']:
            return 'neutral'

        # Rule 4: Sarcasm emoji → force negative
        if symbolic['has_sarcasm_emoji']:
            return 'negative'

        # Rule 5: Low confidence → neutral
        if neural_confidence < 0.6:
            return 'neutral'

        # Default: trust neural network
        return neural_sentiment

    def classify(self, text: str) -> dict:
        """Hybrid classification combining neural and symbolic approaches."""
        if not text or pd.isna(text):
            return {'polarity': 'neutral', 'neural_sentiment': 'neutral',
                    'neural_confidence': 0.5, 'symbolic_flags': {}}

        processed = self.preprocess(text)

        # Neural component (RoBERTa) - with error handling
        neural_sentiment = 'neutral'
        neural_confidence = 0.5

        if len(processed) > 10:
            try:
                # Call the model safely
                result = self.neural_model(processed)
                if result and len(result) > 0:
                    neural_sentiment = result[0]['label'].lower()
                    neural_confidence = result[0]['score']
            except Exception as e:
                # If model fails, fall back to neutral
                print(f"  Warning: Model error on text length {len(processed)}: {e}")
                neural_sentiment = 'neutral'
                neural_confidence = 0.5

        # Symbolic component (rule-based + knowledge base)
        symbolic = self._symbolic_analysis(text)

        # Hybrid resolution
        final_sentiment = self._resolve_conflicts(neural_sentiment, neural_confidence, symbolic)

        return {
            'polarity': final_sentiment,
            'neural_sentiment': neural_sentiment,
            'neural_confidence': round(neural_confidence, 3),
            'symbolic_flags': {k: v for k, v in symbolic.items() if v is True}
        }


def evaluate_hybrid_vs_baseline():
    """Compare Hybrid classifier accuracy against baseline RoBERTa"""

    BASE_DIR = r"C:\Users\Sutheerth\PycharmProjects\PythonProject2"
    EVAL_FILE = f"{BASE_DIR}/eval.xls"

    print("="*70)
    print("HYBRID CLASSIFIER ACCURACY EVALUATION")
    print("="*70)

    # Load evaluation data
    print(f"\nLoading evaluation data from: {EVAL_FILE}")
    df = pd.read_excel(EVAL_FILE)
    y_true = df['sentiment_label'].str.lower().str.strip().tolist()
    texts = df['text'].tolist()
    print(f"Loaded {len(texts)} labeled records")
    print(f"Distribution: {pd.Series(y_true).value_counts().to_dict()}")

    # Initialize classifier
    print("\nInitializing Hybrid Classifier...")
    classifier = HybridClassifier()

    # Run hybrid classifier on eval set
    print("\nRunning Hybrid Classifier on eval set...")
    hybrid_predictions = []
    rule_overrides = 0
    errors = 0

    for i, text in enumerate(texts):
        try:
            result = classifier.classify(text)
            hybrid_predictions.append(result['polarity'])

            # Count when symbolic rules overrode neural
            if result['polarity'] != result['neural_sentiment']:
                rule_overrides += 1
        except Exception as e:
            # If any error, default to neutral
            hybrid_predictions.append('neutral')
            errors += 1

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(texts)} records")

    if errors > 0:
        print(f"  Errors encountered: {errors} (defaulted to neutral)")

    # Calculate hybrid accuracy
    hybrid_accuracy = accuracy_score(y_true, hybrid_predictions)

    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)

    print(f"\nHybrid Classifier Accuracy: {hybrid_accuracy:.4f} ({hybrid_accuracy*100:.1f}%)")
    print(f"Rule overrides applied: {rule_overrides} ({100*rule_overrides/len(texts):.1f}%)")

    # Load baseline predictions (RoBERTa only from your original classify.py)
    baseline_file = f"{BASE_DIR}/data/analysis/classification_results_eval.csv"

    if os.path.exists(baseline_file):
        baseline_df = pd.read_csv(baseline_file)
        baseline_predictions = baseline_df['polarity'].str.lower().str.strip().tolist()
        baseline_accuracy = accuracy_score(y_true, baseline_predictions)

        print(f"\nBaseline (RoBERTa only) Accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.1f}%)")
        print(f"\n{'='*50}")
        improvement = hybrid_accuracy - baseline_accuracy
        print(f"IMPROVEMENT: +{improvement*100:.1f}%")
        print(f"{'='*50}")
    else:
        print("\n⚠️ Baseline file not found. Run 'python classify.py --eval-only' first")
        print(f"   Expected at: {baseline_file}")

    # Detailed classification report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT (Hybrid Classifier)")
    print("="*70)
    print(classification_report(y_true, hybrid_predictions,
                                labels=['positive', 'negative', 'neutral'],
                                target_names=['Positive', 'Negative', 'Neutral']))

    # Show examples of where hybrid fixed baseline errors
    if os.path.exists(baseline_file):
        print("\n" + "="*70)
        print("EXAMPLES WHERE HYBRID FIXED BASELINE ERRORS")
        print("="*70)

        fixed_count = 0
        for i, (true, baseline_pred, hybrid_pred, text) in enumerate(zip(
            y_true, baseline_predictions, hybrid_predictions, texts)):

            if baseline_pred != true and hybrid_pred == true and fixed_count < 5:
                print(f"\nExample {fixed_count + 1}:")
                print(f"  Text: {str(text)[:150]}...")
                print(f"  True label: {true}")
                print(f"  Baseline (RoBERTa): {baseline_pred} ❌")
                print(f"  Hybrid: {hybrid_pred} ✅")

                # Show which rule fixed it
                result = classifier.classify(text)
                if result['polarity'] != result['neural_sentiment']:
                    flags = list(result['symbolic_flags'].keys())
                    if flags:
                        print(f"  Rule applied: {flags}")
                fixed_count += 1

        if fixed_count == 0:
            print("\n  Checking baseline vs hybrid differences...")
            # Show any difference
            for i, (baseline_pred, hybrid_pred) in enumerate(zip(baseline_predictions, hybrid_predictions)):
                if baseline_pred != hybrid_pred and fixed_count < 5:
                    print(f"\nExample {fixed_count + 1}:")
                    print(f"  Text: {str(texts[i])[:150]}...")
                    print(f"  Baseline: {baseline_pred} → Hybrid: {hybrid_pred}")
                    fixed_count += 1

    return hybrid_accuracy


if __name__ == "__main__":
    import os
    evaluate_hybrid_vs_baseline()