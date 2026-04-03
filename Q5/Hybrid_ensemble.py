"""
Hybrid_Ensemble.py — Combined Hybrid + Ensemble Classifier
Uses both: Hybrid (neural+symbolic) AND Ensemble (voting+rules)
"""

import pandas as pd
import re
from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class HybridEnsembleClassifier:
    """
    Combines both innovations:
    1. Hybrid: Neural (RoBERTa) + Symbolic (rules + knowledge base)
    2. Ensemble: Voting across multiple models (RoBERTa, VADER, TextBlob)
    """

    def __init__(self):
        print("Loading Hybrid+Ensemble models...")

        # Ensemble models
        self.roberta = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            truncation=True,
            max_length=512,
            device=-1
        )
        self.vader = SentimentIntensityAnalyzer()

        # Knowledge base for hybrid component
        self.education_lexicon = {
            'positive': ['engaging', 'insightful', 'thought-provoking', 'well-structured', 'comprehensive',
                         'helpful', 'useful', 'clear', 'effective', 'valuable'],
            'negative': ['boring', 'outdated', 'confusing', 'irrelevant', 'overwhelming', 'useless',
                         'waste', 'terrible', 'awful', 'difficult', 'hard'],
            'neutral_indicators': ['according to', 'refers to', 'defines', 'states that', 'the research shows']
        }

        print("Models loaded.")

    def preprocess(self, text: str) -> str:
        if not text or pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:512]

    def get_roberta_sentiment(self, text: str) -> tuple:
        try:
            if len(text) < 10:
                return 'neutral', 0.5
            result = self.roberta(text)[0]
            return result['label'].lower(), result['score']
        except:
            return 'neutral', 0.5

    def get_vader_sentiment(self, text: str) -> tuple:
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        if compound >= 0.05:
            return 'positive', abs(compound)
        elif compound <= -0.05:
            return 'negative', abs(compound)
        else:
            return 'neutral', 1 - abs(compound)

    def get_textblob_sentiment(self, text: str) -> tuple:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return 'positive', polarity
        elif polarity < -0.1:
            return 'negative', abs(polarity)
        else:
            return 'neutral', 1 - abs(polarity)

    def symbolic_analysis(self, text: str) -> dict:
        """Hybrid component: rule-based symbolic analysis"""
        text_lower = text.lower()

        has_positive_edu = any(word in text_lower for word in self.education_lexicon['positive'])
        has_negative_edu = any(word in text_lower for word in self.education_lexicon['negative'])
        is_neutral_indicator = any(word in text_lower for word in self.education_lexicon['neutral_indicators'])
        is_question = '?' in text
        has_sarcasm_emoji = any(emoji in text for emoji in ['🙄', '😏', '😒'])
        has_contrast = 'but' in text_lower or 'however' in text_lower
        has_both_sentiments = has_positive_edu and has_negative_edu

        return {
            'is_question': is_question,
            'has_sarcasm_emoji': has_sarcasm_emoji,
            'has_both_sentiments': has_both_sentiments,
            'has_contrast': has_contrast,
            'is_neutral_indicator': is_neutral_indicator,
            'has_positive_edu': has_positive_edu,
            'has_negative_edu': has_negative_edu
        }

    def hybrid_resolve(self, neural_sentiment: str, neural_conf: float, symbolic: dict) -> str:
        """Hybrid conflict resolution"""
        if symbolic['is_question']:
            return 'neutral'
        if symbolic['is_neutral_indicator']:
            return 'neutral'
        if symbolic['has_both_sentiments'] and symbolic['has_contrast']:
            return 'neutral'
        if symbolic['has_sarcasm_emoji']:
            return 'negative'
        if neural_conf < 0.6:
            return 'neutral'
        return neural_sentiment

    def ensemble_vote(self, predictions: list) -> str:
        """Ensemble voting"""
        weights = {'positive': 0, 'negative': 0, 'neutral': 0}

        roberta_label, roberta_score = predictions[0]
        weights[roberta_label] += 0.5 * roberta_score

        vader_label, vader_score = predictions[1]
        weights[vader_label] += 0.3 * vader_score

        blob_label, blob_score = predictions[2]
        weights[blob_label] += 0.2 * blob_score

        return max(weights, key=weights.get)

    def rule_overrides(self, text: str, current_label: str) -> tuple:
        """Final rule overrides"""
        text_lower = text.lower()

        # Questions
        if '?' in text and len(text.split()) > 3:
            return 'neutral', 'question_override'

        # Off-topic
        off_topic = [r'bets?', r'odds?', r'ML', r'ADD', r'CBB', r'🏀', r'⚽']
        if any(re.search(p, text_lower) for p in off_topic):
            return 'neutral', 'offtopic_override'

        # Strong indicators
        if re.search(r'i love|i enjoy|excellent|amazing|perfect', text_lower):
            return 'positive', 'positive_override'
        if re.search(r'i hate|terrible|awful|useless|waste of|cooked', text_lower):
            return 'negative', 'negative_override'

        return current_label, 'no_override'

    def classify(self, text: str) -> dict:
        """Combined Hybrid + Ensemble classification"""
        processed = self.preprocess(text)

        # Get ensemble predictions
        roberta_label, roberta_score = self.get_roberta_sentiment(processed)
        vader_label, vader_score = self.get_vader_sentiment(processed)
        blob_label, blob_score = self.get_textblob_sentiment(processed)

        predictions = [(roberta_label, roberta_score), (vader_label, vader_score), (blob_label, blob_score)]

        # Ensemble voting
        ensemble_label = self.ensemble_vote(predictions)

        # Hybrid symbolic analysis
        symbolic = self.symbolic_analysis(text)

        # Hybrid resolution (neural + symbolic)
        hybrid_label = self.hybrid_resolve(roberta_label, roberta_score, symbolic)

        # Combine: take the more conservative of ensemble and hybrid
        if ensemble_label == hybrid_label:
            final_label = ensemble_label
        elif hybrid_label == 'neutral':
            final_label = 'neutral'  # Hybrid's neutral is more conservative
        else:
            final_label = ensemble_label  # Default to ensemble

        # Final rule overrides
        final_label, override = self.rule_overrides(text, final_label)

        return {
            'polarity': final_label,
            'ensemble_vote': ensemble_label,
            'hybrid_resolve': hybrid_label,
            'roberta': roberta_label,
            'vader': vader_label,
            'textblob': blob_label,
            'override': override
        }


def evaluate_hybrid_ensemble():
    """Quick evaluation"""
    import os
    from sklearn.metrics import accuracy_score

    BASE_DIR = r"C:\Users\Sutheerth\PycharmProjects\PythonProject2"
    EVAL_FILE = f"{BASE_DIR}/eval.xls"

    df = pd.read_excel(EVAL_FILE)
    y_true = df['sentiment_label'].str.lower().str.strip().tolist()
    texts = df['text'].tolist()

    classifier = HybridEnsembleClassifier()
    predictions = []

    for i, text in enumerate(texts):
        result = classifier.classify(text)
        predictions.append(result['polarity'])
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(texts)}")

    acc = accuracy_score(y_true, predictions)
    print(f"\nHybrid+Ensemble Accuracy: {acc:.4f} ({acc * 100:.1f}%)")
    return acc


if __name__ == "__main__":
    evaluate_hybrid_ensemble()