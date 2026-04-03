"""
Ablation_Complete.py — Complete Ablation Study for Question 5
Compares: Baseline, Hybrid only, Ensemble only, Hybrid+Ensemble
"""

import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Import your classifiers
from Q5ensemble import EnsembleClassifier
from hybrid import HybridClassifier
from Hybrid_ensemble import HybridEnsembleClassifier

# Paths
BASE_DIR = r"C:\Users\Sutheerth\PycharmProjects\PythonProject2"
EVAL_FILE = os.path.join(BASE_DIR, "eval.xls")
BASELINE_FILE = os.path.join(BASE_DIR, "data", "analysis", "classification_results_eval.csv")


def load_ground_truth():
    """Load ground truth labels from eval.xls"""
    df = pd.read_excel(EVAL_FILE)
    y_true = df['sentiment_label'].str.lower().str.strip().tolist()
    texts = df['text'].tolist()
    return y_true, texts


def load_baseline_predictions():
    """Load baseline (RoBERTa only) predictions"""
    if not os.path.exists(BASELINE_FILE):
        print(f"⚠️ Baseline file not found: {BASELINE_FILE}")
        print("   Please run 'python classify.py --eval-only' first")
        return None

    df = pd.read_csv(BASELINE_FILE)
    y_pred = df['polarity'].str.lower().str.strip().tolist()
    return y_pred


def run_classifier(classifier, texts, name):
    """Run a classifier and return predictions"""
    print(f"\n  Running {name}...")
    predictions = []

    for i, text in enumerate(texts):
        result = classifier.classify(text)
        predictions.append(result['polarity'])
        if (i + 1) % 200 == 0:
            print(f"    Processed {i + 1}/{len(texts)} records")

    return predictions


def calculate_metrics(y_true, y_pred, name):
    """Calculate and return metrics for a model"""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        'name': name,
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted
    }


def print_ablation_table(results, baseline_acc, baseline_f1):
    """Print ablation study comparison table"""
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS - QUESTION 5")
    print("="*70)

    print(f"\n{'Configuration':<25} {'Accuracy':<12} {'Macro F1':<12} {'Δ Accuracy':<12}")
    print("-"*70)

    for r in results:
        acc = r['accuracy']
        f1 = r['f1_macro']
        delta_acc = acc - baseline_acc
        delta_f1 = f1 - baseline_f1
        sign = "+" if delta_acc >= 0 else ""
        print(f"{r['name']:<25} {acc:.4f}       {f1:.4f}       {sign}{delta_acc:.4f}")


def print_improvement_summary(results, baseline_acc):
    """Print summary of improvements"""
    print("\n" + "="*70)
    print("IMPROVEMENT SUMMARY")
    print("="*70)

    for r in results:
        if r['name'] != 'Baseline (RoBERTa)':
            improvement = (r['accuracy'] - baseline_acc) * 100
            status = "✅" if improvement > 0 else "⚠️"
            print(f"\n  {status} {r['name']}: +{improvement:.1f}% improvement")
            print(f"     ({baseline_acc:.1%} → {r['accuracy']:.1%})")


def main():
    """Run complete ablation study"""
    print("="*70)
    print("COMPLETE ABLATION STUDY: Baseline vs Hybrid vs Ensemble vs Both")
    print("="*70)

    # Load data
    print("\n[1/5] Loading evaluation data...")
    y_true, texts = load_ground_truth()
    print(f"  Loaded {len(texts)} labeled records")
    print(f"  Distribution: {pd.Series(y_true).value_counts().to_dict()}")

    results = []

    # Load baseline
    print("\n[2/5] Loading baseline predictions...")
    y_pred_baseline = load_baseline_predictions()

    if y_pred_baseline is None:
        print("  Cannot proceed without baseline predictions")
        return

    min_len = min(len(y_true), len(y_pred_baseline))
    y_true = y_true[:min_len]
    y_pred_baseline = y_pred_baseline[:min_len]
    texts = texts[:min_len]

    baseline_metrics = calculate_metrics(y_true, y_pred_baseline, "Baseline (RoBERTa)")
    results.append(baseline_metrics)
    baseline_acc = baseline_metrics['accuracy']
    baseline_f1 = baseline_metrics['f1_macro']
    print(f"  Baseline Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.1f}%)")

    # Run Hybrid
    print("\n[3/5] Running Hybrid Classifier...")
    hybrid_classifier = HybridClassifier()
    y_pred_hybrid = run_classifier(hybrid_classifier, texts, "Hybrid")
    hybrid_metrics = calculate_metrics(y_true, y_pred_hybrid, "Hybrid (Neural+Symbolic)")
    results.append(hybrid_metrics)
    print(f"  Hybrid Accuracy: {hybrid_metrics['accuracy']:.4f} ({hybrid_metrics['accuracy']*100:.1f}%)")

    # Run Ensemble
    print("\n[4/5] Running Ensemble Classifier...")
    ensemble_classifier = EnsembleClassifier()
    y_pred_ensemble = run_classifier(ensemble_classifier, texts, "Ensemble")
    ensemble_metrics = calculate_metrics(y_true, y_pred_ensemble, "Ensemble (Voting+Rules)")
    results.append(ensemble_metrics)
    print(f"  Ensemble Accuracy: {ensemble_metrics['accuracy']:.4f} ({ensemble_metrics['accuracy']*100:.1f}%)")

    # Run Hybrid + Ensemble
    print("\n[5/5] Running Hybrid+Ensemble Classifier...")
    both_classifier = HybridEnsembleClassifier()
    y_pred_both = run_classifier(both_classifier, texts, "Hybrid+Ensemble")
    both_metrics = calculate_metrics(y_true, y_pred_both, "Hybrid + Ensemble (Combined)")
    results.append(both_metrics)
    print(f"  Hybrid+Ensemble Accuracy: {both_metrics['accuracy']:.4f} ({both_metrics['accuracy']*100:.1f}%)")

    # Print results
    print_ablation_table(results, baseline_acc, baseline_f1)
    print_improvement_summary(results, baseline_acc)

    # Find best model
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\n{'='*70}")
    print(f"🏆 BEST MODEL: {best['name']} with {best['accuracy']*100:.1f}% accuracy")
    print(f"{'='*70}")

    # Save results
    results_df = pd.DataFrame([
        {
            'configuration': r['name'],
            'accuracy': r['accuracy'],
            'macro_f1': r['f1_macro'],
            'weighted_f1': r['f1_weighted'],
            'improvement': (r['accuracy'] - baseline_acc) * 100
        }
        for r in results
    ])

    output_file = os.path.join(BASE_DIR, "data", "analysis", "ablation_study_complete.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n📁 Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    main()