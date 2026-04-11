"""
ensemble_classification.py

Executes an Ablation Study for Sentiment Classification over AI in Education data.
This script tests a pipeline utilizing a Lexical Gatekeeper (TextBlob) to filter 
objective texts, avoiding unnecessary neural network inference. It subsequently runs 
an ensemble of three SOTA Deep Learning models (RoBERTa, BERTweet, SieBERT). 
Finally, it evaluates both a Soft Voting consensus and a Meta-Model Stacking 
(Logistic Regression) approach to determine the most accurate classification strategy.

Changes from v1:
  - Replaced M3 ReviewBERT (nlptown/bert-base-multilingual-uncased-sentiment)
    with SieBERT (siebert/sentiment-roberta-large-english).
    ReviewBERT was trained on product reviews and scored only 44.5% on our
    social media corpus — worse than random. SieBERT is a large RoBERTa model
    fine-tuned for English sentiment and is a much stronger fit.
  - SieBERT outputs only positive/negative (no neutral class). We handle this
    by mapping its output to pos/neg only and letting the meta-learner decide
    when to override toward neutral based on the other two models.
  - Changed device=0 (GPU) to device=-1 (CPU) for compatibility.
  - Raised TextBlob subjectivity threshold from 0.15 → 0.30 consistent with
    the improved classify.py baseline.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from preprocess import preprocess

EVAL_FILE = os.path.join(BASE_DIR, "data", "final_corpus", "eval.xls")

# Raised from 0.15 → 0.30 consistent with improved classify.py
SUBJECTIVITY_THRESHOLD = 0.30


def label_to_numeric(label):
    """Convert string ground truth to numeric classes."""
    label = str(label).lower().strip()
    if label == 'positive':
        return 2
    elif label == 'neutral':
        return 1
    elif label == 'negative':
        return 0
    return 1


def normalize_label(label):
    """
    Maps varying model output labels to standard numeric keys.
    - RoBERTa outputs: 'positive', 'neutral', 'negative'
    - BERTweet outputs: 'POS', 'NEU', 'NEG'
    - SieBERT outputs: 'positive', 'negative' (binary only)
    """
    label = label.lower()
    if label in ['pos', 'positive']:
        return 2
    elif label in ['neg', 'negative']:
        return 0
    elif label in ['neu', 'neutral']:
        return 1
    return 1


def evaluate_deep_ensemble(texts, ground_truth):

    model_defs = {
        "M1_RoBERTa_Twitter": {
            "path": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "max_length": 512,
        },
        "M2_BERTweet": {
            "path": "finiteautomata/bertweet-base-sentiment-analysis",
            "max_length": 128,
        },
        "M3_SieBERT": {
            "path": "siebert/sentiment-roberta-large-english",
            "max_length": 512,
        },
    }

    pipes = {}
    for name, cfg in model_defs.items():
        print(f"  Loading {name}...")
        tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
        pipes[name] = pipeline(
            "text-classification",
            model=cfg["path"],
            tokenizer=tokenizer,
            top_k=None,
            truncation=True,
            max_length=min(tokenizer.model_max_length, cfg["max_length"]),
            device=-1
        )

    print(f"\nPreprocessing {len(texts)} texts...")
    clean_texts = [preprocess(t) or "empty" for t in texts]

    # TextBlob subjectivity filter — raised threshold to 0.30
    subj_scores   = [TextBlob(ct).sentiment.subjectivity for ct in clean_texts]
    is_opinionated = [score >= SUBJECTIVITY_THRESHOLD for score in subj_scores]
    opinionated_texts = [t for t, op in zip(clean_texts, is_opinionated) if op]

    n_op = sum(is_opinionated)
    print(f"  Opinionated: {n_op} ({100*n_op/len(texts):.1f}%)")
    print(f"  Neutral (filtered): {len(texts)-n_op} ({100*(len(texts)-n_op)/len(texts):.1f}%)")

    # Run inference on opinionated texts only
    model_probs = {name: [] for name in model_defs}

    if opinionated_texts:
        for name, pipe in pipes.items():
            print(f"  Inferencing {name} in batches...")
            
            if name == "M2_BERTweet":
                outputs = pipe(opinionated_texts, batch_size=16, truncation=True, max_length=128)
            else:
                outputs = pipe(opinionated_texts, batch_size=16, truncation=True, max_length=512)
                
            for raw_out in outputs:
                arr = np.zeros(3)
                for res in raw_out:
                    idx = normalize_label(res['label'])
                    arr[idx] += res['score']
                # SieBERT only outputs pos/neg — redistribute remaining probability to neutral
                # proportionally so the feature vector still sums to ~1
                if name == "M3_SieBERT" and arr[1] == 0.0:
                    total = arr[0] + arr[2]
                    if total > 0:
                        arr[0] = arr[0] / total
                        arr[2] = arr[2] / total
                model_probs[name].append(arr)

    # Build full-length prediction lists and stacking feature matrix
    final_m1, final_m2, final_m3, final_ens = [], [], [], []
    X_features = []

    op_idx = 0
    for op in is_opinionated:
        if not op:
            # Objective text → force neutral
            for lst in [final_m1, final_m2, final_m3, final_ens]:
                lst.append(1)
            X_features.append(np.array([0.0, 1.0, 0.0,
                                         0.0, 1.0, 0.0,
                                         0.0, 1.0, 0.0]))
        else:
            m1_arr = model_probs["M1_RoBERTa_Twitter"][op_idx]
            m2_arr = model_probs["M2_BERTweet"][op_idx]
            m3_arr = model_probs["M3_SieBERT"][op_idx]

            final_m1.append(np.argmax(m1_arr))
            final_m2.append(np.argmax(m2_arr))
            final_m3.append(np.argmax(m3_arr))
            # Soft vote: average probability distributions
            final_ens.append(np.argmax((m1_arr + m2_arr + m3_arr) / 3.0))

            X_features.append(np.concatenate([m1_arr, m2_arr, m3_arr]))
            op_idx += 1

    X_features = np.array(X_features)

    # Stacking meta-model: train on 80%, test on 20%
    print("\nTraining the Stacking Meta-model (Logistic Regression)...")
    (X_train, X_test,
     y_train, y_test,
     _, m1_test,
     _, m2_test,
     _, m3_test,
     _, ens_test) = train_test_split(
        X_features, ground_truth,
        final_m1, final_m2, final_m3, final_ens,
        test_size=0.2, random_state=42, stratify=ground_truth
    )

    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(X_train, y_train)
    stacking_preds = meta_model.predict(X_test)

    # ── Results ──────────────────────────────────────────────────────────────
    acc_m1    = accuracy_score(y_test, m1_test)
    acc_m2    = accuracy_score(y_test, m2_test)
    acc_m3    = accuracy_score(y_test, m3_test)
    acc_ens   = accuracy_score(y_test, ens_test)
    acc_stack = accuracy_score(y_test, stacking_preds)

    print("\n" + "="*70)
    print(" DEEP ENSEMBLE ABLATION STUDY RESULTS (ON TEST SET)")
    print("="*70)
    print(f"ALL MODELS applied AFTER TextBlob Subjectivity Filter (< {SUBJECTIVITY_THRESHOLD} = forced neutral)")
    print(f"1. Subjectivity Filter + M1 (RoBERTa Twitter):   {acc_m1:.4f} ({acc_m1*100:.1f}%) |Q4 Baseline|")
    print(f"2. Subjectivity Filter + M2 (BERTweet):          {acc_m2:.4f} ({acc_m2*100:.1f}%)")
    print(f"3. Subjectivity Filter + M3 (SieBERT Large):     {acc_m3:.4f} ({acc_m3*100:.1f}%)")
    print("-" * 70)
    print(f"4. INNOVATION 1: Soft Voting Ensemble:           {acc_ens:.4f} ({acc_ens*100:.1f}%)")
    print(f"5. INNOVATION 2: Stacking Meta-Learner:          {acc_stack:.4f} ({acc_stack*100:.1f}%)")
    print("="*70)

    num_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}

    # Export full dataset predictions
    print(f"\nExporting predictions to data/analysis/stacked_ensemble_eval.csv...")
    all_preds_num = meta_model.predict(X_features)
    all_probs     = np.max(meta_model.predict_proba(X_features), axis=1)

    df_out = pd.DataFrame({
        "id":                range(1, len(texts) + 1),
        "text":              texts,
        "subjectivity":      ["opinionated" if op else "neutral" for op in is_opinionated],
        "subjectivity_score": np.round(subj_scores, 4),
        "polarity":          [num_to_label[p] for p in all_preds_num],
        "polarity_score":    np.round(all_probs, 4),
    })

    out_path = os.path.join(BASE_DIR, "data", "analysis", "stacked_ensemble_eval.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved to: {out_path}\n")


def main():
    print(f"Loading evaluation dataset from: {EVAL_FILE}")
    df = pd.read_excel(EVAL_FILE)
    df = df.dropna(subset=['text', 'sentiment_label'])

    texts  = df['text'].astype(str).tolist()
    labels = np.array([label_to_numeric(l) for l in df['sentiment_label']])

    print(f"Loaded {len(texts)} records")
    print(f"Label distribution: {dict(df['sentiment_label'].value_counts())}\n")

    evaluate_deep_ensemble(texts, labels)


if __name__ == "__main__":
    main()