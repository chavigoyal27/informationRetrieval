"""
ensemble_classification.py

Executes an Ablation Study for Sentiment Classification over AI in Education data.
This script tests a pipeline utilizing a Lexical Gatekeeper (TextBlob) to filter 
objective texts, avoiding unnecessary neural network inference. It subsequently runs 
an ensemble of three SOTA Deep Learning models (RoBERTa, BERTweet, ReviewBERT). 
Finally, it evaluates both a Soft Voting consensus and a Meta-Model Stacking 
(Logistic Regression) approach to determine the most accurate classification strategy.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob
from transformers import pipeline

# For importing preprocess function
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from preprocess import preprocess

EVAL_FILE = os.path.join(BASE_DIR, "data", "final_corpus", "eval.xls")

def label_to_numeric(label):
    """Convert string ground truth to numeric classes."""
    label = str(label).lower().strip()
    
    if label == 'positive': 
        return 2
    elif label == 'neutral': 
        return 1
    elif label == 'negative': 
        return 0
        
    return 1 # Default neutral

def normalize_label(label):
    """
    Maps varying model output labels to standard keys.
    - RoBERTa outputs: 'positive', 'neutral', 'negative'
    - BERTweet outputs: 'POS', 'NEU', 'NEG'
    - ReviewBERT outputs: '1 star', '2 stars', '3 stars', '4 stars', '5 stars'
    """
    label = label.lower()
    
    # Map Positive labels to 2
    if label in ['pos', 'positive', '4 stars', '5 stars']: 
        return 2
        
    # Map Negative labels to 0
    elif label in ['neg', 'negative', '1 star', '2 stars']: 
        return 0
        
    # Map Neutral labels to 1
    elif label in ['neu', 'neutral', '3 stars']: 
        return 1
        
    return 1

def evaluate_deep_ensemble(texts, ground_truth):
    
    # Load HuggingFace pipelines
    print("Loading Deep Learning Sentiment Models...")
    model_defs = {
        "M1_RoBERTa_Twitter": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "M2_BERTweet": "finiteautomata/bertweet-base-sentiment-analysis",
        "M3_ReviewBERT": "nlptown/bert-base-multilingual-uncased-sentiment"
    }
    
    pipes = {}
    # Load all models first to avoid repeated loading during inference with fixed shared params
    for name, path in model_defs.items():
        print(f"  Loading {name}...")
        pipes[name] = pipeline("text-classification", model=path, top_k=None, truncation=True, max_length=128, device=0)
        
    print(f"\nPreprocessing {len(texts)} texts...")
    clean_texts = [preprocess(t) or "empty" for t in texts]
    
    # TextBlob: Filter out objective texts (subjectivity < 0.15)
    subj_scores = [TextBlob(ct).sentiment.subjectivity for ct in clean_texts]
    is_opinionated = [score >= 0.15 for score in subj_scores]
    opinionated_texts = [t for t, op in zip(clean_texts, is_opinionated) if op]
    
    # Inference on opinionated texts only
    model_probs = {name: [] for name in model_defs.keys()}
    
    if opinionated_texts:
        for name, pipe in pipes.items():
            print(f"  Inferencing {name} in batches...")
            outputs = pipe(opinionated_texts, batch_size=16)
            for raw_out in outputs:
                arr = np.zeros(3)
                for res in raw_out:
                    arr[normalize_label(res['label'])] += res['score']
                model_probs[name].append(arr)
                
    # Resolve Soft Voting probabilities and Stacking Meta-model features
    final_m1, final_m2, final_m3, final_ens = [], [], [], []
    X_features = []
    
    op_idx = 0
    for op in is_opinionated:
        if not op: # Text was flagged as objective -> Force neutral (1)
            for lst in [final_m1, final_m2, final_m3, final_ens]:
                lst.append(1)
            X_features.append(np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]))
        else: 
            m1_arr = model_probs["M1_RoBERTa_Twitter"][op_idx]
            m2_arr = model_probs["M2_BERTweet"][op_idx]
            m3_arr = model_probs["M3_ReviewBERT"][op_idx]
            
            final_m1.append(np.argmax(m1_arr))
            final_m2.append(np.argmax(m2_arr))
            final_m3.append(np.argmax(m3_arr))
            # Soft vote by averaging probabilities across the 3 models
            final_ens.append(np.argmax((m1_arr + m2_arr + m3_arr) / 3.0)) 
            
            X_features.append(np.concatenate([m1_arr, m2_arr, m3_arr]))
            op_idx += 1
            
    X_features = np.array(X_features)
    
    # Stacking Meta-model (Train on 80%, Test on 20%)
    print("Training the Stacking Meta-model (Logistic Regression)...")
    X_train, X_test, y_train, y_test, _, m1_test, _, m2_test, _, m3_test, _, ens_test = train_test_split(
        X_features, ground_truth, final_m1, final_m2, final_m3, final_ens, 
        test_size=0.2, random_state=42, stratify=ground_truth
    )
    
    meta_model = LogisticRegression(max_iter=1000)
    meta_model.fit(X_train, y_train)
    stacking_preds = meta_model.predict(X_test)

    print("\n" + "="*70)
    print(" DEEP ENSEMBLE ABLATION STUDY RESULTS (ON TEST SET)")
    print("="*70)
    
    acc_m1 = accuracy_score(y_test, m1_test)
    acc_m2 = accuracy_score(y_test, m2_test)
    acc_m3 = accuracy_score(y_test, m3_test)
    acc_ens = accuracy_score(y_test, ens_test)
    acc_stack = accuracy_score(y_test, stacking_preds)
    
    print("ALL MODELS applied AFTER TextBlob Subjectivity Filter (<0.15 = forced neutral)")
    print(f"1. Subjectivity Filter + M1 (RoBERTa Twitter):      {acc_m1:.4f} ({acc_m1*100:.1f}%) |Q4 Implementation|")
    print(f"2. Subjectivity Filter + M2 (BERTweet):             {acc_m2:.4f} ({acc_m2*100:.1f}%)")
    print(f"3. Subjectivity Filter + M3 (ReviewBERT Forum):     {acc_m3:.4f} ({acc_m3*100:.1f}%)")
    print("-" * 70)
    print(f"4. INNOVATION 1: Soft Voting Ensemble:              {acc_ens:.4f} ({acc_ens*100:.1f}%)")
    print(f"5. INNOVATION 2: Stacking Meta:                     {acc_stack:.4f} ({acc_stack*100:.1f}%)")
    print("="*70)
    
    # Export full dataset predictions to CSV
    print(f"\nExporting dataset predictions to data/analysis/stacked_ensemble_eval.csv...")
    all_preds_num = meta_model.predict(X_features)
    
    # Get maximum probability (confidence) score across classes
    all_probs = np.max(meta_model.predict_proba(X_features), axis=1)
    
    # Map index integers back to string labels
    num_to_label_str = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    df_out = pd.DataFrame({
        "id": range(1, len(texts) + 1),
        "text": texts,
        "subjectivity": ["opinionated" if op else "neutral" for op in is_opinionated],
        "subjectivity_score": np.round(subj_scores, 4),
        "polarity": [num_to_label_str[p] for p in all_preds_num],
        "polarity_score": np.round(all_probs, 4)
    })
    
    out_path = os.path.join(BASE_DIR, "data", "analysis", "stacked_ensemble_eval.csv")
    df_out.to_csv(out_path, index=False)
    print(f"File successfully saved to: {out_path}\n")

def main():
    print(f"Loading evaluation dataset from: {EVAL_FILE}")
    df = pd.read_excel(EVAL_FILE)
    
    df = df.dropna(subset=['text', 'sentiment_label'])
    texts = df['text'].astype(str).tolist()
    labels = np.array([label_to_numeric(l) for l in df['sentiment_label']])
    
    evaluate_deep_ensemble(texts, labels)

if __name__ == "__main__":
    main()
