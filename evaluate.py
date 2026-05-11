"""
evaluate.py
-----------
Heart Disease Prediction System
Evaluates all saved ensemble models and prints:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix (text)
  - Saves comparison CSV to models/model_comparison.csv
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from preprocessing import run_preprocessing

DATASET_PATH = "dataset/heart.csv"
TARGET_COL   = "target"
MODELS_DIR   = "models"
MODEL_NAMES  = ["RandomForest", "GradientBoosting", "AdaBoost", "VotingClassifier"]


def load_model(name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        print(f"[WARN] Model file not found: {path}")
        return None
    return joblib.load(path)


def evaluate_model(name, model, X_test, y_test):
    """Compute and print all metrics for one model."""
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  Model: {name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred,
                                target_names=["No Disease", "Heart Disease"]))

    return {"Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1 Score": f1}


def main():
    print("=" * 55)
    print("   Heart Disease Prediction — Model Evaluation")
    print("=" * 55)

    print("\n[STEP 1] Preprocessing data (re-running split) ...")
    _, X_test, _, y_test, _, _ = run_preprocessing(DATASET_PATH, TARGET_COL)

    results = []
    for name in MODEL_NAMES:
        model = load_model(name)
        if model is not None:
            row = evaluate_model(name, model, X_test, y_test)
            results.append(row)

    if results:
        df_results = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
        df_results.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)

        print("\n" + "=" * 55)
        print("   SUMMARY TABLE (sorted by Accuracy)")
        print("=" * 55)
        print(df_results.to_string(index=False))
        print(f"\n[SAVED] Comparison table → models/model_comparison.csv")


if __name__ == "__main__":
    main()