"""
train_model.py
--------------
Heart Disease Prediction System
Trains four ensemble ML models:
  1. Random Forest
  2. Gradient Boosting
  3. AdaBoost
  4. Voting Classifier
Saves the best model as trained_model.pkl
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score

# Make sure src/ is importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from preprocessing import run_preprocessing

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATASET_PATH = "dataset/heart.csv"
TARGET_COL   = "target"
MODELS_DIR   = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# DEFINE MODELS
# ─────────────────────────────────────────────
def get_models():
    """Return a dict of ensemble models to train."""
    rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    gb  = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)

    voting = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('ada', ada)],
        voting='soft'          # uses predicted probabilities
    )

    return {
        "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting":   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        "AdaBoost":           AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
        "VotingClassifier":   voting,
    }


# ─────────────────────────────────────────────
# TRAIN ALL MODELS
# ─────────────────────────────────────────────
def train_all(X_train, y_train, models):
    """Fit every model and return the fitted dict."""
    trained = {}
    for name, model in models.items():
        print(f"[TRAIN] Training {name} ...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"  → {name} trained successfully")
    return trained


# ─────────────────────────────────────────────
# SELECT BEST MODEL
# ─────────────────────────────────────────────
def select_best(trained_models, X_test, y_test):
    """Compare test accuracies and return name + model of the best."""
    results = {}
    for name, model in trained_models.items():
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"  {name:25s} → Test Accuracy: {acc:.4f}")

    best_name  = max(results, key=results.get)
    best_model = trained_models[best_name]
    print(f"\n[BEST] {best_name} with accuracy {results[best_name]:.4f}")
    return best_name, best_model, results


# ─────────────────────────────────────────────
# SAVE MODELS
# ─────────────────────────────────────────────
def save_models(trained_models, best_name):
    """Save every model individually and mark the best."""
    for name, model in trained_models.items():
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, path)
        print(f"  Saved: {path}")

    best_path = os.path.join(MODELS_DIR, "trained_model.pkl")
    joblib.dump(trained_models[best_name], best_path)
    print(f"\n[SAVED] Best model → {best_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("   Heart Disease Prediction — Model Training")
    print("=" * 55)

    # Step 1: Preprocess
    print("\n[STEP 1] Preprocessing data ...")
    X_train, X_test, y_train, y_test, feature_names, scaler = \
        run_preprocessing(DATASET_PATH, TARGET_COL)

    # Save feature names for web app
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

    # Step 2: Define models
    models = get_models()

    # Step 3: Train
    print("\n[STEP 2] Training models ...")
    trained_models = train_all(X_train, y_train, models)

    # Step 4: Evaluate and select
    print("\n[STEP 3] Evaluating models on test set ...")
    best_name, best_model, results = select_best(trained_models, X_test, y_test)

    # Step 5: Save
    print("\n[STEP 4] Saving all models ...")
    save_models(trained_models, best_name)

    print("\n[DONE] Training complete!")
    print(f"        Best model: {best_name}")
    print("        All files saved in models/ folder")


if __name__ == "__main__":
    main()