"""
preprocessing.py
----------------
Heart Disease Prediction System
Handles all data preprocessing tasks:
  - Loading dataset
  - Cleaning (duplicates, types)
  - Missing value imputation
  - Encoding categorical features
  - Feature scaling
  - Train/test split
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
def load_data(filepath):
    """Load CSV dataset and return a DataFrame."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 2. CLEAN DATA
# ─────────────────────────────────────────────
def clean_data(df):
    """Remove duplicates and fix data types."""
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"[INFO] Removed {before - after} duplicate rows")

    # Ensure all expected numeric columns are numeric
    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("[INFO] Data cleaning complete")
    return df


# ─────────────────────────────────────────────
# 3. HANDLE MISSING VALUES
# ─────────────────────────────────────────────
def handle_missing_values(df):
    """
    Impute missing values:
      - Numerical columns → Median imputation
      - Categorical columns → Mode imputation
    """
    print(f"[INFO] Missing values before imputation:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  → '{col}' filled with median = {median_val}")
            else:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"  → '{col}' filled with mode = {mode_val}")

    print("[INFO] Missing value imputation complete")
    return df


# ─────────────────────────────────────────────
# 4. ENCODE CATEGORICAL FEATURES
# ─────────────────────────────────────────────
def encode_features(df, target_col='target'):
    """
    Encode categorical features using Label Encoding.
    The target column is binarized to 0 (No Disease) / 1 (Disease).
    Returns: encoded DataFrame, list of encoded column names
    """
    # Binarize target: original dataset has values 0-4; 0 = no disease, 1-4 = disease
    if df[target_col].max() > 1:
        df[target_col] = df[target_col].apply(lambda x: 1 if x > 0 else 0)
        print(f"[INFO] Target '{target_col}' binarized (0=No Disease, 1=Disease)")

    # Identify categorical columns (excluding target)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        print(f"  → '{col}' label-encoded")

    # Save encoders
    os.makedirs('models', exist_ok=True)
    joblib.dump(le_dict, 'models/label_encoders.pkl')
    print("[INFO] Label encoders saved to models/label_encoders.pkl")

    return df, cat_cols


# ─────────────────────────────────────────────
# 5. FEATURE SCALING
# ─────────────────────────────────────────────
def scale_features(X_train, X_test):
    """
    Apply StandardScaler to numerical features.
    Fit on training data only; transform both train and test.
    Returns: scaled X_train, scaled X_test, fitted scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Save scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("[INFO] StandardScaler fitted and saved to models/scaler.pkl")

    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
def split_data(df, target_col='target', test_size=0.2, random_state=42):
    """
    Split data into features (X) and target (y),
    then into training and testing sets (80/20 split).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y          # preserve class balance
    )
    print(f"[INFO] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"[INFO] Class distribution in train:\n{y_train.value_counts()}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


# ─────────────────────────────────────────────
# FULL PIPELINE (called by train_model.py)
# ─────────────────────────────────────────────
def run_preprocessing(filepath, target_col='target'):
    """Run complete preprocessing pipeline and return split data."""
    df = load_data(filepath)
    df = clean_data(df)
    df = handle_missing_values(df)
    df, _ = encode_features(df, target_col)
    X_train, X_test, y_train, y_test, feature_names = split_data(df, target_col)
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
    return X_train_sc, X_test_sc, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    run_preprocessing("dataset/heart.csv")