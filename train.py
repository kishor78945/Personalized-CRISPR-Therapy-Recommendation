"""
Training pipeline for the Personalized CRISPR Therapy Recommendation System.

Trains two Random Forest classifiers on the Breast Cancer Gene Expression
(CuMiDa) dataset:
  1. Main model   — classifies all 6 breast cancer subtypes
  2. Specialist   — binary classifier for Basal vs Luminal-B

Usage
-----
    python train.py --data Breast_GSE45827.csv
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import (
    CV_FOLDS,
    CV_RESULTS_PATH,
    MAIN_ENCODER_PATH,
    MAIN_MODEL_PATH,
    MAIN_RF_PARAMS,
    MAIN_SCALER_PATH,
    MODELS_DIR,
    NUM_TOP_FEATURES,
    RANDOM_STATE,
    SPEC_ENCODER_PATH,
    SPEC_MODEL_PATH,
    SPEC_RF_PARAMS,
    SPEC_SCALER_PATH,
    TEST_SIZE,
)
from utils import save_artifact

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """Load the CSV dataset and drop the 'samples' column if present."""
    log.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    if "samples" in df.columns:
        df = df.drop(columns=["samples"])
        log.info("Dropped 'samples' column")
    log.info("Dataset shape: %s", df.shape)
    return df


def balance_with_smote(X: pd.DataFrame, y: pd.Series):
    """Apply SMOTE to balance all classes."""
    log.info("Applying SMOTE to balance classes …")
    smote = SMOTE(sampling_strategy="auto", random_state=RANDOM_STATE)
    X_bal, y_bal = smote.fit_resample(X, y)
    log.info("Balanced dataset shape: %s", X_bal.shape)
    return X_bal, y_bal


def select_top_features(X, y, k: int = NUM_TOP_FEATURES):
    """Select the top-K genes using ANOVA F-test."""
    log.info("Selecting top %d features via ANOVA F-test …", k)
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    mask = selector.get_support()
    return X_selected, mask, selector


def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    rf_params: dict,
    label_encoder: LabelEncoder,
    model_name: str = "Model",
):
    """Train a Random Forest, evaluate it, and return the fitted model."""
    log.info("Training %s …", model_name)
    clf = RandomForestClassifier(**rf_params)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    log.info("─── %s Evaluation ───", model_name)
    log.info("Accuracy: %.4f", accuracy)
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
    )
    print(f"\n{'═' * 50}")
    print(f"  {model_name} — Classification Report")
    print(f"{'═' * 50}")
    print(report)

    # Cross-validation
    cv_scores = cross_val_score(
        clf, X_train, y_train, cv=CV_FOLDS, n_jobs=-1,
    )
    log.info(
        "Cross-Validation (%d-fold): %.4f ± %.4f",
        CV_FOLDS, cv_scores.mean(), cv_scores.std(),
    )

    return clf, accuracy, cv_scores


# ---------------------------------------------------------------------------
# Main training pipelines
# ---------------------------------------------------------------------------

def train_main_model(df: pd.DataFrame):
    """
    Train the main 6-class breast cancer subtype classifier.

    Steps: encode → SMOTE → feature selection → scale → split → train → save.
    """
    log.info("=" * 60)
    log.info("MAIN MODEL — All 6 Subtypes")
    log.info("=" * 60)

    # Encode target labels
    label_encoder = LabelEncoder()
    df["type"] = label_encoder.fit_transform(df["type"])
    log.info("Classes: %s", list(label_encoder.classes_))

    X = df.drop(columns=["type"])
    y = df["type"]

    # Balance
    X_bal, y_bal = balance_with_smote(X, y)

    # Feature selection
    X_sel, mask, _ = select_top_features(X_bal, y_bal, NUM_TOP_FEATURES)
    selected_features = list(X.columns[mask])

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_bal,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_bal,
    )

    # Train & evaluate
    model, accuracy, cv_scores = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        MAIN_RF_PARAMS, label_encoder,
        model_name="Main Model (6-class)",
    )

    # Save artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_artifact(
        {"model": model, "feature_names": selected_features, "accuracy": accuracy},
        MAIN_MODEL_PATH,
    )
    save_artifact(scaler, MAIN_SCALER_PATH)
    save_artifact(label_encoder, MAIN_ENCODER_PATH)

    # Save cross-validation results
    cv_df = pd.DataFrame({
        "CV Mean Accuracy": [cv_scores.mean()],
        "CV Std Dev": [cv_scores.std()],
        "Test Accuracy": [accuracy],
        "Features Before": [X.shape[1]],
        "Features After": [len(selected_features)],
    })
    cv_df.to_csv(CV_RESULTS_PATH, index=False)
    log.info("Cross-validation results → %s", CV_RESULTS_PATH)

    return label_encoder


def train_specialist_model(df: pd.DataFrame, label_encoder: LabelEncoder):
    """
    Train a specialized binary classifier for Basal vs Luminal-B.

    These two subtypes can be harder to distinguish, so a dedicated model
    improves accuracy when the main model is uncertain.
    """
    log.info("")
    log.info("=" * 60)
    log.info("SPECIALIST MODEL — Basal vs Luminal B")
    log.info("=" * 60)

    # Re-encode if needed (df['type'] should already be encoded)
    basal_id = label_encoder.transform(["basal"])[0]
    lumB_id = label_encoder.transform(["luminal_B"])[0]

    df_filtered = df[df["type"].isin([basal_id, lumB_id])].copy()
    log.info("Filtered dataset shape: %s", df_filtered.shape)

    X = df_filtered.drop(columns=["type"])
    y = df_filtered["type"]

    # Balance
    X_bal, y_bal = balance_with_smote(X, y)

    # Feature selection
    X_sel, mask, _ = select_top_features(X_bal, y_bal, NUM_TOP_FEATURES)
    selected_features = list(X.columns[mask])

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_bal,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_bal,
    )

    # Train & evaluate
    model, accuracy, _ = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        SPEC_RF_PARAMS, label_encoder,
        model_name="Specialist Model (Basal vs Luminal B)",
    )

    # Save artifacts
    save_artifact(
        {"model": model, "feature_names": selected_features},
        SPEC_MODEL_PATH,
    )
    save_artifact(scaler, SPEC_SCALER_PATH)
    save_artifact(label_encoder, SPEC_ENCODER_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train breast cancer subtype classifiers for CRISPR therapy recommendations.",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the Breast_GSE45827.csv gene expression dataset.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        log.error("Dataset file not found: %s", args.data)
        sys.exit(1)

    # Load the raw dataset
    df = load_dataset(args.data)

    # Train both models (the label encoder from the main model is reused)
    label_encoder = train_main_model(df.copy())
    train_specialist_model(df.copy(), label_encoder)

    log.info("")
    log.info("✅ All models trained and saved to %s", MODELS_DIR)


if __name__ == "__main__":
    main()
