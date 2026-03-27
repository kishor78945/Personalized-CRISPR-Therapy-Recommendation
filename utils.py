"""
Utility functions for the Personalized CRISPR Therapy Recommendation System.

Provides model loading, data validation, and shared helpers used by both
the training pipeline and the Streamlit application.
"""

import os
import pickle
import logging

import pandas as pd

from config import (
    MAIN_MODEL_PATH,
    MAIN_SCALER_PATH,
    MAIN_ENCODER_PATH,
    SPEC_MODEL_PATH,
    SPEC_SCALER_PATH,
    SPEC_ENCODER_PATH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model persistence helpers
# ---------------------------------------------------------------------------

def save_artifact(obj, path: str) -> None:
    """Serialize a Python object to a pickle file, creating dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved artifact → %s", path)


def load_artifact(path: str):
    """Deserialize a Python object from a pickle file."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info("Loaded artifact ← %s", path)
    return obj


def load_main_model():
    """
    Load the main 6-class model and its associated scaler & label encoder.

    Returns
    -------
    tuple : (model, scaler, label_encoder, feature_names)
        Returns (None, None, None, None) if any file is missing.
    """
    try:
        model_data = load_artifact(MAIN_MODEL_PATH)
        scaler = load_artifact(MAIN_SCALER_PATH)
        label_encoder = load_artifact(MAIN_ENCODER_PATH)
        return (
            model_data["model"],
            scaler,
            label_encoder,
            model_data["feature_names"],
        )
    except FileNotFoundError as exc:
        logger.error("Main model files not found: %s", exc)
        return None, None, None, None


def load_specialist_model():
    """
    Load the specialized Basal / Luminal-B binary model.

    Returns
    -------
    tuple : (model, scaler, label_encoder, feature_names)
        Returns (None, None, None, None) if any file is missing.
    """
    try:
        model_data = load_artifact(SPEC_MODEL_PATH)
        scaler = load_artifact(SPEC_SCALER_PATH)
        label_encoder = load_artifact(SPEC_ENCODER_PATH)
        return (
            model_data["model"],
            scaler,
            label_encoder,
            model_data["feature_names"],
        )
    except FileNotFoundError as exc:
        logger.error("Specialist model files not found: %s", exc)
        return None, None, None, None


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

def validate_patient_data(
    patient_df: pd.DataFrame,
    required_features: list[str],
) -> list[str]:
    """
    Check that a patient DataFrame contains all required feature columns.

    Parameters
    ----------
    patient_df : pd.DataFrame
        Uploaded patient gene expression data.
    required_features : list[str]
        Feature names the model expects.

    Returns
    -------
    list[str]
        List of missing column names (empty if all present).
    """
    return sorted(set(required_features) - set(patient_df.columns))
