"""
Generate synthetic test patient data for the CRISPR Therapy Recommendation System.

Creates a CSV file with random gene expression values that matches the
feature schema expected by the trained model. Useful for testing the
Streamlit app without real patient data.

Usage
-----
    python generate_test_data.py
"""

import os
import sys
import logging

import numpy as np
import pandas as pd

from config import MAIN_MODEL_PATH, RANDOM_STATE
from utils import load_artifact

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

NUM_SAMPLES = 5
OUTPUT_FILE = "test_patient_data.csv"


def main():
    # Load feature names from the trained model
    if not os.path.isfile(MAIN_MODEL_PATH):
        log.error(
            "Trained model not found at %s. Run train.py first.", MAIN_MODEL_PATH
        )
        sys.exit(1)

    model_data = load_artifact(MAIN_MODEL_PATH)
    feature_names = model_data["feature_names"]
    log.info("Model expects %d features", len(feature_names))

    # Generate random expression values in a realistic range
    np.random.seed(RANDOM_STATE)
    data = np.random.uniform(low=4.0, high=14.0, size=(NUM_SAMPLES, len(feature_names)))

    df = pd.DataFrame(data, columns=feature_names)
    df.to_csv(OUTPUT_FILE, index=False)

    log.info("✅ Generated %d synthetic patient samples → %s", NUM_SAMPLES, OUTPUT_FILE)
    log.info("Upload this file in the Streamlit app to test predictions.")


if __name__ == "__main__":
    main()
