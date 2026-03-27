"""
Configuration module for the Personalized CRISPR Therapy Recommendation System.

Centralizes all hyperparameters, file paths, and gene therapy reference data.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Main 6-class model artifacts
MAIN_MODEL_PATH = os.path.join(MODELS_DIR, "trained_model.pkl")
MAIN_SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
MAIN_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# Specialized Basal / Luminal-B model artifacts
SPEC_MODEL_PATH = os.path.join(MODELS_DIR, "model_basal_luminalB.pkl")
SPEC_SCALER_PATH = os.path.join(MODELS_DIR, "scaler_basal_luminalB.pkl")
SPEC_ENCODER_PATH = os.path.join(MODELS_DIR, "encoder_basal_luminalB.pkl")

# Cross-validation results
CV_RESULTS_PATH = os.path.join(MODELS_DIR, "cross_validation_results.csv")

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
NUM_TOP_FEATURES = 500          # Top-K genes selected via ANOVA F-test

MAIN_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 8,
    "min_samples_leaf": 4,
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

SPEC_RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Gene–Therapy Mapping
# ---------------------------------------------------------------------------
# Each subtype maps to a list of dicts with consistent keys:
#   Gene, Function, Effect, CRISPR Recommendation
GENE_THERAPY_MAP = {
    "HER": [
        {"Gene": "ERBB2",  "Function": "Encodes HER2 receptor",     "Effect": "Overexpressed in HER2+ cancers",    "CRISPR Recommendation": "Suppress with CRISPRi"},
        {"Gene": "GRB7",   "Function": "Growth signaling protein",   "Effect": "Drives aggressive tumor growth",    "CRISPR Recommendation": "Suppress with CRISPRi"},
        {"Gene": "PIK3CA", "Function": "Cell signaling enzyme",      "Effect": "Mutated in HER2-positive cases",    "CRISPR Recommendation": "Suppress with CRISPRi"},
        {"Gene": "FOXA1",  "Function": "Transcription factor",       "Effect": "Regulates HER2 gene expression",    "CRISPR Recommendation": "Suppress with CRISPRi"},
        {"Gene": "PTEN",   "Function": "Tumor suppressor",           "Effect": "Loss leads to HER2 activation",     "CRISPR Recommendation": "Activate with CRISPRa"},
        {"Gene": "AKT1",   "Function": "Serine/threonine kinase",    "Effect": "Enhances tumor cell survival",      "CRISPR Recommendation": "Suppress with CRISPRi"},
        {"Gene": "HER3",   "Function": "HER family receptor",        "Effect": "Heterodimerizes with HER2",         "CRISPR Recommendation": "Suppress with CRISPRi"},
        {"Gene": "VEGFA",  "Function": "Angiogenesis factor",        "Effect": "Promotes tumor blood supply",       "CRISPR Recommendation": "Suppress with CRISPRi"},
        {"Gene": "CDH1",   "Function": "Cell adhesion molecule",     "Effect": "Loss disrupts cell adhesion",       "CRISPR Recommendation": "Activate with CRISPRa"},
        {"Gene": "RB1",    "Function": "Cell cycle checkpoint",      "Effect": "Loss removes growth checkpoint",    "CRISPR Recommendation": "Activate with CRISPRa"},
    ],
    "luminal_A": [
        {"Gene": "ESR1",   "Function": "Estrogen receptor",          "Effect": "Promotes estrogen-dependent growth", "CRISPR Recommendation": "Suppress with CRISPRi"},
        {"Gene": "PGR",    "Function": "Progesterone receptor",      "Effect": "Regulates hormone response",        "CRISPR Recommendation": "Suppress with CRISPRi"},
        {"Gene": "GATA3",  "Function": "Transcription factor",       "Effect": "Regulates luminal differentiation", "CRISPR Recommendation": "Activate with CRISPRa"},
        {"Gene": "AR",     "Function": "Androgen receptor",          "Effect": "Involved in hormonal response",     "CRISPR Recommendation": "Suppress with CRISPRi"},
        {"Gene": "BRCA2",  "Function": "DNA repair protein",         "Effect": "Deficiency increases cancer risk",  "CRISPR Recommendation": "Activate with CRISPRa"},
        {"Gene": "FOXA1",  "Function": "Transcription factor",       "Effect": "Regulates estrogen signaling",      "CRISPR Recommendation": "Activate with CRISPRa"},
    ],
    "luminal_B": [
        {"Gene": "CCND1",  "Function": "Cell cycle regulator",       "Effect": "Overactive in Luminal B",           "CRISPR Recommendation": "Suppress with CRISPRi"},
        {"Gene": "FOXA1",  "Function": "Transcription factor",       "Effect": "Regulates estrogen response",       "CRISPR Recommendation": "Suppress with CRISPRi"},
        {"Gene": "MYB",    "Function": "Proliferation regulator",    "Effect": "Upregulated in Luminal B",          "CRISPR Recommendation": "Suppress with CRISPRi"},
        {"Gene": "BRCA1",  "Function": "DNA repair protein",         "Effect": "Deficiency can lead to cancer",     "CRISPR Recommendation": "Activate with CRISPRa"},
        {"Gene": "CDK4",   "Function": "Cell cycle kinase",          "Effect": "Promotes uncontrolled division",    "CRISPR Recommendation": "Suppress with CRISPRi"},
    ],
    "basal": [
        {"Gene": "TP53",   "Function": "Tumor suppressor",           "Effect": "Mutated in basal cancers",          "CRISPR Recommendation": "Activate with CRISPRa"},
        {"Gene": "EGFR",   "Function": "Growth factor receptor",     "Effect": "Drives basal-like breast cancer",   "CRISPR Recommendation": "Suppress with CRISPRi"},
        {"Gene": "RB1",    "Function": "Cell cycle regulator",       "Effect": "Lost in basal subtypes",            "CRISPR Recommendation": "Activate with CRISPRa"},
        {"Gene": "BRCA1",  "Function": "DNA repair protein",         "Effect": "Loss leads to basal cancer",        "CRISPR Recommendation": "Activate with CRISPRa"},
        {"Gene": "MMP9",   "Function": "Matrix metalloproteinase",   "Effect": "Drives metastasis",                 "CRISPR Recommendation": "Suppress with CRISPRi"},
    ],
    "normal": [
        {"Gene": "GATA3",  "Function": "Transcription factor",       "Effect": "Maintains normal tissue identity",  "CRISPR Recommendation": "Activate with CRISPRa"},
        {"Gene": "BRCA1",  "Function": "DNA repair protein",         "Effect": "Prevents genetic damage",           "CRISPR Recommendation": "Activate with CRISPRa"},
        {"Gene": "FOXA1",  "Function": "Transcription factor",       "Effect": "Regulates hormonal genes",          "CRISPR Recommendation": "Activate with CRISPRa"},
    ],
    "cell_line": [
        {"Gene": "MYC",    "Function": "Oncogene",                   "Effect": "Drives cell proliferation",         "CRISPR Recommendation": "Suppress with CRISPRi"},
        {"Gene": "TERT",   "Function": "Telomerase enzyme",          "Effect": "Maintains unlimited growth",        "CRISPR Recommendation": "Suppress with CRISPRi"},
        {"Gene": "CDK4",   "Function": "Cell cycle kinase",          "Effect": "Promotes rapid division",           "CRISPR Recommendation": "Suppress with CRISPRi"},
        {"Gene": "CCNE1",  "Function": "Cell cycle regulator",       "Effect": "Amplified in aggressive cancers",   "CRISPR Recommendation": "Suppress with CRISPRi"},
    ],
}
