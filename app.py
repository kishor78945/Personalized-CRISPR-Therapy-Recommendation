"""
Streamlit Web Application — Personalized CRISPR Therapy Recommendation System.

Upload a patient's gene expression CSV to:
  1. Predict the breast cancer subtype
  2. View affected genes and their roles
  3. Get CRISPR therapy recommendations (CRISPRi / CRISPRa)

Usage
-----
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

from config import GENE_THERAPY_MAP
from utils import load_main_model, validate_patient_data

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CRISPR Therapy Recommender",
    page_icon="🧬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧬 Personalized CRISPR Therapy Recommendation System")
st.markdown(
    "Upload a patient's breast cancer **gene expression profile** to classify "
    "the cancer subtype and receive targeted **CRISPR gene-editing** recommendations."
)
st.divider()

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

@st.cache_resource
def get_model():
    """Load and cache the trained ML model."""
    return load_main_model()


model, scaler, label_encoder, feature_names = get_model()

if model is None:
    st.error(
        "🚨 **Model files not found.** Please train the model first:\n\n"
        "```bash\npython train.py --data Breast_GSE45827.csv\n```"
    )
    st.stop()

st.success("✅ Model loaded successfully!", icon="🤖")

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
st.subheader("📤 Upload Patient Gene Expression Data")

st.info(
    "Upload a CSV file containing gene expression values. "
    "The file should **not** contain a `type` column. "
    f"The model expects **{len(feature_names)}** gene features.",
    icon="ℹ️",
)

patient_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    key="patient_upload",
)

if patient_file is None:
    st.stop()

# ---------------------------------------------------------------------------
# Process patient data
# ---------------------------------------------------------------------------
patient_df = pd.read_csv(patient_file)

if "type" in patient_df.columns:
    st.error("🚨 The uploaded file should **not** contain a `type` column. Please remove it and re-upload.")
    st.stop()

# Show preview
with st.expander("📋 Patient Data Preview", expanded=False):
    st.dataframe(patient_df.head(), use_container_width=True)
    st.caption(f"Shape: {patient_df.shape[0]} samples × {patient_df.shape[1]} features")

# Validate features
missing = validate_patient_data(patient_df, feature_names)
if missing:
    st.error(
        f"🚨 The uploaded file is missing **{len(missing)}** required gene features. "
        f"First few missing: `{', '.join(missing[:10])}`"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
patient_features = patient_df[feature_names]
patient_scaled = scaler.transform(patient_features)
predictions = model.predict(patient_scaled)
predicted_labels = label_encoder.inverse_transform(predictions)

st.divider()
st.subheader("🩺 Prediction Results")

# Display predictions
results_df = pd.DataFrame({
    "Sample": [f"Patient {i + 1}" for i in range(len(predicted_labels))],
    "Predicted Subtype": predicted_labels,
})
st.dataframe(results_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# CRISPR recommendations per subtype
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🧬 CRISPR Therapy Recommendations")

unique_subtypes = np.unique(predicted_labels)

for subtype in unique_subtypes:
    count = int(np.sum(predicted_labels == subtype))
    st.markdown(f"### {subtype}  ({count} sample{'s' if count > 1 else ''})")

    if subtype in GENE_THERAPY_MAP:
        genes_df = pd.DataFrame(GENE_THERAPY_MAP[subtype])

        # Color-code the recommendation column
        st.dataframe(
            genes_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "CRISPR Recommendation": st.column_config.TextColumn(
                    "CRISPR Recommendation",
                    help="CRISPRi = suppress oncogenes · CRISPRa = activate tumor suppressors",
                ),
            },
        )
    else:
        st.warning(f"⚠️ No gene therapy data available for subtype **{subtype}**.")

# ---------------------------------------------------------------------------
# CRISPR explainer
# ---------------------------------------------------------------------------
st.divider()
with st.expander("🧪 How does CRISPR gene editing work?", expanded=False):
    st.markdown("""
**CRISPR** (Clustered Regularly Interspaced Short Palindromic Repeats) enables
precise, targeted modifications to DNA. Two key strategies are used in cancer
therapy research:

| Strategy | Mechanism | Protein | Use Case |
|----------|-----------|---------|----------|
| **CRISPRi** | Blocks transcription | dCas9-KRAB | Silencing overexpressed **oncogenes** |
| **CRISPRa** | Boosts transcription | dCas9-VP64 | Reactivating **tumor suppressor** genes |

**Guide RNA (gRNA)** directs the CRISPR machinery to the exact gene of
interest. Delivery methods include viral vectors, lipid nanoparticles, and
electroporation.
    """)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Built with Streamlit · Model: Random Forest (scikit-learn) · "
    "Data: Breast Cancer Gene Expression CuMiDa"
)
