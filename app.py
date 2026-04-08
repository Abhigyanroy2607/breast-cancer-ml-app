# ──────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────
import streamlit as st
import numpy as np
import joblib
import os
from sklearn.datasets import load_breast_cancer

# ──────────────────────────────────────────────
# PAGE CONFIG (must be the first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(page_title="Breast Cancer Detection", page_icon="🧬")

# ──────────────────────────────────────────────
# CUSTOM STYLING
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Subtle card-style container for prediction results */
    .result-card {
        padding: 1.2rem;
        border-radius: 0.8rem;
        margin-top: 0.5rem;
    }
    /* Tighten slider spacing */
    .stSlider { margin-bottom: -0.4rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("About")
    st.info("""
    ML-based breast cancer prediction system.

    **Models evaluated:**
    - SVM
    - Random Forest
    - AdaBoost

    The best-performing model from training
    is loaded automatically.
    """)
    st.markdown("---")
    st.caption("Built with Streamlit · scikit-learn")

# ──────────────────────────────────────────────
# MODEL & SCALER LOADING
# ──────────────────────────────────────────────
model  = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Handle model name safely — use saved file if it exists, otherwise fallback
if os.path.exists("model_name.pkl"):
    model_display_name = joblib.load("model_name.pkl")
else:
    model_display_name = "SVM (Best Performing Model)"

# Load feature names from the dataset
data = load_breast_cancer()
feature_names = data.feature_names

# ──────────────────────────────────────────────
# TITLE & DESCRIPTION
# ──────────────────────────────────────────────
st.title("🧬 Breast Cancer Detection App")

st.markdown("""
This app predicts whether a tumor is **Benign** or **Malignant**  
using a trained Machine Learning model.
""")

# ──────────────────────────────────────────────
# TUMOR FEATURE INPUTS
# ──────────────────────────────────────────────
st.markdown("### 🧪 Tumor Feature Inputs")
st.subheader("Enter Tumor Details")

mean_radius     = st.slider("Mean Radius",     5.0,  30.0,   14.0)
mean_texture    = st.slider("Mean Texture",     5.0,  40.0,   19.0)
mean_perimeter  = st.slider("Mean Perimeter",  40.0, 200.0,   90.0)
mean_area       = st.slider("Mean Area",      200.0, 2500.0, 600.0)
mean_smoothness = st.slider("Mean Smoothness",  0.05,  0.2,    0.1)

# Assemble input array
inputs = [
    mean_radius,
    mean_texture,
    mean_perimeter,
    mean_area,
    mean_smoothness,
]

# ──────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────
predict_col, reset_col = st.columns([1, 1])

with predict_col:
    predict_clicked = st.button("🔍 Predict", use_container_width=True)

with reset_col:
    reset_clicked = st.button("🔄 Reset", use_container_width=True)

# Reset always works — it's outside the Predict block
if reset_clicked:
    st.rerun()

if predict_clicked:
    # Loading spinner during prediction
    with st.spinner("Analyzing tumor data..."):
        input_array  = np.array([inputs])
        input_scaled = scaler.transform(input_array)

        prediction  = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]

    # ── Model info ──
    st.write(f"🧠 Model Used: **{model_display_name}**")

    # ── Input summary (expandable) ──
    with st.expander("📊 View Input Summary"):
        st.write({
            "Mean Radius":     mean_radius,
            "Mean Texture":    mean_texture,
            "Mean Perimeter":  mean_perimeter,
            "Mean Area":       mean_area,
            "Mean Smoothness": mean_smoothness,
        })

    # ── Separator before results ──
    st.markdown("---")

    # ── Prediction result ──
    st.markdown("## 🧾 Prediction Result")

    if prediction[0] == 0:
        st.error("⚠️ Malignant Tumor Detected")
        confidence = 1 - float(probability)
    else:
        st.success("✅ Benign Tumor Detected")
        confidence = float(probability)

    # Single progress bar — confidence score
    st.write(f"**Confidence Score:** {confidence:.2%}")
    st.progress(confidence)

    # ── Medical disclaimer ──
    st.markdown("---")
    st.warning("⚠️ This is a machine learning prediction. Please consult a doctor for medical advice.")