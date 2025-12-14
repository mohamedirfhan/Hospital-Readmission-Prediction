import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# -----------------------------
# Load trained model bundle
# -----------------------------
bundle = joblib.load("models/readmission_model.pkl")
model = bundle["model"]
FEATURES = bundle["features"]

# -----------------------------
# SHAP Explainer (cached)
# -----------------------------
@st.cache_resource
def load_shap_explainer(_model):
    return shap.Explainer(_model)

explainer = load_shap_explainer(model)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Hospital Readmission Risk",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Hospital Readmission Risk Prediction")
st.write(
    "Clinical decision support tool to identify patients at risk of "
    "**30-day hospital readmission**."
)

st.divider()

# -----------------------------
# Input Form
# -----------------------------
with st.form("patient_form"):
    age = st.slider("Age", 0, 100, 60)
    time_in_hospital = st.slider("Length of Stay (days)", 1, 14, 3)
    num_lab_procedures = st.slider("Lab Procedures", 0, 100, 30)
    num_procedures = st.slider("Procedures Performed", 0, 10, 1)
    num_medications = st.slider("Number of Medications", 0, 50, 10)

    number_outpatient = st.number_input("Past Outpatient Visits", 0, 50, 0)
    number_emergency = st.number_input("Past Emergency Visits", 0, 20, 0)
    number_inpatient = st.number_input("Past Inpatient Visits", 0, 20, 0)

    number_diagnoses = st.slider("Number of Diagnoses", 1, 10, 3)
    insulin = st.selectbox("Insulin Used", [0, 1])
    diabetesMed = st.selectbox("Diabetes Medication", [0, 1])

    submit = st.form_submit_button("🔍 Predict Readmission Risk")

# -----------------------------
# Prediction
# -----------------------------
if submit:
    input_df = pd.DataFrame([[
        age,
        time_in_hospital,
        num_lab_procedures,
        num_procedures,
        num_medications,
        number_outpatient,
        number_emergency,
        number_inpatient,
        number_diagnoses,
        insulin,
        diabetesMed
    ]], columns=FEATURES)

    # -----------------------------
    # SHAP Explanation
    # -----------------------------
    st.divider()
    st.subheader("🔍 Why this prediction?")

    shap_values = explainer(input_df)

    fig, ax = plt.subplots()
    shap.plots.waterfall(
        shap_values[0],
        max_display=8,
        show=False
    )
    st.pyplot(fig)


    probability = model.predict_proba(input_df)[0][1]
    risk_percent = round(probability * 100, 2)

    st.divider()
    st.subheader("📊 Risk Assessment Result")

    st.metric("Readmission Probability", f"{risk_percent} %")

    if probability >= 0.7:
        st.error("⚠ HIGH RISK: Immediate follow-up recommended")
    else:
        st.success("✅ LOW RISK: Standard discharge care sufficient")

    st.caption(
        "⚠ This tool supports clinical decisions and does not replace "
        "professional medical judgment."
    )