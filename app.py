import streamlit as st
import pandas as pd
import joblib
import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../model/hospital_readmission_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "../model/feature_names.pkl")

# --- Load model ---
if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Model not found! Please run main.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)

# --- Streamlit Page Setup ---
st.set_page_config(page_title="🏥 Hospital Readmission Prediction", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            color: #FF6F00;
            text-align: center;
            font-weight: bold;
        }
        .sub-title {
            color: #555;
            font-size: 18px;
            text-align: center;
        }
        .card {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        .stButton button {
            background-color: #FF6F00;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            height: 3em;
            width: 100%;
        }
        .stButton button:hover {
            background-color: #E65100;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🏥 Hospital Readmission Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Predict if a patient will be readmitted within 30 days based on hospital data</div><br>", unsafe_allow_html=True)

# --- Sidebar inputs ---
st.sidebar.header("🧍‍♂️ Enter Patient Details")

age = st.sidebar.slider("Age (years)", 0, 100, 45)
time_in_hospital = st.sidebar.slider("Days in Hospital", 1, 14, 4)
num_lab_procedures = st.sidebar.slider("Lab Procedures", 1, 150, 40)
num_medications = st.sidebar.slider("Number of Medications", 1, 50, 15)
number_diagnoses = st.sidebar.slider("Number of Diagnoses", 1, 10, 5)
A1Cresult = st.sidebar.selectbox("A1C Result", ["Normal", "Above Normal", "Below Normal"])
change = st.sidebar.selectbox("Medication Changed?", ["No", "Yes"])
diabetesMed = st.sidebar.selectbox("On Diabetes Medication?", ["No", "Yes"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
admission_type_id = st.sidebar.selectbox("Admission Type", ["Emergency", "Urgent", "Elective"])

# --- Prepare data ---
user_input = {
    "age": age,
    "time_in_hospital": time_in_hospital,
    "num_lab_procedures": num_lab_procedures,
    "num_medications": num_medications,
    "number_diagnoses": number_diagnoses,
    "A1Cresult": {"Normal": 0, "Above Normal": 1, "Below Normal": 2}[A1Cresult],
    "change": 1 if change == "Yes" else 0,
    "diabetesMed": 1 if diabetesMed == "Yes" else 0,
    "gender": 1 if gender == "Male" else 0,
    "admission_type_id": {"Emergency": 1, "Urgent": 2, "Elective": 3}[admission_type_id]
}
input_df = pd.DataFrame([user_input])

# Match feature order
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_names]

# --- Display summary ---
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📋 Patient Summary")
display_data = {
    "Age": f"{age} years",
    "Hospital Stay": f"{time_in_hospital} days",
    "Lab Procedures": num_lab_procedures,
    "Medications": num_medications,
    "Diagnoses": number_diagnoses,
    "A1C Result": A1Cresult,
    "Medication Changed": change,
    "On Diabetes Medication": diabetesMed,
    "Gender": gender,
    "Admission Type": admission_type_id,
}
st.table(pd.DataFrame([display_data]))
st.markdown("</div>", unsafe_allow_html=True)

# --- Predict ---
if st.button("🔍 Predict Readmission"):
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)[0][1] * 100

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk: Patient likely to be readmitted. Probability: **{proba:.2f}%**")
    else:
        st.success(f"✅ Low Risk: Patient unlikely to be readmitted. Probability: **{proba:.2f}%**")

    st.progress(int(proba))