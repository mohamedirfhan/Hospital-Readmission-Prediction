🏥 Hospital Readmission Prediction System

A production-ready machine learning application that predicts 30-day hospital readmission risk using patient admission and clinical data.
The system provides real-time predictions with explainable AI (SHAP) to support clinical decision-making.

📌 Problem Statement

Hospital readmissions within 30 days increase healthcare costs and indicate potential gaps in patient care.
This project aims to identify high-risk patients before discharge, enabling proactive follow-up and improved care planning.

🎯 Objective

Predict whether a patient is likely to be readmitted within 30 days

Provide interpretable risk explanations for clinicians

Deliver predictions through a real-time web interface

📊 Dataset

Source: UCI Machine Learning Repository

Dataset: Diabetes 130-US Hospitals (1999–2008)

Records: ~100,000 hospital encounters

Features: Demographics, admission details, diagnoses, procedures, medications

📌 The model is trained on historical clinical data, which reflects how real hospital ML systems operate.

🧠 Machine Learning Approach

Problem Type: Binary Classification

Target Variable:

1 → Readmitted within 30 days

0 → Not readmitted

Selected Key Features

Age group

Length of hospital stay

Number of diagnoses

Previous inpatient, outpatient, and emergency visits

Number of medications

Insulin usage

Diabetes medication usage

Model

Algorithm: XGBoost Classifier

Imbalance Handling: scale_pos_weight

Evaluation Metrics: Recall, Precision, ROC-AUC
(Recall prioritized due to healthcare risk sensitivity)

🔍 Explainable AI (SHAP)

To ensure clinical trust and transparency, SHAP (SHapley Additive exPlanations) is integrated:

Explains why a patient is predicted as high or low risk

Shows feature contributions for individual predictions

Helps clinicians understand key risk drivers

🖥️ Application Interface

Built using Streamlit

User-friendly clinical form

Risk probability with LOW / HIGH risk classification

SHAP waterfall plot for explainability

Designed as a clinical decision support tool
