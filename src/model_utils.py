from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

# -----------------------------
# Selected important features
# -----------------------------
IMPORTANT_FEATURES = [
    "age",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "insulin",
    "diabetesMed"
]

# -----------------------------
# Model training function
# -----------------------------
def train_model(df):
    X = df[IMPORTANT_FEATURES]
    y = df["readmitted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("📊 Model Evaluation")
    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))

    # Save model + feature list together
    joblib.dump(
        {
            "model": model,
            "features": IMPORTANT_FEATURES
        },
        "models/readmission_model.pkl"
    )

    print("✅ Model saved to models/readmission_model.pkl")