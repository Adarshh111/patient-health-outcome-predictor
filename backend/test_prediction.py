from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "patient_outcome_model.pkl"

# Load trained model
model = joblib.load(MODEL_PATH)

# Sample patient
patient = pd.DataFrame([{
    "race": "Caucasian",
    "gender": "Male",
    "age": "[60-70)",
    "admission_type_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 5,
    "num_lab_procedures": 40,
    "num_procedures": 1,
    "num_medications": 15,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 8,
    "A1Cresult": "Norm",
    "insulin": "No",
    "change": "Ch",
    "diabetesMed": "Yes"
}])

# Make prediction
prediction = model.predict(patient)[0]
probability = model.predict_proba(patient)[0][1]

print("\n==============================")
print("PATIENT OUTCOME PREDICTION")
print("==============================")

if prediction == 1:
    print("Prediction: Readmitted within 30 days")
else:
    print("Prediction: Not readmitted within 30 days")

print(f"Readmission probability: {probability * 100:.2f}%")