from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "patient_outcome_model.pkl"

# Flask application
app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return jsonify({
        "message": "Patient Health Outcome Predictor API is running!"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    patient = pd.DataFrame([{
        "race": data["race"],
        "gender": data["gender"],
        "age": data["age"],
        "admission_type_id": int(data["admission_type_id"]),
        "admission_source_id": int(data["admission_source_id"]),
        "time_in_hospital": int(data["time_in_hospital"]),
        "num_lab_procedures": int(data["num_lab_procedures"]),
        "num_procedures": int(data["num_procedures"]),
        "num_medications": int(data["num_medications"]),
        "number_outpatient": int(data["number_outpatient"]),
        "number_emergency": int(data["number_emergency"]),
        "number_inpatient": int(data["number_inpatient"]),
        "number_diagnoses": int(data["number_diagnoses"]),
        "A1Cresult": data["A1Cresult"],
        "insulin": data["insulin"],
        "change": data["change"],
        "diabetesMed": data["diabetesMed"]
    }])

    # Make prediction
    prediction = model.predict(patient)[0]

    # Probability of readmission within 30 days
    probability = model.predict_proba(patient)[0][1]

    if prediction == 1:
        result = "Readmitted within 30 days"
    else:
        result = "Not readmitted within 30 days"

    return jsonify({
        "prediction": int(prediction),
        "result": result,
        "probability": round(float(probability) * 100, 2)
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )