import requests

url = "http://127.0.0.1:5000/predict"

patient_data = {
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
}

response = requests.post(url, json=patient_data)

print("\n==============================")
print("API TEST RESULT")
print("==============================")

print("Status Code:", response.status_code)
print("Response:")
print(response.json())