from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data.csv"
MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

# Replace ? with missing values
df = df.replace("?", pd.NA)


# --------------------------------------------------
# 3. Select features
# --------------------------------------------------

features = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "A1Cresult",
    "insulin",
    "change",
    "diabetesMed"
]

target = "readmitted"

df = df[features + [target]]


# --------------------------------------------------
# 4. Convert target into binary
# --------------------------------------------------

df[target] = df[target].apply(
    lambda x: 1 if x == "<30" else 0
)


# --------------------------------------------------
# 5. Separate input and target
# --------------------------------------------------

X = df[features]
y = df[target]


# --------------------------------------------------
# 6. Define categorical features
# --------------------------------------------------

categorical_features = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "admission_source_id",
    "A1Cresult",
    "insulin",
    "change",
    "diabetesMed"
]


# --------------------------------------------------
# 7. Define numerical features
# --------------------------------------------------

numerical_features = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses"
]


# --------------------------------------------------
# 8. Preprocessing for categorical data
# --------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


# --------------------------------------------------
# 9. Preprocessing for numerical data
# --------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# --------------------------------------------------
# 10. Combine preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        )
    ]
)


# --------------------------------------------------
# 11. Create Random Forest model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=30,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# --------------------------------------------------
# 12. Create complete ML pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# --------------------------------------------------
# 13. Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining the model...")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 14. Train model
# --------------------------------------------------

pipeline.fit(X_train, y_train)


print("\nModel training completed!")


# --------------------------------------------------
# 15. Make predictions
# --------------------------------------------------

y_pred = pipeline.predict(X_test)


# --------------------------------------------------
# 16. Calculate evaluation metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# --------------------------------------------------
# 17. Display results
# --------------------------------------------------

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# --------------------------------------------------
# 18. Save trained model
# --------------------------------------------------

model_path = MODEL_DIR / "patient_outcome_model.pkl"

joblib.dump(
    pipeline,
    model_path
)

print("\nModel saved successfully!")

print("Location:")
print(model_path)