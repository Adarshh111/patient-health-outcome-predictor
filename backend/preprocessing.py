from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data.csv"


def load_data():
    df = pd.read_csv(DATASET_PATH)

    # Replace ? with missing values
    df = df.replace("?", pd.NA)

    return df


def select_features(df):

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

    selected_columns = features + [target]

    df = df[selected_columns]

    # Convert readmission into binary target
    df["readmitted"] = df["readmitted"].apply(
        lambda x: 1 if x == "<30" else 0
    )

    return df


if __name__ == "__main__":

    df = load_data()

    print("Original dataset shape:")
    print(df.shape)

    df = select_features(df)

    print("\nSelected dataset shape:")
    print(df.shape)

    print("\nSelected columns:")

    for column in df.columns:
        print("-", column)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nTarget distribution:")
    print(df["readmitted"].value_counts())

    print("\nTarget percentage:")
    print(
        df["readmitted"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )