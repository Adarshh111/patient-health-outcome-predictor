from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "dataset" / "diabetic_data.csv"

df = pd.read_csv(DATASET_PATH)

print("\n==============================")
print("PATIENT HEALTH DATASET")
print("==============================")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
for column in df.columns:
    print(column)

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df["readmitted"].value_counts())

print("\nTarget Percentage:")
print(
    df["readmitted"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)