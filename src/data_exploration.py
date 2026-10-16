
from pathlib import Path

import pandas as pd

# Locate the project folder and dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def main():
    # Check whether the dataset exists
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Place the CSV file inside the project's data folder.")
        return

    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    # Display basic information
    print("\n--- FIRST FIVE CUSTOMERS ---")
    print(df.head())

    print("\n--- DATASET SHAPE ---")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n--- COLUMN DATA TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    print("\n--- CHURN DISTRIBUTION ---")
    print(df["Churn"].value_counts(dropna=False))

    print("\n--- SUMMARY STATISTICS ---")
    print(df.describe(include="all").T)


if __name__ == "__main__":
    main()
