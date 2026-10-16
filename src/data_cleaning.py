
from pathlib import Path

import pandas as pd

# Locate project files
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "cleaned_churn.csv"


def main():
    # 1. Check that the dataset exists
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    # 2. Load the original dataset
    df = pd.read_csv(DATA_PATH)

    print("Original dataset shape:", df.shape)

    # 3. Remove unnecessary spaces from text columns
    text_columns = df.select_dtypes(
        include=["object", "str"]
    ).columns

    for column in text_columns:
        df[column] = df[column].str.strip()

    # 4. Check for duplicate records
    duplicate_count = df.duplicated().sum()

    print("\nDuplicate rows:", duplicate_count)

    if duplicate_count > 0:
        df = df.drop_duplicates()

    # 5. Convert TotalCharges from text to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # 6. Inspect missing values after conversion
    print("\nMissing values after conversion:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # 7. Handle missing TotalCharges
    # Inspect these rows before deciding how to fill them.
    missing_total = df["TotalCharges"].isnull()

    print("\nRows with missing TotalCharges:")
    print(df.loc[
        missing_total,
        ["customerID", "tenure", "TotalCharges"]
    ])

    # In this dataset, blank TotalCharges corresponds to
    # customers with zero tenure and no accumulated charges.
    valid_zero_charge_rows = (
        missing_total & df["tenure"].eq(0)
    )

    df.loc[valid_zero_charge_rows, "TotalCharges"] = 0

    # Stop if any unexpected missing values remain.
    remaining_missing = df.isnull().sum()
    remaining_missing = remaining_missing[
        remaining_missing > 0
    ]

    if not remaining_missing.empty:
        print("\nPlease investigate remaining missing values:")
        print(remaining_missing)
        return

    # 8. Create the output folder if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 9. Save the cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n--- CLEANING COMPLETE ---")
    print("Cleaned dataset shape:", df.shape)
    print("Total missing values:", df.isnull().sum().sum())
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
