
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "outputs" / "cleaned_churn.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "eda_charts"

# Consistent chart styling
sns.set_theme(style="whitegrid")


def main():
    if not DATA_PATH.exists():
        print("Cleaned dataset not found. Run data_cleaning.py first.")
        return

    # Load cleaned data
    df = pd.read_csv(DATA_PATH)

    # Create folder for charts
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Dataset shape:", df.shape)

    # ---------------------------------------
    # Chart 1: Overall churn distribution
    # ---------------------------------------
    churn_counts = df["Churn"].value_counts()

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        x=churn_counts.index,
        y=churn_counts.values
    )

    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn Status")
    plt.ylabel("Number of Customers")

    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "churn_distribution.png", dpi=150)
    plt.close()

    # ---------------------------------------
    # Chart 2: Churn rate by contract type
    # ---------------------------------------
    contract_churn = (
        df.groupby("Contract", observed=True)["Churn"]
        .apply(lambda values: (values == "Yes").mean() * 100)
        .sort_values(ascending=False)
        .reset_index(name="ChurnRate")
    )

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=contract_churn,
        x="Contract",
        y="ChurnRate"
    )

    plt.title("Churn Rate by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=15)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "churn_by_contract.png", dpi=150)
    plt.close()

    # ---------------------------------------
    # Chart 3: Customer tenure distribution
    # ---------------------------------------
    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x="tenure",
        hue="Churn",
        bins=24,
        multiple="layer",
        element="step"
    )

    plt.title("Customer Tenure Distribution by Churn")
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Number of Customers")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tenure_distribution.png", dpi=150)
    plt.close()

    # ---------------------------------------
    # Chart 4: Monthly charges by churn status
    # ---------------------------------------
    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=df,
        x="Churn",
        y="MonthlyCharges"
    )

    plt.title("Monthly Charges by Churn Status")
    plt.xlabel("Churn Status")
    plt.ylabel("Monthly Charges")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monthly_charges_by_churn.png", dpi=150)
    plt.close()

    # ---------------------------------------
    # Print useful business summaries
    # ---------------------------------------
    overall_churn_rate = (df["Churn"] == "Yes").mean() * 100

    print(f"\nOverall churn rate: {overall_churn_rate:.2f}%")

    print("\nChurn rate by contract type:")
    print(contract_churn.to_string(index=False))

    print("\nAverage tenure by churn status:")
    print(df.groupby("Churn")["tenure"].mean().round(2))

    print("\nAverage monthly charges by churn status:")
    print(df.groupby("Churn")["MonthlyCharges"].mean().round(2))

    print(f"\nCharts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
