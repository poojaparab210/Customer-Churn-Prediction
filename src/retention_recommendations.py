
import os
import pandas as pd

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "outputs", "cleaned_churn.csv")
RISK_PATH = os.path.join(
    BASE_DIR, "outputs", "customer_risk", "customer_risk_report.csv"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "retention")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "retention_recommendations.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load customer data and predicted risk
customers = pd.read_csv(DATA_PATH)
risk = pd.read_csv(RISK_PATH)

# Match each customer to their risk report
df = customers.merge(risk, on="customerID", how="left")

def generate_recommendations(row):
    recommendations = []

    # Prioritize customers with high predicted risk
    if row["RiskCategory"] == "High Risk":
        recommendations.append(
            "Prioritize for a retention review and personalized outreach"
        )
    elif row["RiskCategory"] == "Medium Risk":
        recommendations.append(
            "Monitor customer experience and consider a proactive check-in"
        )
    else:
        recommendations.append(
            "Maintain service quality and monitor for changes in risk"
        )

    # Contract-related suggestion
    if row.get("Contract") == "Month-to-month":
        recommendations.append(
            "Offer information about suitable longer-term contract options"
        )

    # Service and support suggestions
    if row.get("OnlineSecurity") == "No":
        recommendations.append(
            "Explain available online security options and their pricing"
        )

    if row.get("TechSupport") == "No":
        recommendations.append(
            "Share available technical support options"
        )

    # Payment-related suggestion
    if row.get("PaymentMethod") == "Electronic check":
        recommendations.append(
            "Check whether the customer prefers other convenient payment methods"
        )

    # Charges-related suggestion
    if pd.notna(row.get("MonthlyCharges")) and row["MonthlyCharges"] >= 75:
        recommendations.append(
            "Review the customer's plan and explain available pricing options"
        )

    return "; ".join(recommendations)

# Generate recommendations for every customer
df["RetentionRecommendations"] = df.apply(
    generate_recommendations, axis=1
)

# Keep useful fields in the output
output_columns = [
    "customerID",
    "ChurnProbability",
    "RiskCategory",
    "Contract",
    "tenure",
    "MonthlyCharges",
    "RetentionRecommendations",
]

output_df = df[output_columns].sort_values(
    by="ChurnProbability", ascending=False
)

output_df.to_csv(OUTPUT_PATH, index=False)

print("\nRetention recommendations generated successfully!")
print(f"Customers processed: {len(output_df)}")
print("\nSample recommendations:")
print(
    output_df[
        ["customerID", "RiskCategory", "RetentionRecommendations"]
    ].head(5).to_string(index=False)
)
print(f"\nReport saved to: {OUTPUT_PATH}")
