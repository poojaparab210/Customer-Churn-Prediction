
import os
import pandas as pd
import joblib

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "churn_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "outputs", "cleaned_churn.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "customer_risk")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the saved model and cleaned customer data
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Keep customer IDs for the output report
customer_ids = (
    df["customerID"].copy()
    if "customerID" in df.columns
    else pd.Series(range(1, len(df) + 1), name="customerID")
)

# Prepare input features
X = df.drop(columns=["Churn", "customerID"], errors="ignore")

# Predict probability of churn
churn_probabilities = model.predict_proba(X)[:, 1]

# Build customer risk report
risk_df = pd.DataFrame({
    "customerID": customer_ids,
    "ChurnProbability": churn_probabilities
})

# Assign risk groups using defined probability bands
def assign_risk(probability):
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.60:
        return "Medium Risk"
    else:
        return "High Risk"

risk_df["RiskCategory"] = risk_df["ChurnProbability"].apply(assign_risk)
risk_df["ChurnProbability"] = (
    risk_df["ChurnProbability"] * 100
).round(2)

# Sort high-risk customers first
risk_df = risk_df.sort_values(
    by="ChurnProbability", ascending=False
)

# Save report
output_path = os.path.join(OUTPUT_DIR, "customer_risk_report.csv")
risk_df.to_csv(output_path, index=False)

# Display summary
print("\nCustomer risk segmentation completed!")
print("\nCustomers in each risk category:")
print(risk_df["RiskCategory"].value_counts().to_string())

print("\nTop 10 customers by predicted churn probability:")
print(risk_df.head(10).to_string(index=False))

print(f"\nFull report saved to: {output_path}")
