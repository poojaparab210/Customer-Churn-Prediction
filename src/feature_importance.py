
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "churn_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "outputs", "cleaned_churn.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "feature_importance")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the trained Random Forest model and cleaned data
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Remove target and ID columns from the input data
X = df.drop(columns=["Churn", "customerID"], errors="ignore")

# Get the preprocessing and classifier steps from the pipeline
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Extract Random Forest feature importance scores
importance_scores = classifier.feature_importances_

# Create a feature importance table
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_scores
})

importance_df = importance_df.sort_values(
    by="Importance", ascending=False
)

# Save all feature importance scores
csv_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
importance_df.to_csv(csv_path, index=False)

# Plot the top 15 features
top_features = importance_df.head(15).sort_values(
    by="Importance", ascending=True
)

plt.figure(figsize=(11, 7))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.title("Top 15 Features Influencing Churn Model Predictions")
plt.tight_layout()

chart_path = os.path.join(OUTPUT_DIR, "top_15_features.png")
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nFeature importance analysis completed!")
print("\nTop 15 features:")
print(importance_df.head(15).to_string(index=False))
print(f"\nCSV saved to: {csv_path}")
print(f"Chart saved to: {chart_path}")
