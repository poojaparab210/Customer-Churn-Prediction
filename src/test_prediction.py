
from pathlib import Path

import joblib
import pandas as pd


# 1. Define project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "outputs" / "models" / "churn_model.pkl"

# 2. Load the saved model
model = joblib.load(MODEL_PATH)

# 3. Create sample data for one new customer
customer = pd.DataFrame(
    [
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.50,
            "TotalCharges": 1026.00,
        }
    ]
)

# 4. Predict churn
prediction = model.predict(customer)[0]
churn_probability = model.predict_proba(customer)[0][1]

# 5. Display the result
print("Customer Churn Prediction")
print("-" * 30)

if prediction == 1:
    print("Prediction: Customer may churn")
else:
    print("Prediction: Customer is likely to stay")

print(f"Estimated churn probability: {churn_probability:.2%}")
