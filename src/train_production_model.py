
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "outputs" / "cleaned_churn.csv"
MODEL_DIR = BASE_DIR / "outputs" / "models"
MODEL_PATH = MODEL_DIR / "churn_model_logistic.pkl"


def main():
    print("Loading cleaned customer churn dataset...")

    df = pd.read_csv(DATA_PATH)

    # Remove customer ID because it is not a useful predictive feature
    columns_to_drop = ["Churn", "customerID"]
    X = df.drop(
        columns=[col for col in columns_to_drop if col in df.columns]
    )

    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected values found in the Churn column.")

    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Numerical preprocessing
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical preprocessing
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Build the final model pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training Logistic Regression on the full dataset...")

    model.fit(X, y)

    # Create model directory if it does not exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save trained model
    joblib.dump(model, MODEL_PATH)

    print("\nModel training completed successfully!")
    print(f"Dataset rows: {len(df)}")
    print(f"Features used: {X.shape[1]}")
    print(f"Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    main()
