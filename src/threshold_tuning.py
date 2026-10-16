
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Load the cleaned dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "outputs" / "cleaned_churn.csv"

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["customerID"])

X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 2. Identify feature types
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str", "category", "string"]
).columns.tolist()

# 3. Create preprocessing pipelines
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

# 4. Split off the final test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 5. Define Logistic Regression pipeline
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

# 6. Get cross-validated probabilities for training customers only
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

print("Generating cross-validated training probabilities...")

train_probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1,
)[:, 1]

# 7. Choose threshold with the best cross-validated F1-score
thresholds = [i / 100 for i in range(20, 71, 5)]
threshold_results = []

for threshold in thresholds:
    train_predictions = (
        train_probabilities >= threshold
    ).astype(int)

    threshold_results.append(
        {
            "Threshold": threshold,
            "Precision": precision_score(
                y_train, train_predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_train, train_predictions, zero_division=0
            ),
            "F1-score": f1_score(
                y_train, train_predictions, zero_division=0
            ),
        }
    )

threshold_df = pd.DataFrame(threshold_results)

print("\nCross-validated threshold results:")
print(threshold_df.round(4).to_string(index=False))

best_row = threshold_df.loc[threshold_df["F1-score"].idxmax()]
best_threshold = float(best_row["Threshold"])

print(f"\nSelected threshold: {best_threshold:.2f}")

# 8. Train on the full training set
print("\nTraining final Logistic Regression pipeline...")
model.fit(X_train, y_train)

# 9. Evaluate once on the held-out test set
test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\nHELD-OUT TEST RESULTS")
print("Threshold:", round(best_threshold, 2))
print("Accuracy:", round(accuracy_score(y_test, test_predictions), 4))
print(
    "Precision (Churn):",
    round(precision_score(y_test, test_predictions, zero_division=0), 4),
)
print(
    "Recall (Churn):",
    round(recall_score(y_test, test_predictions, zero_division=0), 4),
)
print(
    "F1-score (Churn):",
    round(f1_score(y_test, test_predictions, zero_division=0), 4),
)
print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, test_probabilities), 4),
)

# 10. Save experiment results
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "threshold_tuning_results.csv"
threshold_df.to_csv(RESULTS_PATH, index=False)

print(f"\nThreshold experiment saved to: {RESULTS_PATH}")
