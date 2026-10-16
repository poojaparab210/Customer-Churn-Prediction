
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "outputs", "cleaned_churn.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "model_evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["customerID"], errors="ignore")

# Encode target
y = df["Churn"].map({"No": 0, "Yes": 1})
X = df.drop(columns=["Churn"])

# Identify feature types
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features),
])

# Use one fixed, stratified test split for both models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

# Use the existing experiment's thresholds
thresholds = {
    "Logistic Regression": 0.60,
    "Random Forest": 0.50,
}

results = []
fitted_models = {}

for name, classifier in models.items():
    print(f"\nEvaluating {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    threshold = thresholds[name]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    result = {
        "Model": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, predictions),
        "Churn_Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Churn_Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Churn_F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp),
    }

    results.append(result)
    fitted_models[name] = pipeline

    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy: {result['Accuracy']:.4f}")
    print(f"Churn precision: {result['Churn_Precision']:.4f}")
    print(f"Churn recall: {result['Churn_Recall']:.4f}")
    print(f"Churn F1-score: {result['Churn_F1']:.4f}")
    print(f"ROC-AUC: {result['ROC_AUC']:.4f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["No Churn", "Churn"],
        zero_division=0,
    ))

# Save comparison table
results_df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "final_model_comparison.csv")
results_df.to_csv(csv_path, index=False)

print("\nFINAL COMPARISON")
print(results_df.round(4).to_string(index=False))
print(f"\nComparison saved to: {csv_path}")

print(
    "\nDecision guidance: compare churn recall, precision, F1, ROC-AUC, "
    "and false positives against false negatives. Select the model and "
    "threshold that best fit the business objective."
)
