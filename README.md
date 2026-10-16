# Customer Churn Prediction

A Data Science and Machine Learning project that predicts customer churn, identifies customers at risk of leaving, and provides data-driven insights to support customer retention.

The project combines machine learning, data analysis, and an interactive Streamlit dashboard to help explore customer behaviour and understand potential churn patterns.

## Project Overview

Customer churn is an important business problem because losing customers can affect revenue and long-term growth. This project uses the **Telco Customer Churn dataset** to analyse customer behaviour and predict the likelihood of churn.

The application allows users to explore churn patterns, make predictions for individual customers, analyse customer risk levels, and generate retention suggestions.

## Key Features

- **Interactive Dashboard** — Explore customer churn statistics and behavioural patterns.
- **Churn Prediction** — Estimate the likelihood of a customer leaving the service.
- **Customer Risk Analysis** — Categorise customers as low, medium, or high risk based on predicted churn probability.
- **Retention Recommendations** — Generate rule-based suggestions based on customer details and risk category.
- **Feature Importance** — Explore features associated with the model's predictions.
- **Batch Prediction** — Upload a CSV file containing multiple customers and download prediction results.
- **Model Evaluation** — Compare Logistic Regression and Random Forest using classification metrics.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data analysis and application development |
| Pandas and NumPy | Data manipulation and preprocessing |
| Scikit-learn | Model training, evaluation, and prediction |
| Plotly | Interactive visualisations |
| Streamlit | Interactive web application |
| Joblib | Saving and loading the trained model |
| Matplotlib and Seaborn | Exploratory data visualisation |

## Dataset

This project uses the **Telco Customer Churn dataset**, which contains customer information related to services, contracts, billing, and churn status.

- **Target column:** `Churn`
- **Objective:** Predict whether a customer is likely to leave the service.
- **Data preparation:** Cleaned and prepared the dataset before analysis and modelling.

## Project Workflow

1. **Data Cleaning** — Prepared the dataset for analysis and modelling.
2. **Exploratory Data Analysis (EDA)** — Investigated churn patterns and relationships between customer attributes and churn.
3. **Model Training** — Trained Logistic Regression and Random Forest models.
4. **Model Evaluation** — Compared models using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.
5. **Threshold Tuning** — Evaluated different classification thresholds and selected Logistic Regression for the prediction pipeline.
6. **Risk Segmentation** — Grouped customers according to their predicted churn probabilities.
7. **Retention Recommendations** — Added rule-based suggestions to support customer retention.
8. **Streamlit Dashboard** — Built an interactive application to bring the analysis and prediction features together.

## Model Performance

The selected Logistic Regression model was evaluated using a classification threshold of `0.60`.

| Metric | Score |
|---|---:|
| Accuracy | 76.22% |
| Precision | 53.99% |
| Recall | 70.59% |
| F1-score | 61.18% |
| ROC-AUC | 84.16% |

These results reflect the evaluation performed on the test dataset. Precision and recall provide additional insight into how the model identifies customers who may churn. The probability threshold affects the balance between identifying potential churners and producing false positives.

## Project Structure

```text
CustomerChurnProject/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── src/
│   ├── data_cleaning.py
│   ├── data_exploration.py
│   ├── eda_analysis.py
│   ├── feature_importance.py
│   ├── model_training.py
│   ├── model_comparison.py
│   ├── threshold_tuning.py
│   ├── final_model_evaluation.py
│   ├── train_production_model.py
│   ├── customer_risk_segmentation.py
│   ├── retention_recommendations.py
│   └── test_prediction.py
└── outputs/
    ├── cleaned_churn.csv
    ├── eda_charts/
    ├── feature_importance/
    ├── model_evaluation/
    ├── models/
    ├── customer_risk/
    └── retention/
```

*Note: The structure above highlights the main project files and folders; generated files may vary.*

## Getting Started

### Prerequisites

- Python installed
- Git installed
- Access to the project repository

### 1. Clone the repository

```bash
git clone https://github.com/poojaparab210/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
### 3. Install dependencies
```powershell
python -m pip install -r requirements.txt
```
### 4. Run the application
```powershell
python -m streamlit run app.py
```
Streamlit will provide a local URL to open the application in your browser.

## What I Learned

Through this project, I gained practical experience in:

- Data cleaning and preprocessing
- Exploratory data analysis and visualisation
- Training and comparing machine learning models
- Classification metrics and threshold tuning
- Customer risk segmentation
- Building an interactive application with Streamlit
- Presenting machine learning predictions in a business-oriented context

## Future Improvements

- Deploy the application online.
- Test the model with new customer data.
- Improve model performance and calibration.
- Explore more advanced retention strategies.
- Evaluate the business impact of retention recommendations.

## Author

**Pooja Parab**  
B.Sc. Computer Science  
Aspiring Data Science and Machine Learning Professional

