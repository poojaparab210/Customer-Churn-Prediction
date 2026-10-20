# Customer Churn Prediction & Customer Retention Copilot

A Data Science and Machine Learning project that predicts customer churn, identifies at-risk customers, and provides evidence-based retention guidance through an interactive dashboard and API-powered copilot.

The project combines machine learning, customer risk analysis, retrieval-augmented generation (RAG) techniques, and rule-based response generation to explore how churn predictions can support customer retention decisions.

## Project Overview

Customer churn can reduce revenue and affect long-term business growth. This project uses the Telco Customer Churn dataset to analyse customer behaviour, estimate churn probability, and identify customers who may need retention support.

The project includes two main components:

- **Customer Churn Prediction:** A Streamlit dashboard for exploring churn patterns, reviewing model performance, and predicting customer churn.
- **Customer Retention Copilot:** A FastAPI backend and Streamlit interface that retrieve relevant retention guidance, answer questions using retrieved evidence and rule-based logic, and analyse individual customer risk.

**Important:** The current copilot uses RAG-based retrieval and rule-based response generation. It does not currently use a live LLM to generate answers.

## Key Features

### Customer Churn Prediction
- Interactive dashboard for exploring customer churn patterns.
- Churn probability prediction using a trained Logistic Regression pipeline.
- Customer risk analysis based on predicted churn probability.
- Batch prediction support.
- Exploratory data analysis and feature importance.
- Model comparison and classification threshold tuning.

### Customer Retention Copilot
- **Retention Chat:** Ask questions about churn prevention and retention strategies.
- **Knowledge Retrieval:** Retrieve relevant information from a retention knowledge document using semantic search.
- **Customer Risk Analysis:** Analyse customer information and return a churn prediction with a risk category.
- **Evidence-Based Guidance:** Display retrieved sources and supporting evidence.
- **FastAPI Endpoints:** Expose prediction, retrieval, chat, and customer analysis functionality.
- **Automated API Tests:** Test core endpoints and invalid input handling.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development and machine learning |
| Pandas and NumPy | Data manipulation and preprocessing |
| Scikit-learn | Model training, evaluation, and prediction |
| Streamlit | Interactive dashboards and copilot interface |
| FastAPI | REST API development |
| Sentence Transformers | Text embeddings for semantic retrieval |
| ChromaDB | Vector storage and similarity search |
| Joblib | Saving and loading the trained model |
| Pytest and HTTPX | Automated API testing |
| Matplotlib, Seaborn, and Plotly | Data visualisation |

## Machine Learning Workflow

1. Data cleaning and preprocessing.
2. Exploratory Data Analysis (EDA).
3. Model training and comparison.
4. Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC.
5. Classification threshold tuning.
6. Customer risk segmentation.
7. Retention recommendation development.
8. Streamlit dashboard integration.

## Model Performance

The selected Logistic Regression model was evaluated on the held-out test dataset using a classification threshold of `0.60`.

| Metric | Score |
|---|---:|
| Accuracy | 76.22% |
| Precision | 53.99% |
| Recall | 70.59% |
| F1-score | 61.18% |
| ROC-AUC | 84.16% |

These metrics describe the model's performance on the evaluated test split. The selected threshold prioritises identifying more potential churners while balancing false positives. Results do not guarantee the same performance on new data.

## Customer Retention Copilot Architecture

1. **Customer input:** Submit a customer record or ask a retention-related question.
2. **Prediction:** Use the saved Logistic Regression pipeline to estimate churn probability.
3. **Knowledge retrieval:** Convert retention documents into embeddings and retrieve relevant passages using semantic similarity search.
4. **Response generation:** Apply rule-based logic to the retrieved evidence and customer information.
5. **Results:** Return churn risk information, retention guidance, retrieved sources, and supporting evidence through the API and interface.

The copilot's risk categories are application-defined thresholds, not calibrated business guarantees. Retrieved guidance may be incomplete, so recommendations should be reviewed before being used for real customer decisions.

## API Endpoints

The FastAPI application exposes the following endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /` | API welcome and status information |
| `GET /health` | Health check |
| `POST /predict` | Predict churn for a customer |
| `POST /retrieve` | Retrieve relevant retention knowledge |
| `POST /chat` | Answer retention questions using retrieval and rule-based logic |
| `POST /analyze-customer` | Combine churn prediction and retention evidence |

Interactive API documentation is available at `/docs` when the backend is running locally.

## Project Structure

```text
Customer-Churn-Prediction/
├── app.py
├── copilot_ui.py
├── requirements.txt
├── README.md
├── .gitignore
├── app/
│   └── api.py
├── rag/
│   ├── rag_pipeline.py
│   └── retention_docs/
│       └── retention_guide.txt
├── tests/
│   └── test_api.py
├── data/
├── src/
│   ├── data_cleaning.py
│   ├── data_exploration.py
│   ├── eda_analysis.py
│   ├── model_training.py
│   ├── model_comparison.py
│   ├── threshold_tuning.py
│   └── final_model_evaluation.py
└── outputs/
    ├── models/
    ├── model_evaluation/
    └── eda_charts/
```

Generated files and additional scripts may exist beyond the structure shown above. The ChromaDB vector database is generated locally and should not be committed.

## Getting Started

### Prerequisites

- Python 3.13 recommended for the current development environment.
- Git.
- A terminal such as PowerShell.

### 1. Clone the repository

```powershell
git clone https://github.com/poojaparab210/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the original churn dashboard

```powershell
python -m streamlit run app.py
```

### 5. Run the Customer Retention Copilot API

Open a second terminal in the project directory and activate the virtual environment. Then run:

```powershell
python -m uvicorn app.api:app --reload
```

Open `http://127.0.0.1:8000/docs` to explore and test the API endpoints.

### 6. Run the copilot interface

Open a third terminal, activate the virtual environment, and run:

```powershell
python -m streamlit run copilot_ui.py
```

Keep the API running in its own terminal while using the copilot interface.

### 7. Run automated API tests

With the required dependencies installed, run:

```powershell
python -m pytest tests/test_api.py -v
```

The current test suite has passed **7 API tests** in the development environment.

## Evaluation and Limitations

Initial manual checks covered general retention questions, contract-related guidance, and a question about an unspecified discount policy.

- Relevant retention information was retrieved for the tested questions.
- The copilot returned relevant guidance for a month-to-month customer scenario.
- The unsupported discount-policy question did not produce an invented percentage, but the response should explicitly state when the policy is unavailable.

The evaluation is preliminary and uses a small number of manual test cases. More testing is needed to assess retrieval quality, response correctness, robustness, and suitability for real-world use.

## What I Learned

- Data cleaning, preprocessing, and exploratory data analysis.
- Training and evaluating classification models.
- Interpreting precision, recall, F1-score, and ROC-AUC.
- Tuning classification thresholds for customer risk identification.
- Building interactive applications with Streamlit.
- Developing REST APIs with FastAPI.
- Implementing semantic search with embeddings and a vector database.
- Connecting model predictions with retrieved knowledge and rule-based guidance.
- Writing automated API tests and documenting evaluation limitations.

## Future Improvements

- Improve retrieval and response evaluation using a larger test set.
- Strengthen handling of questions that cannot be answered from the knowledge base.
- Compare retrieval strategies and tune retrieval parameters.
- Add more retention documents and test different customer scenarios.
- Explore LLM integration when a suitable API or local model is available.
- Deploy the application and evaluate performance with new data.
- Assess model calibration and the business impact of retention recommendations.

## Author

**Pooja Parab**  
B.Sc. Computer Science  
Aspiring Data Science and Machine Learning Professional

GitHub: [poojaparab210](https://github.com/poojaparab210)

