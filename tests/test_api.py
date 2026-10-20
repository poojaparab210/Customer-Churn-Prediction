
from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)

CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.90,
    "TotalCharges": 1078.80,
}


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_endpoint():
    response = client.post("/predict", json=CUSTOMER)

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert 0 <= result["churn_probability"] <= 1
    assert 0 <= result["churn_probability_percent"] <= 100


def test_retrieve_endpoint():
    response = client.post(
        "/retrieve",
        json={
            "question": "How can we retain customers on monthly contracts?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    result = response.json()
    assert "results" in result
    assert "sources" in result


def test_chat_endpoint():
    response = client.post(
        "/chat",
        json={
            "question": (
                "What should we investigate for a high-risk "
                "customer on a month-to-month contract?"
            ),
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    result = response.json()
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0
    assert "sources" in result


def test_analyze_customer_endpoint():
    response = client.post(
        "/analyze-customer",
        json={
            "customer": CUSTOMER,
            "question": "How can we help retain this customer?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    result = response.json()
    prediction = result["customer_prediction"]

    assert prediction["prediction"] in [0, 1]
    assert 0 <= prediction["churn_probability"] <= 1
    assert result["risk_level"] in ["High", "Moderate", "Lower"]
    assert isinstance(result["answer"], str)


def test_predict_rejects_invalid_customer():
    response = client.post("/predict", json={})

    assert response.status_code == 422
