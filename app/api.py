from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag.rag_pipeline import retrieve_context

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "models" / "churn_model_logistic.pkl"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Customer Retention Copilot API",
    description="API for predicting customer churn.",
    version="1.0.0",
)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=1000)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/")
def home():
    return {"message": "Customer Retention Copilot API is running"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):
    customer_data = customer.model_dump()
    feature_names = model.feature_names_in_.tolist()

    customer_df = pd.DataFrame(
        [customer_data],
        columns=feature_names,
    )

    prediction = int(model.predict(customer_df)[0])
    probability = float(model.predict_proba(customer_df)[0][1])

    result = (
        "Customer may churn"
        if prediction == 1
        else "Customer is likely to stay"
    )

    return {
        "prediction": prediction,
        "result": result,
        "churn_probability": round(probability, 4),
        "churn_probability_percent": round(probability * 100, 2),
    }

class RetrievalQuestion(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=5)


@app.post("/retrieve")
def retrieve_knowledge(request: RetrievalQuestion):
    results = retrieve_context(
        question=request.question,
        top_k=request.top_k,
    )

    return {
        "question": request.question,
        "results": results,
        "sources": sorted(
            {item["source"] for item in results}
        ),
    }


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=5)


@app.post("/chat")
def chat(request: ChatRequest):
    # Retrieve relevant guidance from the existing RAG pipeline.
    results = retrieve_context(
        question=request.question,
        top_k=request.top_k,
    )

    if not results:
        return {
            "question": request.question,
            "answer": (
                "I could not find relevant guidance. "
                "Please consult the retention team."
            ),
            "sources": [],
            "evidence": [],
        }

    question = request.question.lower()
    recommendations = []

    # Provide guidance using simple keyword-based rules.
    if any(word in question for word in [
        "contract", "month-to-month", "month to month"
    ]):
        recommendations.append(
            "Investigate concerns about pricing, service quality, "
            "and contract flexibility. Explain options transparently "
            "without pressuring the customer."
        )

    if any(word in question for word in [
        "tenure", "new customer", "recently joined", "onboarding"
    ]):
        recommendations.append(
            "Check for unresolved setup or billing questions and "
            "offer appropriate onboarding guidance."
        )

    if any(word in question for word in [
        "technical", "support", "security", "online security"
    ]):
        recommendations.append(
            "Ask about the customer's actual technical or security "
            "concerns before recommending support services."
        )

    if any(word in question for word in [
        "bill", "billing", "payment", "charge", "price", "pricing"
    ]):
        recommendations.append(
            "Review unexpected charges, explain fees clearly, "
            "and discuss relevant payment options."
        )

    if any(word in question for word in [
        "feedback", "complaint", "concern", "dissatisfied"
    ]):
        recommendations.append(
            "Ask the customer about their main concern and record "
            "the action offered and its outcome."
        )

    if any(word in question for word in [
        "churn", "risk", "high-risk", "high risk", "prediction"
    ]):
        recommendations.append(
            "Treat the churn score as an estimate of risk, not proof "
            "of why the customer may leave. Investigate the customer's "
            "actual circumstances before choosing an action."
        )

    if not recommendations:
        recommendations.append(
            "Review the retrieved evidence below, ask the customer "
            "about their needs, and choose an appropriate action. "
            "The guide does not establish the customer's actual reason "
            "for leaving."
        )

    answer = (
        "Based on the retrieved example retention guide:\n\n"
        + "\n".join(
            f"{index}. {item}"
            for index, item in enumerate(recommendations, start=1)
        )
        + "\n\nThese are general suggestions, not verified company "
          "policies or guaranteed ways to prevent churn."
    )

    return {
        "question": request.question,
        "answer": answer,
        "sources": sorted({
            item["source"] for item in results
        }),
        "evidence": [
            {
                "text": item["text"],
                "source": item["source"],
                "chunk": item["chunk"],
                "distance": item["distance"],
            }
            for item in results
        ],
    }


class CustomerAnalysisRequest(BaseModel):
    customer: CustomerData
    question: str = Field(
        default="What should we investigate to help retain this customer?",
        min_length=3,
        max_length=1000,
    )
    top_k: int = Field(default=3, ge=1, le=5)


@app.post("/analyze-customer")
def analyze_customer(request: CustomerAnalysisRequest):
    # Step 1: Predict churn using the existing trained model.
    prediction_result = predict_churn(request.customer)

    churn_probability = prediction_result["churn_probability"]
    churn_percent = prediction_result["churn_probability_percent"]

    # Step 2: Create a retrieval query using the customer details.
    customer = request.customer

    retrieval_question = (
        f"{request.question} "
        f"Contract: {customer.Contract}. "
        f"Tenure: {customer.tenure} months. "
        f"Monthly charges: {customer.MonthlyCharges}. "
        f"Internet service: {customer.InternetService}. "
        f"Online security: {customer.OnlineSecurity}. "
        f"Tech support: {customer.TechSupport}. "
        f"Payment method: {customer.PaymentMethod}. "
        f"Predicted churn probability: {churn_percent}%."
    )

    # Step 3: Retrieve relevant retention guidance.
    evidence = retrieve_context(
        question=retrieval_question,
        top_k=request.top_k,
    )

    # Step 4: Generate a structured response without an LLM.
    if churn_probability >= 0.70:
        risk_level = "High"
    elif churn_probability >= 0.40:
        risk_level = "Moderate"
    else:
        risk_level = "Lower"

    answer_parts = [
        f"The model estimates a churn probability of {churn_percent}%.",
        f"The illustrative risk category is {risk_level}.",
        (
            "This prediction estimates risk; it does not establish "
            "the customer's reason for potentially leaving."
        ),
    ]

    if customer.Contract == "Month-to-month":
        answer_parts.append(
            "Review whether the customer has concerns about pricing, "
            "service quality, or contract flexibility."
        )

    if customer.tenure <= 12:
        answer_parts.append(
            "Consider checking onboarding, setup, and unresolved "
            "billing or service questions."
        )

    if customer.TechSupport == "No":
        answer_parts.append(
            "Ask about any technical issues before suggesting "
            "support options. Lack of a support subscription alone "
            "does not prove dissatisfaction."
        )

    if not evidence:
        answer_parts.append(
            "No supporting guide passages were retrieved. "
            "Consult the retention team before choosing an action."
        )

    answer = "\n\n".join(answer_parts)

    return {
        "customer_prediction": prediction_result,
        "risk_level": risk_level,
        "question": request.question,
        "answer": answer,
        "sources": sorted({
            item["source"] for item in evidence
        }),
        "evidence": evidence,
        "llm_used": False,
    }
