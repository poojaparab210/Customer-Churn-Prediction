
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Customer Retention Copilot",
    page_icon="💬",
    layout="wide",
)

st.title("Customer Retention Copilot")
st.caption(
    "Explore churn risk and retrieve evidence-based retention guidance."
)

chat_tab, analysis_tab = st.tabs(
    ["Retention Chat", "Customer Risk Analysis"]
)

with chat_tab:
    st.subheader("Ask a retention question")

    question = st.chat_input(
        "Example: What should we investigate for a high-risk customer?"
    )

    if question:
        with st.spinner("Retrieving retention guidance..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"question": question, "top_k": 3},
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()

                st.markdown("### Answer")
                st.write(data["answer"])

                st.markdown("### Sources")
                for source in data.get("sources", []):
                    st.write(f"- {source}")

                with st.expander("View retrieved evidence"):
                    for item in data.get("evidence", []):
                        st.caption(
                            f"Source: {item['source']} | "
                            f"Chunk: {item['chunk']} | "
                            f"Distance: {item['distance']}"
                        )
                        st.write(item["text"])

            except requests.RequestException as error:
                st.error(
                    "Could not reach the API. Make sure your "
                    "FastAPI server is running on port 8000."
                )
                st.caption(str(error))

with analysis_tab:
    st.subheader("Analyze a customer")

    with st.form("customer_form"):
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Has partner?", ["Yes", "No"])
        dependents = st.selectbox("Has dependents?", ["Yes", "No"])
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=1000, value=12
        )
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple lines", ["No", "Yes", "No phone service"]
        )
        internet = st.selectbox(
            "Internet service", ["Fiber optic", "DSL", "No"]
        )
        security = st.selectbox(
            "Online security", ["No", "Yes", "No internet service"]
        )
        backup = st.selectbox(
            "Online backup", ["No", "Yes", "No internet service"]
        )
        protection = st.selectbox(
            "Device protection", ["No", "Yes", "No internet service"]
        )
        support = st.selectbox(
            "Tech support", ["No", "Yes", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming movies", ["No", "Yes", "No internet service"]
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly = st.number_input(
            "Monthly charges", min_value=0.0, value=89.90
        )
        total = st.number_input(
            "Total charges", min_value=0.0, value=1078.80
        )

        submitted = st.form_submit_button("Analyze customer")

    if submitted:
        customer = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        with st.spinner("Analyzing customer..."):
            try:
                response = requests.post(
                    f"{API_URL}/analyze-customer",
                    json={
                        "customer": customer,
                        "question": (
                            "What should we investigate to help "
                            "retain this customer?"
                        ),
                        "top_k": 3,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()

                prediction = data["customer_prediction"]

                st.metric(
                    "Predicted churn probability",
                    f"{prediction['churn_probability_percent']:.2f}%",
                )
                st.write("**Prediction:**", prediction["result"])
                st.write("**Illustrative risk category:**", data["risk_level"])
                st.info(data["answer"])

                with st.expander("View evidence and sources"):
                    for item in data.get("evidence", []):
                        st.caption(
                            f"{item['source']} — chunk {item['chunk']}"
                        )
                        st.write(item["text"])

            except requests.RequestException as error:
                st.error(
                    "Could not reach the API. Check that FastAPI "
                    "is running and try again."
                )
                st.caption(str(error))
