
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "outputs" / "cleaned_churn.csv"
MODEL_PATH = BASE_DIR / "outputs" / "models" / "churn_model_logistic.pkl"
FEATURE_IMPORTANCE_PATH = (
    BASE_DIR / "outputs" / "feature_importance" / "feature_importance.csv"
)
MODEL_THRESHOLD = 0.60

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)

# ---------------- CUSTOM THEME: LIGHT + DARK MODE ----------------
st.markdown(
    """
    <style>
    /* Use Streamlit's active theme colors */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    /* Main headings */
    h1 {
        color: var(--primary-color) !important;
        font-weight: 800 !important;
    }

    h2 {
        color: var(--primary-color) !important;
        font-weight: 700 !important;
    }

    h3 {
        color: var(--primary-color) !important;
        font-weight: 700 !important;
    }

    /* General text and labels */
    .stApp p,
    .stApp label,
    .stApp li,
    .stApp span {
        color: var(--text-color);
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid var(--primary-color);
        border-top: 4px solid var(--primary-color);
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(100, 80, 160, 0.10);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-color) !important;
    }

    div[data-testid="stMetricValue"] {
        color: var(--primary-color) !important;
        font-weight: 800 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text-color);
    }

    /* Buttons */
    .stButton > button,
    .stFormSubmitButton > button {
        background-color: var(--primary-color);
        color: white !important;
        border: 1px solid var(--primary-color);
        border-radius: 9px;
        font-weight: 700;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        filter: brightness(0.90);
        color: white !important;
    }

    /* Input fields and dropdowns */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"],
    .stMultiSelect div[data-baseweb="select"],
    .stTextArea textarea {
        border-radius: 8px;
    }

    /* Tables and dataframes */
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        border: 1px solid var(--primary-color);
        border-radius: 10px;
        overflow: hidden;
    }

    /* Progress bars */
    div[data-testid="stProgressBar"] > div > div {
        background-color: #16B8A6;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* File uploader */
    section[data-testid="stFileUploader"] {
        border-radius: 10px;
    }

    /* Links */
    .stApp a {
        color: var(--primary-color);
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 2. LOAD DATA AND MODEL
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
    model = load_model()
except FileNotFoundError as error:
    st.error(f"Required project file not found: {error}")
    st.stop()


if "Churn" not in df.columns:
    st.error("The cleaned dataset must contain a Churn column.")
    st.stop()


# Prepare the feature columns used by the model
DROP_COLUMNS = ["Churn", "customerID"]
FEATURE_COLUMNS = [
    column for column in df.columns if column not in DROP_COLUMNS
]

if "customerID" in df.columns:
    customer_ids = df["customerID"].astype(str)
else:
    customer_ids = pd.Series(
        [f"Customer_{i + 1}" for i in range(len(df))]
    )


# --------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------

def predict_customers(input_df):
    """Return churn probabilities and predictions."""

    X = input_df[FEATURE_COLUMNS].copy()

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= MODEL_THRESHOLD).astype(int)

    results = input_df.copy()
    results["ChurnProbability"] = probabilities
    results["ChurnPrediction"] = [
        "Yes" if prediction == 1 else "No"
        for prediction in predictions
    ]

    results["RiskCategory"] = pd.cut(
        probabilities,
        bins=[-0.000001, 0.30, MODEL_THRESHOLD, 1.000001],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        right=False,
    ).astype(str)

    return results


def recommend_retention(row):
    """Generate simple rule-based retention suggestions."""

    recommendations = []

    if row["RiskCategory"] == "High Risk":
        recommendations.append("Prioritize personal retention outreach")
    elif row["RiskCategory"] == "Medium Risk":
        recommendations.append("Send a satisfaction check-in")
    else:
        recommendations.append("Continue regular engagement")

    if "Contract" in row.index:
        if row["Contract"] == "Month-to-month":
            recommendations.append("Offer a longer-term contract incentive")

    if "OnlineSecurity" in row.index:
        if row["OnlineSecurity"] == "No":
            recommendations.append("Explain available online security options")

    if "TechSupport" in row.index:
        if row["TechSupport"] == "No":
            recommendations.append("Offer a technical support plan")

    if "PaymentMethod" in row.index:
        if row["PaymentMethod"] == "Electronic check":
            recommendations.append("Offer convenient alternative payment options")

    if "MonthlyCharges" in row.index:
        if row["MonthlyCharges"] >= 75:
            recommendations.append("Review plan value and suitable pricing options")

    return "; ".join(recommendations)


# Calculate predictions for the existing customer dataset
risk_df = predict_customers(df)

risk_df.insert(
    0,
    "CustomerID",
    customer_ids.reset_index(drop=True),
)

risk_df["RetentionRecommendations"] = risk_df.apply(
    recommend_retention,
    axis=1,
)


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("📊 Customer Churn")
st.sidebar.caption("Machine Learning Analytics Dashboard")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Predict Churn",
        "Customer Risk",
        "Retention Recommendations",
        "Feature Importance",
        "Batch Predictions",
        "Project Overview",
    ],
)

st.sidebar.divider()
st.sidebar.metric("Prediction Threshold", f"{MODEL_THRESHOLD:.0%}")
st.sidebar.caption(
    "Predictions are model estimates, not guarantees of future churn."
)

# --------------------------------------------------
# 5. DASHBOARD PAGE
# --------------------------------------------------
if page == "Dashboard":

    st.title("📊 Customer Churn Dashboard")
    st.write(
        "Explore customer churn patterns, customer risk, "
        "and important business indicators."
    )

    # ---------------- KPI METRICS ----------------
    total_customers = len(df)

    actual_churn_count = (df["Churn"] == "Yes").sum()

    actual_churn_rate = (
        actual_churn_count / total_customers * 100
        if total_customers
        else 0
    )

    high_risk_count = (
        risk_df["RiskCategory"] == "High Risk"
    ).sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Customers Who Churned", f"{actual_churn_count:,}")
    col3.metric("Historical Churn Rate", f"{actual_churn_rate:.2f}%")
    col4.metric("High-Risk Estimates", f"{high_risk_count:,}")

    st.divider()

    # ---------------- CHART 1 & 2 ----------------
    left, right = st.columns(2)

    with left:
        churn_counts = (
            df["Churn"]
            .value_counts()
            .rename_axis("Churn")
            .reset_index(name="Customers")
        )

        # CHART 1: HISTORICAL CUSTOMER CHURN
        fig = px.pie(
            churn_counts,
            names="Churn",
            values="Customers",
            title="Historical Customer Churn",
            hole=0.4,
            color="Churn",
            color_discrete_map={
                "Yes": "#FF718B",
                "No": "#16B8A6",
            },
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(
                line=dict(color="white", width=2)
            ),
        )

        fig.update_layout(
            title_x=0.02,
            legend_title_text="Churn Status",
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        if "Contract" in df.columns:
            contract_churn = (
                df.groupby("Contract", observed=True)["Churn"]
                .apply(lambda values: (values == "Yes").mean() * 100)
                .reset_index(name="ChurnRate")
            )

            # CHART 2: CHURN RATE BY CONTRACT
            fig = px.bar(
                contract_churn,
                x="Contract",
                y="ChurnRate",
                title="Historical Churn Rate by Contract",
                labels={"ChurnRate": "Churn Rate (%)"},
                text_auto=".2f",
                color="ChurnRate",
                color_continuous_scale=[
                    "#C4B5FD",
                    "#6554DE",
                    "#312E81",
                ],
            )

            fig.update_traces(
                texttemplate="%{y:.2f}%",
                textposition="outside",
                marker_line_color="white",
                marker_line_width=1,
            )

            fig.update_layout(
                title_x=0.02,
                coloraxis_showscale=False,
                yaxis_title="Churn Rate (%)",
            )

            st.plotly_chart(fig, use_container_width=True)

    # ---------------- CHART 3 & 4 ----------------
    left, right = st.columns(2)

    with left:
        if "tenure" in df.columns:
            # CHART 3: CUSTOMER TENURE DISTRIBUTION
            fig = px.histogram(
                df,
                x="tenure",
                color="Churn",
                nbins=30,
                title="Customer Tenure Distribution",
                labels={"tenure": "Tenure (months)"},
                color_discrete_map={
                    "Yes": "#FF718B",
                    "No": "#16B8A6",
                },
                opacity=0.85,
            )

            fig.update_layout(
                title_x=0.02,
                xaxis_title="Tenure (months)",
                yaxis_title="Number of Customers",
                legend_title_text="Churn Status",
                bargap=0.08,
            )

            st.plotly_chart(fig, use_container_width=True)

    with right:
        if "MonthlyCharges" in df.columns:
            # CHART 4: MONTHLY CHARGES BY CHURN
            fig = px.box(
                df,
                x="Churn",
                y="MonthlyCharges",
                color="Churn",
                title="Monthly Charges by Historical Churn",
                color_discrete_map={
                    "Yes": "#FF718B",
                    "No": "#16B8A6",
                },
                points="outliers",
            )

            fig.update_layout(
                title_x=0.02,
                xaxis_title="Historical Churn",
                yaxis_title="Monthly Charges",
                legend_title_text="Churn Status",
            )

            st.plotly_chart(fig, use_container_width=True)

    # ---------------- CHART 5: CUSTOMER RISK ----------------
    st.subheader("Customer Risk Distribution")

    risk_counts = (
        risk_df["RiskCategory"]
        .value_counts()
        .reindex(
            ["Low Risk", "Medium Risk", "High Risk"],
            fill_value=0,
        )
        .rename_axis("Risk Category")
        .reset_index(name="Customers")
    )

    fig = px.bar(
        risk_counts,
        x="Risk Category",
        y="Customers",
        title="Estimated Risk Categories",
        text="Customers",
        color="Risk Category",
        color_discrete_map={
            "Low Risk": "#16B8A6",
            "Medium Risk": "#F5A623",
            "High Risk": "#FF718B",
        },
        category_orders={
            "Risk Category": [
                "Low Risk",
                "Medium Risk",
                "High Risk",
            ]
        },
    )

    fig.update_traces(
        textposition="outside",
        marker_line_color="white",
        marker_line_width=1,
    )

    fig.update_layout(
        title_x=0.02,
        xaxis_title="Risk Category",
        yaxis_title="Number of Customers",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Historical churn charts show observed outcomes. Risk charts show "
        "model estimates using the selected threshold."
    )


# --------------------------------------------------
# 6. PREDICT CHURN PAGE
# --------------------------------------------------

elif page == "Predict Churn":
    st.title("🔮 Predict Customer Churn")
    st.write(
        "Enter customer details below to estimate the probability "
        "that the customer will leave the service."
    )

    st.info(
        "The model uses a 0.60 probability threshold to classify "
        "a customer as likely to churn."
    )

    with st.form("predict_churn_form"):
        st.subheader("Customer Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"],
            )

            senior_citizen = st.selectbox(
                "Senior Citizen",
                ["No", "Yes"],
            )

            partner = st.selectbox(
                "Has Partner?",
                ["Yes", "No"],
            )

            dependents = st.selectbox(
                "Has Dependents?",
                ["Yes", "No"],
            )

            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=12,
                step=1,
            )

        with col2:
            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"],
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"],
            )

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"],
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"],
            )

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"],
            )

        with col3:
            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"],
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"],
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"],
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"],
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        st.subheader("Billing Information")

        bill_col1, bill_col2 = st.columns(2)

        with bill_col1:
            monthly_charges = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0,
            )

        with bill_col2:
            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                value=float(monthly_charges * tenure),
                step=10.0,
            )

        submitted = st.form_submit_button(
            "Predict Churn",
            use_container_width=True,
        )

    if submitted:
        # Adjust service values to match the dataset's conventions.
        if phone_service == "No":
            multiple_lines = "No phone service"

        if internet_service == "No":
            online_security = "No internet service"
            online_backup = "No internet service"
            device_protection = "No internet service"
            tech_support = "No internet service"
            streaming_tv = "No internet service"
            streaming_movies = "No internet service"

        # Create one customer row using the training feature names.
        customer_data = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        customer_df = pd.DataFrame(
            [customer_data],
            columns=FEATURE_COLUMNS,
        )

        # Get the model's estimated churn probability.
        churn_probability = float(
            model.predict_proba(customer_df)[0][1]
        )

        prediction = (
            "Yes" if churn_probability >= 0.60 else "No"
        )

        if churn_probability >= 0.60:
            risk_category = "High Risk"
        elif churn_probability >= 0.30:
            risk_category = "Medium Risk"
        else:
            risk_category = "Low Risk"

        st.divider()
        st.subheader("Prediction Results")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric(
                "Estimated Churn Probability",
                f"{churn_probability:.1%}",
            )

        with result_col2:
            st.metric(
                "Predicted to Churn",
                prediction,
            )

        with result_col3:
            st.metric(
                "Customer Risk",
                risk_category,
            )

        st.progress(
            min(max(churn_probability, 0.0), 1.0)
        )

        if prediction == "Yes":
            st.error(
                "This customer is classified as likely to churn. "
                "Consider contacting them with suitable retention offers."
            )
        else:
            st.success(
                "This customer is not classified as likely to churn "
                "at the current 0.60 threshold. Continue monitoring "
                "their engagement and satisfaction."
            )

        # Display retention recommendations.
        st.subheader("Retention Recommendations")

        if "recommend_retention" in globals():
            try:
                recommendations = recommend_retention(
                    customer_data,
                    churn_probability,
                )

                if isinstance(recommendations, str):
                    st.write(recommendations)
                elif recommendations:
                    for recommendation in recommendations:
                        st.markdown(f"- {recommendation}")
                else:
                    st.write(
                        "Continue monitoring the customer's account."
                    )

            except Exception as error:
                st.warning(
                    "The prediction succeeded, but recommendations "
                    f"could not be generated: {error}"
                )
        else:
            st.write(
                "Review the customer's contract, billing, and service "
                "preferences to identify suitable retention actions."
            )

        with st.expander("View customer details used for prediction"):
            st.dataframe(
                customer_df,
                use_container_width=True,
                hide_index=True,
            )

# --------------------------------------------------
# 7. CUSTOMER RISK PAGE
# --------------------------------------------------

elif page == "Customer Risk":

    st.title("⚠️ Customer Risk Analysis")
    st.write(
        "Identify customers with higher predicted churn probabilities "
        "so the business can prioritize retention efforts."
    )

    risk_filter = st.selectbox(
        "Filter by Risk Category",
        ["All", "Low Risk", "Medium Risk", "High Risk"],
    )

    search_id = st.text_input(
        "Search Customer ID",
        placeholder="Enter a customer ID...",
    )

    filtered_risk = risk_df.copy()

    if risk_filter != "All":
        filtered_risk = filtered_risk[
            filtered_risk["RiskCategory"] == risk_filter
        ]

    if search_id.strip():
        filtered_risk = filtered_risk[
            filtered_risk["CustomerID"].str.contains(
                search_id.strip(),
                case=False,
                na=False,
            )
        ]

    st.caption(f"Customers found: {len(filtered_risk):,}")

    display_columns = [
        "CustomerID",
        "ChurnProbability",
        "ChurnPrediction",
        "RiskCategory",
    ]

    st.dataframe(
        filtered_risk[display_columns].sort_values(
            "ChurnProbability",
            ascending=False,
        ).style.format(
            {"ChurnProbability": "{:.1%}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = filtered_risk.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Risk Report CSV",
        data=csv_data,
        file_name="customer_risk_report.csv",
        mime="text/csv",
    )

    st.subheader("Highest-Risk Customers")

    top_customers = (
        risk_df.sort_values("ChurnProbability", ascending=False)
        .head(10)
    )

    fig = px.bar(
        top_customers,
        x="CustomerID",
        y="ChurnProbability",
        title="Top 10 Customers by Estimated Churn Probability",
        labels={"ChurnProbability": "Churn Probability"},
    )

    fig.update_yaxes(tickformat=".0%")

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Risk categories use the estimated probability: Low below 30%, "
        "Medium from 30% to below 60%, and High at 60% or above."
    )


# --------------------------------------------------
# 8. RETENTION RECOMMENDATIONS PAGE
# --------------------------------------------------

elif page == "Retention Recommendations":

    st.title("🎯 Retention Recommendations")

    st.write(
        "Review suggested retention actions for customers based on "
        "their predicted risk and selected service characteristics."
    )

    recommendation_filter = st.selectbox(
        "Choose Customer Risk",
        ["All", "High Risk", "Medium Risk", "Low Risk"],
    )

    recommendations_df = risk_df.copy()

    if recommendation_filter != "All":
        recommendations_df = recommendations_df[
            recommendations_df["RiskCategory"] == recommendation_filter
        ]

    recommendation_columns = [
        "CustomerID",
        "ChurnProbability",
        "RiskCategory",
        "RetentionRecommendations",
    ]

    recommendations_df = recommendations_df.sort_values(
        "ChurnProbability",
        ascending=False,
    )

    st.dataframe(
        recommendations_df[recommendation_columns].style.format(
            {"ChurnProbability": "{:.1%}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = recommendations_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Retention Recommendations",
        data=csv_data,
        file_name="retention_recommendations.csv",
        mime="text/csv",
    )

    st.subheader("Retention Planning")

    high_risk = recommendations_df[
        recommendations_df["RiskCategory"] == "High Risk"
    ]

    medium_risk = recommendations_df[
        recommendations_df["RiskCategory"] == "Medium Risk"
    ]

    col1, col2 = st.columns(2)

    col1.metric("High-Risk Customers", f"{len(high_risk):,}")
    col2.metric("Medium-Risk Customers", f"{len(medium_risk):,}")

    st.info(
        "These are rule-based suggestions for a project demonstration. "
        "They are not automatically optimized offers and should be "
        "validated against business results before real-world use."
    )


# --------------------------------------------------
# 9. FEATURE IMPORTANCE PAGE
# --------------------------------------------------

elif page == "Feature Importance":

    st.title("📈 Feature Importance")
    st.write(
        "Explore which customer characteristics are associated "
        "with the predictions made by the Logistic Regression model."
    )

    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
        coefficients = classifier.coef_[0]

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefficients,
        })

        importance_df["AbsoluteCoefficient"] = (
            importance_df["Coefficient"].abs()
        )

        importance_df = importance_df.sort_values(
            "AbsoluteCoefficient",
            ascending=False,
        )

        top_features = importance_df.head(15).sort_values(
            "Coefficient",
            ascending=True,
        )

        fig = px.bar(
            top_features,
            x="Coefficient",
            y="Feature",
            orientation="h",
            title="Top 15 Features by Absolute Logistic Regression Coefficient",
            labels={
                "Coefficient": "Model Coefficient",
                "Feature": "Feature",
            },
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature Coefficient Details")

        st.dataframe(
            importance_df[
                ["Feature", "Coefficient", "AbsoluteCoefficient"]
            ].head(30).style.format({
                "Coefficient": "{:.4f}",
                "AbsoluteCoefficient": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        csv_data = importance_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Feature Coefficients",
            data=csv_data,
            file_name="logistic_regression_feature_coefficients.csv",
            mime="text/csv",
        )

        st.caption(
            "Positive coefficients generally increase the model's estimated "
            "churn odds, while negative coefficients generally decrease them, "
            "holding the other encoded features constant. These are model "
            "relationships, not proof of causation."
        )

    except (AttributeError, ValueError) as error:
        st.error(f"Unable to calculate feature coefficients: {error}")


# --------------------------------------------------
# 10. BATCH PREDICTIONS PAGE
# --------------------------------------------------

elif page == "Batch Predictions":

    st.title("📂 Batch Customer Predictions")

    st.write(
        "Upload a CSV containing multiple customers to predict their "
        "churn probabilities in one batch."
    )

    st.info(
        "Your CSV must contain the same feature columns as "
        "outputs/cleaned_churn.csv. The Churn target column is not required."
    )

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
    )

    if uploaded_file is not None:

        try:
            upload_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data Preview")
            st.dataframe(
                upload_df.head(10),
                use_container_width=True,
                hide_index=True,
            )

            missing_columns = [
                column for column in FEATURE_COLUMNS
                if column not in upload_df.columns
            ]

            if missing_columns:
                st.error(
                    "Your CSV is missing these required columns: "
                    + ", ".join(missing_columns)
                )

            else:
                batch_features = upload_df[FEATURE_COLUMNS].copy()

                batch_results = predict_customers(batch_features)

                if "customerID" in upload_df.columns:
                    batch_results.insert(
                        0,
                        "CustomerID",
                        upload_df["customerID"].astype(str).values,
                    )
                else:
                    batch_results.insert(
                        0,
                        "CustomerID",
                        [
                            f"Uploaded_Customer_{i + 1}"
                            for i in range(len(upload_df))
                        ],
                    )

                st.subheader("Batch Prediction Results")

                st.dataframe(
                    batch_results.style.format({
                        "ChurnProbability": "{:.1%}"
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Customers Processed",
                    f"{len(batch_results):,}",
                )

                col2.metric(
                    "Predicted to Churn",
                    f"{
                        (batch_results['ChurnPrediction'] == 'Yes').sum():,}"
                )

                col3.metric(
                    "High-Risk Customers",
                    f"{
                        (batch_results['RiskCategory'] == 'High Risk').sum():,}"
                )

                csv_data = batch_results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Batch Predictions",
                    data=csv_data,
                    file_name="batch_churn_predictions.csv",
                    mime="text/csv",
                )

        except (ValueError, pd.errors.ParserError) as error:
            st.error(f"Unable to read the uploaded CSV: {error}")


# --------------------------------------------------
# 11. PROJECT OVERVIEW PAGE
# --------------------------------------------------

elif page == "Project Overview":

    st.title("ℹ️ Project Overview")

    st.write(
        "Customer Churn Prediction is a machine learning project "
        "that estimates which telecom customers may leave a service."
    )

    st.subheader("Project Objectives")

    st.markdown(
        """
        - Explore historical customer churn patterns.
        - Clean and prepare customer data for machine learning.
        - Train a Logistic Regression classification model.
        - Estimate churn probabilities for individual customers.
        - Segment customers into estimated risk categories.
        - Suggest rule-based customer retention actions.
        - Support batch predictions through CSV uploads.
        """
    )

    st.subheader("Technology Stack")

    col1, col2, col3 = st.columns(3)

    col1.markdown(
        """
        **Programming**
        - Python
        - Pandas
        - NumPy
        """
    )

    col2.markdown(
        """
        **Machine Learning**
        - Scikit-learn
        - Logistic Regression
        - One-Hot Encoding
        """
    )

    col3.markdown(
        """
        **Dashboard**
        - Streamlit
        - Plotly
        - CSV reporting
        """
    )

    st.subheader("Selected Model Evaluation")

    st.write(
        "The following metrics come from the held-out test evaluation "
        "using a 0.60 decision threshold. The production model is then "
        "retrained on the full cleaned dataset for dashboard predictions."
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Accuracy", "76.22%")
    metric2.metric("Churn Recall", "70.59%")
    metric3.metric("Churn F1-Score", "61.18%")
    metric4.metric("ROC-AUC", "0.8416")

    st.caption(
        "These are evaluation results from the test split, not measurements "
        "of the production model on new unseen customers. Accuracy alone "
        "does not describe all model performance."
    )

    st.subheader("Decision Threshold")

    st.write(
        f"The dashboard uses a probability threshold of {MODEL_THRESHOLD:.0%}. "
        "Customers at or above this probability are classified as predicted "
        "to churn. Changing the threshold affects false positives and "
        "false negatives."
    )

    st.subheader("Important Limitations")

    st.markdown(
        """
        - Predictions are estimates, not guarantees.
        - Historical associations do not establish causation.
        - Retention suggestions use simple business rules.
        - Model performance may change on new customer populations.
        - Customer data should be handled responsibly and securely.
        """
    )

    st.success("Customer Churn Prediction dashboard loaded.")
