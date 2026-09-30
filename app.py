from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = APP_DIRECTORY / "churn_model.pkl"
FEATURES_PATH = APP_DIRECTORY / "model_features.pkl"

st.set_page_config(
    page_title="Telco Retention Intelligence & Risk Simulator",
    page_icon=":material/network_cell:",
    layout="wide",
)

st.title("Telco Retention Intelligence & Risk Simulator")
st.caption("Explore a customer profile and estimate churn risk before planning an intervention.")


@st.cache_resource
def load_model_artifacts(model_path: str, feature_path: str):
    model = joblib.load(model_path)
    feature_columns = joblib.load(feature_path)
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("The saved model feature list is missing or invalid.")
    if not hasattr(model, "predict_proba"):
        raise TypeError("The saved model does not support probability predictions.")
    return model, feature_columns


if not MODEL_PATH.is_file() or not FEATURES_PATH.is_file():
    st.error("Model artifacts are missing. Run `python model.py` from the project root first.")
    st.stop()

try:
    model, feature_columns = load_model_artifacts(str(MODEL_PATH), str(FEATURES_PATH))
except Exception as exc:
    st.error(f"Unable to load the churn model artifacts: {exc}")
    st.stop()

required_numeric_features = {"tenure", "MonthlyCharges"}
missing_numeric_features = required_numeric_features - set(feature_columns)
if missing_numeric_features:
    st.error(
        "The trained model is missing required input features: "
        f"{', '.join(sorted(missing_numeric_features))}. Run `python model.py` again."
    )
    st.stop()

with st.sidebar:
    st.header("Customer profile")
    tenure = st.slider("Tenure (months)", min_value=1, max_value=72, value=12, step=1)
    monthly_charges = st.slider(
        "Monthly charges ($)",
        min_value=18.0,
        max_value=120.0,
        value=75.0,
        step=1.0,
    )
    contract_type = st.selectbox(
        "Contract type", ["Month-to-month", "One year", "Two year"]
    )
    internet_service = st.selectbox(
        "Internet service",
        ["DSL", "Fiber optic", "No"],
        format_func=lambda value: "No internet service" if value == "No" else value,
    )
    tech_support = st.selectbox(
        "Tech support", ["No", "Yes", "No internet service"]
    )
    payment_method = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

st.caption(
    "Other profile fields use their training reference category. Total charges are "
    "estimated as monthly charges multiplied by tenure."
)

profile = pd.DataFrame(0.0, index=[0], columns=feature_columns)
profile.loc[0, "tenure"] = tenure
profile.loc[0, "MonthlyCharges"] = monthly_charges
if "TotalCharges" in profile.columns:
    profile.loc[0, "TotalCharges"] = monthly_charges * tenure

selected_categories = {
    "Contract": contract_type,
    "InternetService": internet_service,
    "TechSupport": tech_support,
    "PaymentMethod": payment_method,
}
for column, value in selected_categories.items():
    dummy_column = f"{column}_{value}"
    if dummy_column in profile.columns:
        profile.loc[0, dummy_column] = 1.0

try:
    churn_class_index = list(model.classes_).index(1)
    churn_risk = float(model.predict_proba(profile[feature_columns])[0, churn_class_index])
except (AttributeError, IndexError, TypeError, ValueError) as exc:
    st.error(f"The model could not score this profile: {exc}")
    st.stop()

if churn_risk > 0.50:
    risk_label = "High risk"
    st.error("High risk: prioritize a personalized retention contact.")
    recommendations = [
        "Reach out promptly with a tailored save offer and confirm the customer’s main concern.",
    ]
    if contract_type == "Month-to-month":
        recommendations.append(
            "Offer a clear annual-plan benefit, while keeping the choice voluntary."
        )
    if tech_support != "Yes":
        recommendations.append(
            "Offer a proactive technical-support check or guided setup session."
        )
    if payment_method == "Electronic check":
        recommendations.append(
            "Offer an optional automatic-payment method for convenience."
        )
elif churn_risk >= 0.25:
    risk_label = "Moderate risk"
    st.warning("Moderate risk: use a targeted service check-in and relevant benefit.")
    recommendations = [
        "Review recent service experience before offering a retention incentive.",
        "Promote a contract or support option that fits this customer’s selected profile.",
    ]
else:
    risk_label = "Stable"
    st.success("Stable: reinforce value and monitor for changes in service experience.")
    recommendations = [
        "Reinforce the value of the current plan and recognize customer loyalty.",
        "Avoid broad discounts unless new service issues or risk signals emerge.",
    ]

risk_column, action_column = st.columns([1, 1.4])
with risk_column:
    with st.container(border=True):
        st.subheader("Churn risk")
        st.metric("Model-estimated churn probability", f"{churn_risk:.1%}")
        st.progress(churn_risk)
        st.caption(f"Risk segment: {risk_label}")

with action_column:
    with st.container(border=True):
        st.subheader("Recommended retention actions")
        for recommendation in recommendations:
            st.markdown(f"- {recommendation}")
