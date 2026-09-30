from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

APP_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = APP_DIRECTORY / "churn_model.pkl"
FEATURES_PATH = APP_DIRECTORY / "model_features.pkl"

st.set_page_config(
    page_title="Telco Churn Intelligence Platform",
    page_icon="📡",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    model_features = joblib.load(FEATURES_PATH)
    if not isinstance(model_features, list) or not model_features:
        raise ValueError("The saved model feature list is missing or invalid.")
    if model.n_features_in_ != len(model_features):
        raise ValueError("The model and saved feature list do not match.")
    if not hasattr(model, "predict_proba"):
        raise TypeError("The saved model does not support probability predictions.")
    return model, model_features, shap.TreeExplainer(model)


try:
    model, model_features, explainer = load_artifacts()
except Exception as exc:
    st.error(f"Error loading model artifacts: {exc}")
    st.stop()

st.title("📡 Telecom Retention Intelligence & Decision Engine")
st.caption(
    "Enterprise churn scoring, explainable AI (SHAP), and financial ROI scenarios"
)

with st.sidebar:
    st.header("Customer profile controls")
    tenure = st.slider("Tenure (months)", min_value=1, max_value=72, value=4)
    monthly_charges = st.slider(
        "Monthly charges ($)", min_value=18.0, max_value=120.0, value=85.0
    )
    contract = st.selectbox(
        "Contract type", ["Month-to-month", "One year", "Two year"]
    )
    internet_service = st.selectbox(
        "Internet service", ["Fiber optic", "DSL", "No"]
    )
    tech_support = st.selectbox(
        "Tech support", ["No", "Yes", "No internet service"]
    )
    online_security = st.selectbox(
        "Online security", ["No", "Yes", "No internet service"]
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
    paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
    senior_citizen = st.selectbox("Senior citizen", [0, 1])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])

total_charges = tenure * monthly_charges
input_dict = {
    "SeniorCitizen": senior_citizen,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Partner_Yes": int(partner == "Yes"),
    "Dependents_Yes": int(dependents == "Yes"),
    "PaperlessBilling_Yes": int(paperless_billing == "Yes"),
}

if contract == "One year":
    input_dict["Contract_One year"] = 1
elif contract == "Two year":
    input_dict["Contract_Two year"] = 1

if internet_service == "Fiber optic":
    input_dict["InternetService_Fiber optic"] = 1
elif internet_service == "No":
    input_dict["InternetService_No"] = 1

if tech_support == "Yes":
    input_dict["TechSupport_Yes"] = 1
elif tech_support == "No internet service":
    input_dict["TechSupport_No internet service"] = 1

if online_security == "Yes":
    input_dict["OnlineSecurity_Yes"] = 1
elif online_security == "No internet service":
    input_dict["OnlineSecurity_No internet service"] = 1

if payment_method == "Credit card (automatic)":
    input_dict["PaymentMethod_Credit card (automatic)"] = 1
elif payment_method == "Electronic check":
    input_dict["PaymentMethod_Electronic check"] = 1
elif payment_method == "Mailed check":
    input_dict["PaymentMethod_Mailed check"] = 1

input_df = pd.DataFrame(0.0, index=[0], columns=model_features)
for feature, value in input_dict.items():
    if feature in input_df.columns:
        input_df.at[0, feature] = value

if "TotalCharges" in input_df.columns:
    input_df.at[0, "TotalCharges"] = total_charges

try:
    churn_class_index = list(model.classes_).index(1)
    churn_prob = float(
        model.predict_proba(input_df[model_features])[0, churn_class_index]
    )
except (AttributeError, IndexError, TypeError, ValueError) as exc:
    st.error(f"Unable to score this customer profile: {exc}")
    st.stop()

churn_pct = churn_prob * 100
tab1, tab2, tab3 = st.tabs(
    [
        "🎯 Retention Simulator",
        "🧠 Explainable AI (SHAP)",
        "💰 Financial ROI Optimizer",
    ]
)

with tab1:
    col_metric, col_status = st.columns([1, 2])
    with col_metric:
        st.metric(label="Predicted churn risk", value=f"{churn_pct:.1f}%")
    with col_status:
        if churn_pct >= 50:
            st.error("**HIGH FLIGHT RISK**: prioritize a personalized intervention.")
            risk_label = "High risk"
        elif churn_pct >= 25:
            st.warning("**MODERATE RISK**: review service experience and retention needs.")
            risk_label = "Moderate risk"
        else:
            st.success("**STABLE ACCOUNT**: maintain standard loyalty monitoring.")
            risk_label = "Stable"

    st.progress(churn_prob)
    st.subheader("Prescriptive retention playbook")
    recommendations = []
    if contract == "Month-to-month":
        recommendations.append(
            "Offer a voluntary one-year plan comparison, including any eligible savings."
        )
    if tech_support == "No":
        recommendations.append(
            "Offer a technical-support check or trial where the customer is eligible."
        )
    if internet_service == "Fiber optic" and monthly_charges > 80:
        recommendations.append(
            "Review service usage and plan value before proposing an add-on or plan change."
        )
    if payment_method == "Electronic check":
        recommendations.append(
            "Offer an optional automatic-payment method and explain its convenience."
        )

    if recommendations:
        for recommendation in recommendations:
            st.markdown(f"- {recommendation}")
    else:
        st.info("Maintain standard loyalty monitoring and respond to any service issues.")

    st.caption(
        "Fields without a sidebar control use their training reference category. "
        "Total charges are estimated as monthly charges multiplied by tenure."
    )

with tab2:
    st.subheader("Local feature attribution")
    st.write(
        "SHAP shows how this profile's feature values influence the model output. "
        "Positive values push toward churn; negative values push away from churn. "
        "These are model explanations, not evidence of causation."
    )

    try:
        with st.spinner("Computing TreeSHAP values..."):
            shap_output = explainer.shap_values(input_df[model_features])

        if isinstance(shap_output, list):
            churn_shap = np.asarray(shap_output[churn_class_index])[0]
        else:
            shap_array = np.asarray(shap_output)
            class_count = len(model.classes_)
            feature_count = len(model_features)
            if shap_array.ndim == 3 and shap_array.shape[1] == feature_count:
                churn_shap = shap_array[0, :, churn_class_index]
            elif (
                shap_array.ndim == 3
                and shap_array.shape[0] == class_count
                and shap_array.shape[2] == feature_count
            ):
                churn_shap = shap_array[churn_class_index, 0, :]
            elif (
                shap_array.ndim == 3
                and shap_array.shape[2] == feature_count
                and shap_array.shape[1] == class_count
            ):
                churn_shap = shap_array[0, churn_class_index, :]
            elif shap_array.ndim == 2 and shap_array.shape[1] == feature_count:
                churn_shap = shap_array[0]
            else:
                raise ValueError(f"Unexpected SHAP output shape: {shap_array.shape}")

        impact_df = pd.DataFrame(
            {"Feature": model_features, "SHAP value": np.asarray(churn_shap)}
        )
        top_drivers = pd.concat(
            [
                impact_df.nlargest(5, "SHAP value"),
                impact_df.nsmallest(3, "SHAP value"),
            ]
        ).drop_duplicates(subset="Feature")
        top_drivers = top_drivers.sort_values("SHAP value")

        fig, ax = plt.subplots(figsize=(8, max(4, 0.55 * len(top_drivers))))
        bar_colors = [
            "#c94c4c" if value > 0 else "#25855a"
            for value in top_drivers["SHAP value"]
        ]
        ax.barh(
            top_drivers["Feature"],
            top_drivers["SHAP value"],
            color=bar_colors,
            height=0.6,
        )
        ax.axvline(0, color="#263238", linestyle="--", linewidth=0.8)
        ax.set_xlabel("SHAP impact on model output")
        ax.set_ylabel("")
        ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as exc:
        st.error(f"Unable to calculate SHAP explanations: {exc}")

with tab3:
    st.subheader("Expected monetary value and retention ROI")
    st.write("Estimate the financial value of a retention offer using editable assumptions.")

    col1, col2, col3 = st.columns(3)
    with col1:
        annual_clv = st.number_input(
            "Estimated customer annual value ($)",
            min_value=0.0,
            value=float(monthly_charges * 12),
            step=100.0,
        )
    with col2:
        offer_cost = st.number_input(
            "Annual cost of retention offer ($)",
            min_value=0.0,
            value=120.0,
            step=10.0,
        )
    with col3:
        expected_risk_drop = st.slider(
            "Assumed churn-risk reduction (percentage points)",
            min_value=10,
            max_value=60,
            value=35,
            step=5,
        ) / 100.0

    prob_without_action = churn_prob
    prob_with_action = max(0.0, churn_prob - expected_risk_drop)
    expected_loss_no_action = prob_without_action * annual_clv
    expected_loss_with_action = (prob_with_action * annual_clv) + offer_cost
    net_financial_benefit = expected_loss_no_action - expected_loss_with_action

    st.divider()
    f_col1, f_col2, f_col3 = st.columns(3)
    f_col1.metric("Risk without offer", f"{prob_without_action:.1%}")
    f_col2.metric("Scenario risk after offer", f"{prob_with_action:.1%}")
    f_col3.metric(
        "Net expected annual savings",
        f"${net_financial_benefit:,.2f}",
        delta=f"${net_financial_benefit:,.2f}",
    )

    if annual_clv > 0:
        break_even_reduction = (offer_cost / annual_clv) * 100
        st.caption(
            f"Break-even risk reduction under these assumptions: "
            f"{break_even_reduction:.1f} percentage points."
        )
    st.info(
        "ROI is a scenario estimate: the assumed risk reduction is user-provided "
        "and is not a measured causal effect of the offer."
    )

    if net_financial_benefit > 0:
        st.success(
            "Positive business case under the selected assumptions: expected net "
            f"benefit is ${net_financial_benefit:,.2f} per subscriber."
        )
    else:
        st.error(
            "Negative business case under the selected assumptions: the offer "
            "cost exceeds expected loss avoidance."
        )
