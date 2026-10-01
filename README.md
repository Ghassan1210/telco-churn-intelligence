# Telecom Retention Intelligence

An interactive telecom churn risk simulator with retention recommendations, local model explanations, and offer-value scenarios.

**Live app:** [ghn-telco-churn.streamlit.app](https://ghn-telco-churn.streamlit.app/)

## Explainable AI (SHAP)

The Explainable AI tab uses `shap.TreeExplainer` to show local feature attribution for the current customer profile. A horizontal bar chart highlights the strongest feature pushes toward churn in red and away from churn in green. These attributions explain the model output; they do not establish causation.

## Financial ROI Optimizer

The Financial ROI Optimizer estimates Expected Monetary Value (EMV) for a retention offer using the selected annual customer value, offer cost, and assumed churn-risk reduction:

- Baseline expected loss: `churn probability × annual customer value`
- Post-intervention expected loss: `max(0, churn probability − assumed risk reduction) × annual customer value + offer cost`
- Net expected savings: `baseline expected loss − post-intervention expected loss`

The estimated risk reduction is a user-provided scenario assumption, not a measured causal effect. The app also reports the break-even risk reduction under the selected values.
