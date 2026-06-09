
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

# -----------------------------
# Data
# -----------------------------
comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "XGBoost", "Tuned XGBoost"],
    "Accuracy": [0.958259, 0.983393, 0.986535],
    "Precision": [0.741313, 0.959391, 0.994737],
    "Recall": [0.880734, 0.866972, 0.866972],
    "F1-Score": [0.805031, 0.910843, 0.926471],
    "ROC-AUC": [0.956217, 0.954900, 0.958289]
})

best_model = comparison_df.loc[
    comparison_df["F1-Score"].idxmax()
]

# -----------------------------
# Header
# -----------------------------
st.title("Fraud Detection Stakeholder Dashboard")

st.markdown(
    "A business-focused dashboard showing how the final model supports fraud detection, "
    "false-positive reduction, explainability, and scalable monitoring."
)

# -----------------------------
# KPI Cards
# -----------------------------
st.subheader("Recommended Model: Tuned XGBoost")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "98.65%")
col2.metric("Precision", "99.47%")
col3.metric("Recall", "86.70%")
col4.metric("ROC-AUC", "95.83%")

st.success(
    "Tuned XGBoost is recommended because it provides the best overall balance of "
    "accuracy, precision, F1-score, and ROC-AUC."
)

# -----------------------------
# Performance Table
# -----------------------------
st.subheader("Model Performance Summary")

display_df = comparison_df.copy()
for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
    display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"

st.dataframe(display_df, use_container_width=True, hide_index=True)

# -----------------------------
# Better Comparison Chart
# -----------------------------
st.subheader("Model Performance Comparison")

metric_choice = st.selectbox(
    "Select metric to compare:",
    ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    index=3
)

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(
    comparison_df["Model"],
    comparison_df[metric_choice]
)

ax.set_ylim(0, 1.05)
ax.set_ylabel(metric_choice)
ax.set_title(f"{metric_choice} Comparison by Model")

for i, value in enumerate(comparison_df[metric_choice]):
    ax.text(
        i,
        value + 0.02,
        f"{value:.2%}",
        ha="center",
        fontsize=10
    )

plt.xticks(rotation=0)
st.pyplot(fig)

# -----------------------------
# SHAP Explainability
# -----------------------------
st.subheader("SHAP Explainability: What Drives Fraud Predictions?")

st.markdown(
    "SHAP explains which features push a transaction toward fraud or non-fraud. "
    "The higher the SHAP impact, the more influence that feature has on the model decision."
)

# SHAP feature importance data based on your output
shap_df = pd.DataFrame({
    "Feature": [
        "location_mismatch",
        "transaction_hour",
        "txn_velocity_24h",
        "corridor_risk",
        "fee",
        "exchange_rate_src_to_dest",
        "ip_risk_score",
        "account_age_days",
        "new_device",
        "log_exchange_rate"
    ],
    "SHAP Impact": [
        2.00,
        1.99,
        1.03,
        0.98,
        0.84,
        0.70,
        0.60,
        0.49,
        0.47,
        0.40
    ]
})

fig, ax = plt.subplots(figsize=(9, 5))

shap_df = shap_df.sort_values("SHAP Impact", ascending=True)

ax.barh(
    shap_df["Feature"],
    shap_df["SHAP Impact"]
)

ax.set_xlabel("Mean SHAP Impact")
ax.set_title("Top Fraud Drivers Based on SHAP")

st.pyplot(fig)

st.info(
    "Key takeaway: Location mismatch, transaction hour, transaction velocity, corridor risk, "
    "and transaction fee were the strongest drivers behind fraud predictions."
)

# -----------------------------
# Recommendations
# -----------------------------
st.subheader("Business Recommendations")

rec1, rec2, rec3 = st.columns(3)

with rec1:
    st.info(
        "**Monitor transaction velocity**\n\n"
        "Track rapid transfers within 1-hour and 24-hour windows."
    )

with rec2:
    st.info(
        "**Strengthen IP and device intelligence**\n\n"
        "Use IP reputation, device trust, and location mismatch alerts."
    )

with rec3:
    st.info(
        "**Use SHAP for analyst review**\n\n"
        "Explain fraud alerts clearly for compliance and investigation."
    )

# -----------------------------
# Limitations
# -----------------------------
st.subheader("Model Limitations and Continuous Improvement")

with st.expander("View limitations and next steps"):
    st.markdown("""
    - Fraud behavior changes over time; the model should be retrained regularly.
    - Some fraud cases may still be missed.
    - False positives can still occur if thresholds are too aggressive.
    - Model quality depends on reliable transaction, device, and risk data.
    - Future improvements should include customer history, time since last transaction, and network-based fraud indicators.
    """)

# -----------------------------
# Final Message
# -----------------------------
st.subheader("Stakeholder Takeaway")

st.info(
    "Tuned XGBoost is the preferred model for deployment because it delivers strong fraud detection performance, "
    "very high precision, strong ROC-AUC, and explainable predictions through SHAP."
)