from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "nova_pay_eda_ready.csv"
MODEL_ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "tuned_xgb_bundle.joblib"
TARGET_COL = "is_fraud"
DROP_COLS = ["timestamp"]
BENCHMARK_RESULTS = {
    "accuracy": 0.986535,
    "precision": 0.994737,
    "recall": 0.866972,
    "f1": 0.926471,
    "roc_auc": 0.958289,
}
BENCHMARK_CONFUSION_MATRIX = np.array([[2009, 1], [29, 189]])
USER_INPUT_SPECS = [
    {"feature": "amount_usd", "label": "Transaction Amount (USD)", "kind": "numeric"},
    {"feature": "fee", "label": "Transfer Fee (USD)", "kind": "numeric"},
    {"feature": "exchange_rate_src_to_dest", "label": "Exchange Rate", "kind": "numeric"},
    {"feature": "account_age_days", "label": "Account Age (Days)", "kind": "numeric"},
    {"feature": "risk_score_internal", "label": "Internal Risk Score", "kind": "numeric"},
    {"feature": "device_trust_score", "label": "Device Trust Score", "kind": "numeric"},
    {"feature": "new_device", "label": "New Device", "kind": "bool"},
    {"feature": "ip_risk_score", "label": "IP Risk Score", "kind": "numeric"},
    {"feature": "txn_velocity_12h", "label": "Transactions in Last 12 Hours", "kind": "numeric"},
    {"feature": "txn_velocity_24h", "label": "Transactions in Last 24 Hours", "kind": "numeric"},
    {"feature": "transaction_hour", "label": "Transaction Hour", "kind": "numeric"},
    {"feature": "location_mismatch", "label": "Location Mismatch", "kind": "bool"},
]
USER_INPUT_FEATURES = [spec["feature"] for spec in USER_INPUT_SPECS]
USER_INPUT_ALIASES = {
    spec["label"]: spec["feature"] for spec in USER_INPUT_SPECS
}
USER_INPUT_ALIASES.update({spec["feature"]: spec["feature"] for spec in USER_INPUT_SPECS})


@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_or_train_model(path: str) -> dict:
    if MODEL_ARTIFACT_PATH.exists():
        return joblib.load(MODEL_ARTIFACT_PATH)

    df = load_dataset(path).copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)

    X = df.drop(columns=[TARGET_COL]).copy()
    drop_present = [col for col in DROP_COLS if col in X.columns]
    if drop_present:
        X = X.drop(columns=drop_present)

    # Guard against invalid numeric values coming from engineered log features.
    X = X.replace([np.inf, -np.inf], np.nan)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Tuned XGBoost parameters taken from notebook/model.ipynb best model output.
    tuned_xgb = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        tree_method="hist",
        subsample=0.8,
        scale_pos_weight=1,
        reg_lambda=5,
        reg_alpha=0,
        n_estimators=700,
        min_child_weight=1,
        max_depth=2,
        learning_rate=0.05,
        gamma=0.05,
        colsample_bytree=0.9,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", tuned_xgb),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)

    transformed_feature_names = preprocessor.get_feature_names_out().tolist()

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    cm = confusion_matrix(y_test, y_pred)

    default_inputs: dict = {}
    for col in X.columns:
        series = X[col]
        if is_numeric_dtype(series):
            default_inputs[col] = float(series.median()) if not series.dropna().empty else 0.0
        else:
            mode_series = series.dropna().mode()
            default_inputs[col] = mode_series.iat[0] if not mode_series.empty else "unknown"

    top_input_features = [feature for feature in USER_INPUT_FEATURES if feature in X.columns]

    bundle = {
        "model": model,
        "reference_features": X,
        "feature_cols": X.columns.tolist(),
        "top_input_features": top_input_features,
        "default_inputs": default_inputs,
        "transformed_feature_names": transformed_feature_names,
        "metrics": metrics,
        "confusion_matrix": cm,
        "fraud_rate": float(y.mean()),
        "row_count": int(len(df)),
    }

    persist_model_bundle(bundle)
    return bundle


def persist_model_bundle(bundle: dict) -> None:
    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_ARTIFACT_PATH)


def render_confusion_matrix(cm: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Legit", "Pred: Fraud"])
    ax.set_yticklabels(["Actual: Legit", "Actual: Fraud"])
    ax.set_title("Validation Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)


def format_default_value(value: object) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def parse_numeric_text(value: str, default_value: float) -> float:
    stripped = value.strip()
    if not stripped:
        return default_value
    try:
        return float(stripped)
    except ValueError:
        return default_value


def build_single_input_form(
    feature_df: pd.DataFrame,
    input_specs: list[dict],
    default_inputs: dict,
) -> dict:
    payload: dict = {}

    for spec in input_specs:
        col = spec["feature"]
        label = spec["label"]
        series = feature_df[col] if col in feature_df.columns else None

        if series is not None and is_numeric_dtype(series) and not series.dropna().empty:
            default_value = float(series.median())
        elif col == "txn_velocity_12h":
            default_value = float(default_inputs.get("txn_velocity_24h", 0.0)) / 2.0
        elif spec["kind"] == "numeric":
            default_value = float(default_inputs.get(col, 0.0))
        else:
            default_value = None

        default_text = format_default_value(default_value if default_value is not None else 0.0)

        if spec["kind"] == "bool":
            default_bool = bool(default_inputs.get(col, False))
            payload[col] = st.toggle(label, value=default_bool)
        elif col == "transaction_hour":
            default_hour = int(round(default_value)) if default_value is not None else 0
            hour_text = st.text_input(label, value=str(default_hour), placeholder="0-23")
            payload[col] = parse_numeric_text(hour_text, float(default_hour))
        elif spec["kind"] == "numeric":
            number_text = st.text_input(label, value=default_text, placeholder="Type or clear the value")
            payload[col] = parse_numeric_text(number_text, float(default_value if default_value is not None else 0.0))
        else:
            options = sorted(series.dropna().astype(str).unique().tolist()) if series is not None else []
            if not options:
                options = ["unknown"]

            mode_series = series.dropna().astype(str).mode() if series is not None else pd.Series(dtype=str)
            default_option = mode_series.iat[0] if not mode_series.empty else options[0]
            default_index = options.index(default_option) if default_option in options else 0

            payload[col] = st.selectbox(label, options=options, index=default_index)

    return payload


def assemble_model_input(
    input_df: pd.DataFrame,
    feature_cols: list[str],
    default_inputs: dict,
) -> pd.DataFrame:
    base_df = pd.DataFrame([default_inputs] * len(input_df))

    for col in input_df.columns:
        if col in base_df.columns:
            base_df[col] = input_df[col].values

    if "txn_velocity_12h" in base_df.columns and "txn_velocity_1h" in base_df.columns:
        txn_velocity_12h = pd.to_numeric(base_df["txn_velocity_12h"], errors="coerce")
        txn_velocity_12h = txn_velocity_12h.fillna(float(default_inputs.get("txn_velocity_12h", 0.0)))
        base_df["txn_velocity_1h"] = np.clip(txn_velocity_12h / 12.0, 0, None)

    if "txn_velocity_24h" in base_df.columns:
        txn_velocity_24h = pd.to_numeric(base_df["txn_velocity_24h"], errors="coerce")
        base_df["txn_velocity_24h"] = txn_velocity_24h.fillna(float(default_inputs.get("txn_velocity_24h", 0.0))).clip(lower=0)

    if "exchange_rate_src_to_dest" in base_df.columns and "log_exchange_rate" in base_df.columns:
        exchange_rate = pd.to_numeric(base_df["exchange_rate_src_to_dest"], errors="coerce")
        exchange_rate = exchange_rate.fillna(float(default_inputs.get("exchange_rate_src_to_dest", 0.0)))
        exchange_rate = exchange_rate.clip(lower=0)
        base_df["log_exchange_rate"] = np.log1p(exchange_rate)

    return base_df[feature_cols]


def normalize_input_columns(input_df: pd.DataFrame) -> pd.DataFrame:
    normalized = input_df.copy()
    normalized.columns = [USER_INPUT_ALIASES.get(col, col) for col in normalized.columns]
    return normalized


def humanize_feature_name(feature_name: str) -> str:
    cleaned = feature_name.replace("num__", "").replace("cat__", "")
    label_map = {spec["feature"]: spec["label"] for spec in USER_INPUT_SPECS}
    return label_map.get(cleaned, cleaned.replace("_", " ").title())


def explain_transaction(
    model: Pipeline,
    input_df: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 5,
) -> pd.DataFrame:
    shap = importlib.import_module("shap")
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    transformed = preprocessor.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_row = shap_values[-1][0]
    else:
        shap_row = shap_values[0]

    explanation_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP Value": shap_row,
        }
    ).sort_values("SHAP Value", key=lambda series: series.abs(), ascending=False)

    return explanation_df.head(top_n)


st.set_page_config(page_title="Real-Time Fraud Detection", layout="wide")
st.title("Real-Time Fraud Detection Deployment App")

# Allow the deployed app to proceed if the dataset is missing by letting the
# user upload a CSV in the browser. This avoids forcing the dataset into the
# repository for demos.
if not DATA_PATH.exists():
    st.warning(f"Dataset not found at: {DATA_PATH}")
    uploaded_dataset = st.file_uploader(
        "Upload processed dataset CSV (nova_pay_eda_ready.csv) for this session",
        type=["csv"],
    )
    if uploaded_dataset is None:
        st.info("Upload a CSV here, or add the file to the repository at data/nova_pay_eda_ready.csv and redeploy.")
        st.stop()
    # persist uploaded file so the rest of the app (which expects the file path)
    # can load it the same way as when the CSV is present in the repo.
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_PATH, "wb") as f:
        f.write(uploaded_dataset.getbuffer())
    st.success("Dataset uploaded and saved for this session.")

try:
    bundle = load_or_train_model(str(DATA_PATH))
except Exception as exc:
    st.error(f"Failed to train model: {exc}")
    st.stop()

model = bundle["model"]
feature_cols = bundle["feature_cols"]
top_input_features = bundle["top_input_features"]
default_inputs = bundle["default_inputs"]
transformed_feature_names = bundle["transformed_feature_names"]
reference_features = bundle["reference_features"]
metrics = bundle["metrics"]
cm = bundle["confusion_matrix"]

st.sidebar.header("Scoring Controls")
threshold = st.sidebar.slider("Fraud probability threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.01)
st.sidebar.write(f"Rows used for training: {bundle['row_count']:,}")
st.sidebar.write(f"Base fraud rate: {bundle['fraud_rate']:.2%}")

overview_tab, realtime_tab, batch_tab = st.tabs(["Model Overview", "Real-Time Prediction", "Batch Scoring"])

with overview_tab:
    st.info(
        "Tuned XGBoost benchmark results: Accuracy 0.9865, Precision 0.9947, "
        "Recall 0.8670, F1 0.9265, ROC-AUC 0.9583."
    )

    st.subheader("Tuned XGBoost Benchmark Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{BENCHMARK_RESULTS['accuracy']:.3f}")
    m2.metric("Precision", f"{BENCHMARK_RESULTS['precision']:.3f}")
    m3.metric("Recall", f"{BENCHMARK_RESULTS['recall']:.3f}")
    m4.metric("F1", f"{BENCHMARK_RESULTS['f1']:.3f}")
    m5.metric("ROC-AUC", f"{BENCHMARK_RESULTS['roc_auc']:.3f}")

    st.subheader("Tuned XGBoost Confusion Matrix")
    render_confusion_matrix(BENCHMARK_CONFUSION_MATRIX)

with realtime_tab:
    st.subheader("Score One Live Transaction")
    st.write("Enter standard transaction details and click **Predict Fraud Risk** for a real-time fraud decision.")

    with st.form("single_prediction_form"):
        form_payload = build_single_input_form(reference_features, USER_INPUT_SPECS, default_inputs)
        submit = st.form_submit_button("Predict Fraud Risk")

    if submit:
        single_input_df = pd.DataFrame([form_payload], columns=USER_INPUT_FEATURES)
        single_df = assemble_model_input(single_input_df, feature_cols, default_inputs)
        single_df = single_df.replace([np.inf, -np.inf], np.nan)
        fraud_probability = float(model.predict_proba(single_df)[0, 1])
        fraud_prediction = int(fraud_probability >= threshold)

        st.metric("Fraud Probability", f"{fraud_probability:.2%}")
        st.progress(min(max(fraud_probability, 0.0), 1.0))

        if fraud_prediction == 1:
            st.error("Decision: FRAUD ALERT")
        else:
            st.success("Decision: LEGITIMATE")

        explanation_df = explain_transaction(model, single_df, transformed_feature_names, top_n=5)
        explanation_df["SHAP Value"] = explanation_df["SHAP Value"].round(4)
        explanation_df["Reason"] = explanation_df["Feature"].map(humanize_feature_name)

        st.subheader("Why the Model Flagged This Transaction")
        st.dataframe(
            explanation_df[["Reason", "SHAP Value"]],
            use_container_width=True,
            hide_index=True,
        )

        st.write("Threshold applied:", f"{threshold:.2f}")

with batch_tab:
    st.subheader("Batch Fraud Screening (CSV Upload)")
    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = normalize_input_columns(pd.read_csv(uploaded_file))
        missing_cols = [col for col in USER_INPUT_FEATURES if col not in batch_df.columns]

        if missing_cols:
            st.error("Uploaded file is missing required input columns.")
            st.write(missing_cols)
        else:
            score_input = batch_df[USER_INPUT_FEATURES].copy()
            score_input = assemble_model_input(score_input, feature_cols, default_inputs)
            score_input = score_input.replace([np.inf, -np.inf], np.nan)
            probabilities = model.predict_proba(score_input)[:, 1]
            predictions = (probabilities >= threshold).astype(int)

            scored_df = batch_df.copy()
            scored_df["fraud_probability"] = probabilities
            scored_df["fraud_prediction"] = predictions
            scored_df["risk_band"] = pd.cut(
                scored_df["fraud_probability"],
                bins=[0.0, 0.3, 0.7, 1.0],
                labels=["Low", "Medium", "High"],
                include_lowest=True,
            )

            st.success("Batch scoring complete.")
            st.dataframe(
                scored_df.sort_values("fraud_probability", ascending=False),
                use_container_width=True,
            )

            csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download scored results",
                data=csv_bytes,
                file_name="fraud_scored_results.csv",
                mime="text/csv",
            )
