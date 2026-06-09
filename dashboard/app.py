from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebook" / "model.ipynb"


def load_notebook() -> dict:
    with NOTEBOOK_PATH.open("r", encoding="utf-8") as notebook_file:
        return json.load(notebook_file)


def _cell_text_outputs(cell: dict) -> str:
    parts: list[str] = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            parts.append(text)
    return "\n".join(parts)


def _find_output_text(notebook: dict, source_contains: str) -> str:
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if source_contains in source:
            text = _cell_text_outputs(cell)
            if text.strip():
                return text
    raise ValueError(f"Could not find output for {source_contains!r} in {NOTEBOOK_PATH}")


def load_model_comparison(notebook: dict) -> pd.DataFrame:
    comparison_text = _find_output_text(notebook, "comparison_df = pd.DataFrame")
    rows = []
    pattern = re.compile(
        r"^\s*\d+\s+(.*?)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$",
        re.MULTILINE,
    )
    for model, accuracy, precision, recall, f1_score, roc_auc in pattern.findall(comparison_text):
        rows.append(
            {
                "Model": model.strip(),
                "Accuracy": float(accuracy),
                "Precision": float(precision),
                "Recall": float(recall),
                "F1-Score": float(f1_score),
                "ROC-AUC": float(roc_auc),
            }
        )

    if not rows:
        raise ValueError("No model comparison rows were parsed from the notebook output.")

    return pd.DataFrame(rows)


def load_tuned_confusion_matrix(notebook: dict) -> list[list[int]]:
    tuned_text = _find_output_text(notebook, "Tuned XGBoost Confusion Matrix:")
    match = re.search(
        r"\[\[(\d+)\s+(\d+)\]\s*\[\s*(\d+)\s+(\d+)\]\]",
        tuned_text.replace("\n", " "),
    )
    if not match:
        raise ValueError("Could not parse tuned XGBoost confusion matrix from notebook output.")
    return [
        [int(match.group(1)), int(match.group(2))],
        [int(match.group(3)), int(match.group(4))],
    ]


def load_feature_importance(notebook: dict) -> pd.DataFrame:
    feature_text = _find_output_text(notebook, "feature_importance.head(10)")
    rows = []
    pattern = re.compile(r"^\s*\d+\s+(.*?)\s+([0-9.]+)\s*$", re.MULTILINE)
    for feature, importance in pattern.findall(feature_text):
        rows.append({"Feature": feature.strip(), "Importance": float(importance)})

    if not rows:
        raise ValueError("No feature importance rows were parsed from the notebook output.")

    return pd.DataFrame(rows)


def render_metric_bars_svg(data: pd.DataFrame, title: str) -> str:
    metrics = [col for col in data.columns if col != "Model"]
    chart_width = 1120
    height = 360
    left_pad = 90
    right_pad = 24
    top_pad = 44
    bottom_pad = 68
    plot_height = height - top_pad - bottom_pad
    group_width = (chart_width - left_pad - right_pad) / len(metrics)
    bar_width = 44
    colors = ["#1D4ED8"]

    parts = [
        f'<div class="chart-card"><div class="chart-title">{html.escape(title)}</div>',
        f'<svg viewBox="0 0 {chart_width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<rect x="0" y="0" width="{chart_width}" height="{height}" rx="18" fill="#FFFFFF" stroke="#E5E7EB"/>',
        f'<line x1="{left_pad}" y1="{top_pad}" x2="{left_pad}" y2="{height - bottom_pad}" stroke="#CBD5E1" stroke-width="1.5"/>',
        f'<line x1="{left_pad}" y1="{height - bottom_pad}" x2="{chart_width - right_pad}" y2="{height - bottom_pad}" stroke="#CBD5E1" stroke-width="1.5"/>',
    ]

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top_pad + plot_height - (tick * plot_height)
        parts.append(f'<line x1="{left_pad - 6}" y1="{y}" x2="{left_pad}" y2="{y}" stroke="#94A3B8"/>')
        parts.append(
            f'<text x="{left_pad - 12}" y="{y + 4}" text-anchor="end" font-size="12" fill="#64748B">{tick:.2f}</text>'
        )

    for metric_index, metric in enumerate(metrics):
        x_center = left_pad + metric_index * group_width + group_width / 2
        value = float(data.loc[:, metric].iloc[0])
        bar_height = value * plot_height
        x = x_center - bar_width / 2
        y = top_pad + plot_height - bar_height
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="8" fill="{colors[0]}"/>'
        )
        parts.append(
            f'<text x="{x_center:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="11" fill="#0F172A">{value:.3f}</text>'
        )
        parts.append(
            f'<text x="{x_center:.1f}" y="{height - 24}" text-anchor="middle" font-size="12" fill="#334155">{html.escape(metric)}</text>'
        )

    parts.append(f'<text x="{left_pad + 450}" y="20" text-anchor="middle" font-size="12" fill="#334155">Tuned XGBoost</text>')
    parts.append("</svg></div>")
    return "".join(parts)


def render_confusion_matrix_svg(cm: list[list[int]], title: str) -> str:
    labels = [["TN", "FP"], ["FN", "TP"]]
    max_value = max(max(row) for row in cm)
    width = 680
    height = 360
    cell = 130
    left = 120
    top = 70
    parts = [
        f'<div class="chart-card"><div class="chart-title">{html.escape(title)}</div>',
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="18" fill="#FFFFFF" stroke="#E5E7EB"/>',
    ]

    for row in range(2):
        for col in range(2):
            value = cm[row][col]
            intensity = value / max_value if max_value else 0
            color = f"rgb({245 - int(120 * intensity)}, {250 - int(60 * intensity)}, {243 - int(120 * intensity)})"
            x = left + col * cell
            y = top + row * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#E2E8F0"/>')
            parts.append(
                f'<text x="{x + cell / 2}" y="{y + 48}" text-anchor="middle" font-size="16" fill="#1F2937" font-weight="600">{labels[row][col]}</text>'
            )
            parts.append(
                f'<text x="{x + cell / 2}" y="{y + 82}" text-anchor="middle" font-size="18" fill="#111827">{value}</text>'
            )

    parts.append(
        f'<text x="{left - 52}" y="{top + cell * 0.5 + 6}" text-anchor="middle" font-size="12" fill="#475569" transform="rotate(-90 {left - 52} {top + cell * 0.5 + 6})">Actual Label</text>'
    )
    parts.append(f'<text x="{left + cell}" y="{top + cell * 2 + 46}" text-anchor="middle" font-size="12" fill="#475569">Predicted Label</text>')
    parts.append(f'<text x="{left + cell * 0.5}" y="{top - 16}" text-anchor="middle" font-size="12" fill="#475569">0</text>')
    parts.append(f'<text x="{left + cell * 1.5}" y="{top - 16}" text-anchor="middle" font-size="12" fill="#475569">1</text>')
    parts.append(f'<text x="{left - 20}" y="{top + cell * 0.5}" text-anchor="middle" font-size="12" fill="#475569">0</text>')
    parts.append(f'<text x="{left - 20}" y="{top + cell * 1.5}" text-anchor="middle" font-size="12" fill="#475569">1</text>')
    parts.append("</svg></div>")
    return "".join(parts)


def render_feature_importance_svg(features: pd.DataFrame, title: str) -> str:
    width = 1120
    height = max(380, 60 + 36 * len(features))
    left = 280
    right = 30
    top = 44
    row_h = 32
    plot_width = width - left - right
    max_value = float(features["Importance"].max())
    parts = [
        f'<div class="chart-card"><div class="chart-title">{html.escape(title)}</div>',
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="18" fill="#FFFFFF" stroke="#E5E7EB"/>',
    ]

    for index, row in features.reset_index(drop=True).iterrows():
        y = top + index * row_h
        bar_width = 0 if max_value == 0 else (float(row["Importance"]) / max_value) * plot_width
        parts.append(f'<text x="{left - 12}" y="{y + 18}" text-anchor="end" font-size="13" fill="#334155">{html.escape(str(row["Feature"]))}</text>')
        parts.append(f'<rect x="{left}" y="{y + 4}" width="{bar_width:.1f}" height="18" rx="6" fill="#2563EB"/>')
        parts.append(f'<text x="{left + bar_width + 8:.1f}" y="{y + 18}" font-size="12" fill="#0F172A">{row["Importance"]:.3f}</text>')

    parts.append("</svg></div>")
    return "".join(parts)


st.set_page_config(page_title="Fraud Detection Model Dashboard", layout="wide")

notebook = load_notebook()
comparison_df = load_model_comparison(notebook)
tuned_cm = load_tuned_confusion_matrix(notebook)
feature_importance_df = load_feature_importance(notebook)

tuned_row = comparison_df.loc[comparison_df["Model"] == "Tuned XGBoost"].iloc[0]

st.title("Fraud Detection Model Performance Dashboard")
st.write(
    "This dashboard is populated from the final outputs in notebook/model.ipynb so stakeholders see the same evaluation results used in the analysis."
)

st.caption(f"Source notebook: {NOTEBOOK_PATH.name}")

st.subheader("Model Performance Comparison")
st.dataframe(comparison_df, width="stretch")

metric_left, metric_mid, metric_right, metric_fourth = st.columns(4)
with metric_left:
    st.metric("Tuned Accuracy", f'{tuned_row["Accuracy"]:.3f}')
with metric_mid:
    st.metric("Tuned Precision", f'{tuned_row["Precision"]:.3f}')
with metric_right:
    st.metric("Tuned Recall", f'{tuned_row["Recall"]:.3f}')
with metric_fourth:
    st.metric("Tuned ROC-AUC", f'{tuned_row["ROC-AUC"]:.3f}')

st.markdown(render_metric_bars_svg(tuned_row.to_frame().T, "Tuned XGBoost Metrics"), unsafe_allow_html=True)
st.markdown(render_confusion_matrix_svg(tuned_cm, "Tuned XGBoost Confusion Matrix"), unsafe_allow_html=True)

st.subheader("Explainability AI")
st.markdown(render_feature_importance_svg(feature_importance_df.head(10), "Top Explainability Drivers from Notebook"), unsafe_allow_html=True)

top_features = feature_importance_df.head(3)["Feature"].tolist()
st.markdown(
    f"""
- The notebook's explainability output shows **txn_velocity_24h** and **txn_velocity_1h** as the dominant fraud signals.
- **log_exchange_rate**, **account_age_days**, and **risk_score_internal** are the next strongest drivers.
- This means the tuned model is reacting to sudden bursts in transaction behavior, account maturity, and risk signals rather than a single isolated variable.
- Top three explainability drivers: {", ".join(top_features)}.
"""
)

st.subheader("Key Findings")
st.markdown(
    """
- Tuned XGBoost leads the notebook's final comparison on accuracy, precision, F1-score, and ROC-AUC.
- The confusion matrix shows a very small false-positive count, which is useful for stakeholder trust.
- Explainability results suggest the model is using transaction velocity and risk context to identify fraud patterns.
"""
)

st.subheader("Business Recommendation")
st.success(
    "Tuned XGBoost is recommended as the primary fraud detection model because it leads the notebook's overall balance of precision, F1-score, and ROC-AUC while the explainability output shows sensible fraud drivers aligned with transaction velocity and risk indicators."
)