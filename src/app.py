"""Streamlit app for loan approval prediction with SHAP explanation."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        Image as ReportLabImage,
        ListFlowable,
        ListItem,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

MODEL_CANDIDATE_PATHS = [
    Path("models/loan_pipeline.pkl"),
    Path("models/loan_model.pkl"),
]
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "loan_dataset.csv"
TARGET_COLUMN = "Loan_Status"


@st.cache_resource
def load_model() -> Pipeline:
    for model_path in MODEL_CANDIDATE_PATHS:
        if model_path.exists():
            return joblib.load(model_path)
    raise FileNotFoundError(
        "Model not found. Expected one of: models/loan_pipeline.pkl, models/loan_model.pkl"
    )


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv("data/loan_dataset.csv")


def default_feature_values(df: pd.DataFrame, feature_names: list[str]) -> dict[str, object]:
    defaults: dict[str, object] = {}
    for feature in feature_names:
        if feature in df.columns and not df.empty:
            series = df[feature]
            if pd.api.types.is_numeric_dtype(series):
                defaults[feature] = float(series.median())
            else:
                mode = series.mode(dropna=True)
                defaults[feature] = mode.iloc[0] if not mode.empty else "Unknown"
        else:
            defaults[feature] = 0.0
    return defaults


def _columns_from_transformer(cols: object, all_columns: list[str]) -> list[str]:
    if isinstance(cols, slice):
        return all_columns[cols]
    if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
        return [str(c) for c in list(cols)]
    if cols is None:
        return []
    return [str(cols)]


def get_model_feature_schema(model: Pipeline) -> tuple[list[str], list[str], list[str]]:
    preprocessor = model.named_steps.get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "feature_names_in_"):
        return [], [], []

    expected_features = list(preprocessor.feature_names_in_)
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []

    transformers = getattr(preprocessor, "transformers_", None)
    if transformers is None:
        transformers = getattr(preprocessor, "transformers", [])

    for name, _, cols in transformers:
        resolved = _columns_from_transformer(cols, expected_features)
        if name == "numeric":
            numeric_columns.extend(resolved)
        elif name == "categorical":
            categorical_columns.extend(resolved)

    return expected_features, numeric_columns, categorical_columns


def sanitize_prediction_input(
    model: Pipeline,
    input_df: pd.DataFrame,
    defaults: dict[str, object],
) -> pd.DataFrame:
    expected_features, numeric_columns, categorical_columns = get_model_feature_schema(model)
    if not expected_features:
        return input_df.copy()

    safe_df = input_df.copy().reindex(columns=expected_features)

    for col in expected_features:
        if col not in safe_df.columns:
            safe_df[col] = defaults.get(col, np.nan)

    for col in numeric_columns:
        if col in safe_df.columns:
            safe_df[col] = pd.to_numeric(safe_df[col], errors="coerce")
            default_val = pd.to_numeric(defaults.get(col, 0.0), errors="coerce")
            fill_val = 0.0 if pd.isna(default_val) else float(default_val)
            safe_df[col] = safe_df[col].fillna(fill_val)

    for col in categorical_columns:
        if col in safe_df.columns:
            safe_df[col] = safe_df[col].astype(str).str.strip()
            safe_df[col] = safe_df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})
            fill_val = str(defaults.get(col, "Unknown")).strip() or "Unknown"
            safe_df[col] = safe_df[col].fillna(fill_val).astype(str)

    return safe_df.reindex(columns=expected_features)


def pick_class_shap_values(shap_values: object, class_index: int) -> np.ndarray:
    if isinstance(shap_values, list):
        idx = min(max(class_index, 0), len(shap_values) - 1)
        return np.asarray(shap_values[idx])

    values = np.asarray(shap_values)
    if values.ndim == 3:
        idx = min(max(class_index, 0), values.shape[2] - 1)
        return values[:, :, idx]
    if values.ndim == 2:
        return values
    raise ValueError(f"Unsupported SHAP values shape: {values.shape}")


def is_approved_label(label: object) -> bool:
    """
    Returns True if the model label means approval.
    Handles Y/N, 1/0, Yes/No, Approved/Rejected.
    """
    normalized = str(label).strip().lower()

    return normalized in {"y", "yes", "1", "true", "approved"}


def get_approval_class_index(classifier: object, fallback_label: object) -> int:
    if not hasattr(classifier, "classes_"):
        return 0

    classes = list(classifier.classes_)
    normalized = [str(c).strip().lower() for c in classes]

    for candidate in ("approved", "approve", "yes", "true", "1"):
        if candidate in normalized:
            return normalized.index(candidate)

    if fallback_label in classes:
        return classes.index(fallback_label)

    if len(classes) > 1:
        return 1
    return 0


def get_risk_level(probability: float) -> str:
    if probability > 0.75:
        return "Low Risk"
    if probability >= 0.50:
        return "Medium Risk"
    return "High Risk"


def get_recommendation(risk_level: str) -> str:
    if risk_level == "Low Risk":
        return "Profile is strong for approval. Proceed with the application."
    if risk_level == "Medium Risk":
        return "Improve income-to-loan ratio or credit score to increase approval odds."
    return "High rejection risk. Reduce loan amount and improve credit profile before reapplying."


def get_numeric_defaults(df: pd.DataFrame) -> tuple[float, float, int]:
    income_default = (
        float(df["Income"].median())
        if not df.empty and "Income" in df.columns
        else 40000.0
    )
    loan_default = (
        float(df["LoanAmount"].median())
        if not df.empty and "LoanAmount" in df.columns
        else 150000.0
    )
    score_default = (
        int(df["CreditScore"].median())
        if not df.empty and "CreditScore" in df.columns
        else 700
    )
    return income_default, loan_default, score_default


def predict_with_probability(model: Pipeline, input_df: pd.DataFrame) -> tuple[object, float]:
    prediction = model.predict(input_df)[0]
    classifier = model.named_steps.get("classifier", model)
    approval_prob = 1.0 if is_approved_label(prediction) else 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        approval_idx = get_approval_class_index(classifier, prediction)
        approval_prob = float(probabilities[approval_idx])
    return prediction, approval_prob


def get_individual_model_predictions(
    model: Pipeline, input_df: pd.DataFrame, fallback_prediction: object
) -> list[dict[str, object]]:
    preprocessor = model.named_steps.get("preprocessor")
    classifier = model.named_steps.get("classifier", model)
    if preprocessor is None:
        return []

    if not hasattr(classifier, "estimators_"):
        model_prob = 1.0 if is_approved_label(fallback_prediction) else 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            approval_idx = get_approval_class_index(classifier, fallback_prediction)
            model_prob = float(probs[approval_idx])
        return [
            {
                "model": type(classifier).__name__,
                "prediction": fallback_prediction,
                "approval_probability": model_prob,
            }
        ]

    X_processed = preprocessor.transform(input_df)
    rows: list[dict[str, object]] = []

    named_estimators = getattr(classifier, "named_estimators_", {})
    if named_estimators:
        iterator = named_estimators.items()
    else:
        iterator = [
            (f"model_{idx+1}", est)
            for idx, est in enumerate(getattr(classifier, "estimators_", []))
        ]

    for name, estimator in iterator:
        est_prediction = estimator.predict(X_processed)[0]
        est_prob = 1.0 if is_approved_label(est_prediction) else 0.0
        if hasattr(estimator, "predict_proba"):
            est_probs = estimator.predict_proba(X_processed)[0]
            est_idx = get_approval_class_index(estimator, est_prediction)
            if est_idx < len(est_probs):
                est_prob = float(est_probs[est_idx])
        rows.append(
            {
                "model": str(name),
                "prediction": est_prediction,
                "approval_probability": est_prob,
            }
        )

    return rows


def render_individual_model_predictions(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    model_df = pd.DataFrame(rows)
    display_df = model_df.copy()
    display_df["approval_probability"] = (
        display_df["approval_probability"] * 100
    ).map(lambda x: f"{x:.2f}%")
    st.dataframe(
        display_df.rename(
            columns={
                "model": "Model",
                "prediction": "Prediction",
                "approval_probability": "Approval Probability",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    chart_df = pd.DataFrame(rows).sort_values("approval_probability", ascending=False)
    fig = go.Figure(
        data=[
            go.Bar(
                x=chart_df["model"],
                y=chart_df["approval_probability"] * 100,
                marker_color="#1f77b4",
                text=(chart_df["approval_probability"] * 100).map(lambda x: f"{x:.1f}%"),
                textposition="outside",
                hovertemplate="%{x}<br>Approval Probability: %{y:.2f}%<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Individual Model Approval Probabilities",
        xaxis_title="Base Model",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def build_input_row(model: Pipeline, raw_inputs: dict[str, object], df: pd.DataFrame) -> pd.DataFrame:
    preprocessor = model.named_steps.get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "feature_names_in_"):
        return pd.DataFrame([raw_inputs])

    expected_features = list(preprocessor.feature_names_in_)
    values = default_feature_values(df, expected_features)
    values.update(raw_inputs)
    return pd.DataFrame([values], columns=expected_features)


def evaluate_profile(
    model: Pipeline,
    df: pd.DataFrame,
    defaults: dict[str, object],
    profile_inputs: dict[str, object],
) -> tuple[pd.DataFrame, object, float]:
    raw_input_df = build_input_row(model, profile_inputs, df)
    input_df = sanitize_prediction_input(model, raw_input_df, defaults)
    prediction, approval_prob = predict_with_probability(model, input_df)
    return input_df, prediction, approval_prob


def render_probability_gauge(approval_prob: float) -> None:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=approval_prob * 100,
            number={"suffix": "%", "valueformat": ".2f"},
            title={"text": "Approval Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 50], "color": "#fde2e2"},
                    {"range": [50, 75], "color": "#fff4cc"},
                    {"range": [75, 100], "color": "#dff5e1"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_comparison_probability_chart(
    prob_a: float, prob_b: float, label_a: str = "Profile A", label_b: str = "Profile B"
) -> None:
    fig = go.Figure(
        data=[
            go.Bar(
                x=[label_a, label_b],
                y=[prob_a * 100, prob_b * 100],
                marker_color=["#1f77b4", "#ff7f0e"],
                text=[f"{prob_a * 100:.2f}%", f"{prob_b * 100:.2f}%"],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Approval Probability Comparison",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_contribution_bar(explanation: shap.Explanation) -> None:
    values = np.asarray(explanation.values)
    names = np.asarray(explanation.feature_names)
    if values.ndim != 1 or names.ndim != 1:
        st.info("Feature contribution chart unavailable for this prediction.")
        return

    top_n = min(12, len(values))
    order = np.argsort(np.abs(values))[::-1][:top_n]
    contrib_df = pd.DataFrame(
        {
            "feature": names[order],
            "contribution": values[order],
        }
    ).sort_values("contribution")

    colors = np.where(contrib_df["contribution"] >= 0, "#2ca02c", "#d62728")
    fig = go.Figure(
        go.Bar(
            x=contrib_df["contribution"],
            y=contrib_df["feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>Contribution: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Top Feature Contributions (SHAP)",
        xaxis_title="SHAP Value",
        yaxis_title="Feature",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def normalize_feature_name(feature_name: str) -> str:
    base = feature_name.split("__")[-1]
    lower = base.lower()
    for candidate in ("income", "loanamount", "creditscore", "education", "married"):
        if candidate in lower:
            return candidate
    return lower


def build_shap_recommendations(
    explanation: shap.Explanation, user_inputs: dict[str, object]
) -> list[str]:
    values = np.asarray(explanation.values)
    names = np.asarray(explanation.feature_names)
    if values.ndim != 1 or names.ndim != 1:
        return ["Review the applicant profile and improve the weakest financial factors."]

    negative_idx = np.where(values < 0)[0]
    if len(negative_idx) == 0:
        return ["No strong negative contributors detected; current profile is supportive."]

    ranked = sorted(
        negative_idx,
        key=lambda idx: abs(values[idx]),
        reverse=True,
    )[:4]

    suggestions: list[str] = []
    for idx in ranked:
        feature_key = normalize_feature_name(str(names[idx]))
        if feature_key == "income":
            suggestions.append(
                f"Increase stable income from the current value ({user_inputs['Income']:.0f}) to improve repayment capacity."
            )
        elif feature_key == "loanamount":
            suggestions.append(
                f"Reduce requested loan amount from {user_inputs['LoanAmount']:.0f} to lower risk exposure."
            )
        elif feature_key == "creditscore":
            suggestions.append(
                f"Improve credit score above the current level ({int(user_inputs['CreditScore'])}) by reducing debt utilization and paying on time."
            )
        elif feature_key == "education":
            suggestions.append(
                "Strengthen profile with additional qualification or documented professional stability to offset education-related risk."
            )
        elif feature_key == "married":
            suggestions.append(
                "Provide stronger co-applicant or household financial documentation to reduce relationship-status related risk."
            )
        else:
            suggestions.append(
                f"Improve '{names[idx]}' since it currently lowers approval probability."
            )

    deduped: list[str] = []
    for rec in suggestions:
        if rec not in deduped:
            deduped.append(rec)
    return deduped


def render_shap_recommendations(
    recommendations: list[str],
) -> None:
    st.subheader("Recommendations To Improve Approval")
    for rec in recommendations:
        st.markdown(f"- {rec}")


def build_shap_plot_image_bytes(explanation: shap.Explanation) -> bytes:
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format="png", bbox_inches="tight", dpi=200)
    img_buffer.seek(0)
    plt.close(fig)
    return img_buffer.getvalue()


def build_prediction_summary_pdf(
    user_inputs: dict[str, object],
    final_prediction: object,
    approval_prob: float,
    risk_level: str,
    recommendation: str,
    shap_recommendations: list[str],
    shap_plot_image_bytes: bytes | None,
) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is not installed")

    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story: list[object] = []

    story.append(Paragraph("Loan Prediction Summary", styles["Title"]))
    story.append(Spacer(1, 10))
    final_decision = "Approved" if is_approved_label(final_prediction) else "Rejected"
    story.append(Paragraph(f"Final Decision: <b>{final_decision}</b>", styles["Heading3"]))
    story.append(
        Paragraph(f"Approval Probability: <b>{approval_prob * 100:.2f}%</b>", styles["BodyText"])
    )
    story.append(Paragraph(f"Risk Level: <b>{risk_level}</b>", styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("User Inputs", styles["Heading3"]))
    for key, value in user_inputs.items():
        story.append(Paragraph(f"{key}: {value}", styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Recommendation", styles["Heading3"]))
    story.append(Paragraph(recommendation, styles["BodyText"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Suggested Improvements", styles["Heading3"]))
    bullet_items = [
        ListItem(Paragraph(text, styles["BodyText"])) for text in shap_recommendations
    ]
    story.append(ListFlowable(bullet_items, bulletType="bullet"))
    story.append(Spacer(1, 10))

    if shap_plot_image_bytes:
        story.append(Paragraph("SHAP Waterfall Explanation", styles["Heading3"]))
        story.append(Spacer(1, 6))
        story.append(ReportLabImage(BytesIO(shap_plot_image_bytes), width=500, height=280))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def render_shap_explanation(
    model: Pipeline, input_df: pd.DataFrame, df: pd.DataFrame, predicted_label: object
) -> tuple[shap.Explanation | None, bytes | None]:
    preprocessor = model.named_steps.get("preprocessor")
    classifier = model.named_steps.get("classifier")
    if preprocessor is None or classifier is None:
        st.info("SHAP visualization is unavailable for this model format.")
        return None, None

    # Build SHAP context for a single-user prediction, using dataset rows as background.
    background_df = df.drop(columns=[TARGET_COLUMN], errors="ignore")
    if background_df.empty:
        background_df = input_df.copy()
    if len(background_df) > 200:
        background_df = background_df.sample(n=200, random_state=42)

    X_background = preprocessor.transform(background_df)
    X_input = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()

    class_index = get_approval_class_index(classifier, predicted_label)
    try:
        explainer = shap.TreeExplainer(classifier, X_background)
        shap_values = explainer.shap_values(X_input)
        shap_2d = pick_class_shap_values(shap_values, class_index)
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = np.asarray(expected_value)[class_index]
    except Exception:
        try:
            explainer = shap.Explainer(classifier.predict_proba, X_background)
            shap_result = explainer(X_input)
            shap_2d = pick_class_shap_values(shap_result.values, class_index)
            expected_raw = np.asarray(shap_result.base_values)
            if expected_raw.ndim == 2:
                expected_value = expected_raw[0, class_index]
            elif expected_raw.ndim == 1 and len(expected_raw) > class_index:
                expected_value = expected_raw[class_index]
            else:
                expected_value = float(np.ravel(expected_raw)[0])
        except Exception:
            st.info("SHAP explanation is unavailable for this ensemble configuration.")
            return None, None

    explanation = shap.Explanation(
        values=shap_2d[0],
        base_values=expected_value,
        data=np.asarray(X_input)[0],
        feature_names=feature_names,
    )

    shap_plot_bytes = build_shap_plot_image_bytes(explanation)
    st.image(shap_plot_bytes, use_container_width=True)
    return explanation, shap_plot_bytes


def render_model_insights(model: Pipeline, df: pd.DataFrame) -> None:
    st.subheader("Model Insights")
    if df.empty:
        st.warning("Dataset not found at data/loan_dataset.csv. Model insights are unavailable.")
        return
    if TARGET_COLUMN not in df.columns:
        st.info(f"'{TARGET_COLUMN}' not found in dataset. Unable to compute evaluation metrics.")
        return

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    preprocessor = model.named_steps.get("preprocessor")
    expected_features = (
        list(preprocessor.feature_names_in_)
        if preprocessor is not None and hasattr(preprocessor, "feature_names_in_")
        else list(X.columns)
    )
    defaults = default_feature_values(df, expected_features)
    X_eval = sanitize_prediction_input(model, X, defaults)

    try:
        y_pred = model.predict(X_eval)
    except Exception as exc:
        st.error(f"Failed to evaluate model on dataset: {exc}")
        return

    classifier = model.named_steps.get("classifier", model)
    y_true = pd.Series(y).astype(str)
    y_pred_series = pd.Series(y_pred).astype(str)
    model_classes = [str(lbl) for lbl in getattr(classifier, "classes_", [])]
    labels = model_classes if model_classes else sorted(y_true.unique().tolist())
    positive_idx = get_approval_class_index(
        classifier, y_pred[0] if len(y_pred) else (labels[0] if labels else "")
    )
    positive_label = str(labels[positive_idx]) if labels else None

    metrics_cols = st.columns(4)
    metrics_cols[0].metric("Accuracy", f"{accuracy_score(y_true, y_pred_series):.3f}")
    if positive_label is not None and len(labels) == 2:
        precision = precision_score(
            y_true, y_pred_series, pos_label=positive_label, zero_division=0
        )
        recall = recall_score(
            y_true, y_pred_series, pos_label=positive_label, zero_division=0
        )
        f1 = f1_score(y_true, y_pred_series, pos_label=positive_label, zero_division=0)
    else:
        precision = precision_score(y_true, y_pred_series, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred_series, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred_series, average="macro", zero_division=0)
    metrics_cols[1].metric("Precision", f"{precision:.3f}")
    metrics_cols[2].metric("Recall", f"{recall:.3f}")
    metrics_cols[3].metric("F1 Score", f"{f1:.3f}")

    cm = confusion_matrix(y_true, y_pred_series, labels=labels)
    cm_fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[str(lbl) for lbl in labels],
            y=[str(lbl) for lbl in labels],
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
        )
    )
    cm_fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(cm_fig, use_container_width=True)

    if hasattr(model, "predict_proba") and len(labels) == 2 and positive_label is not None:
        try:
            y_prob = model.predict_proba(X_eval)[:, positive_idx]
            y_true_bin = (y_true == positive_label).astype(int)
            fpr, tpr, _ = roc_curve(y_true_bin, y_prob)
            auc_score = roc_auc_score(y_true_bin, y_prob)
            roc_fig = go.Figure()
            roc_fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"ROC (AUC = {auc_score:.3f})",
                    line=dict(color="#1f77b4", width=3),
                )
            )
            roc_fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random",
                    line=dict(color="#888", dash="dash"),
                )
            )
            roc_fig.update_layout(
                title="ROC Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                yaxis_range=[0, 1],
                xaxis_range=[0, 1],
                height=420,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(roc_fig, use_container_width=True)
        except Exception as exc:
            st.warning(f"ROC curve could not be generated: {exc}")
    else:
        st.info("ROC curve is available for binary classifiers with probability output.")


def render_prediction_tab(model: Pipeline, df: pd.DataFrame) -> None:
    income_default, loan_default, score_default = get_numeric_defaults(df)

    education_options = ["Graduate", "Not Graduate"]
    if "Education" in df.columns:
        vals = [v for v in df["Education"].dropna().unique().tolist() if str(v)]
        if vals:
            education_options = vals
    married_options = ["Yes", "No"]
    if "Married" in df.columns:
        vals = [v for v in df["Married"].dropna().unique().tolist() if str(v)]
        if vals:
            married_options = vals

    st.subheader("Profile A")
    col1, col2 = st.columns(2)
    with col1:
        income = st.slider(
            "Income",
            min_value=0.0,
            max_value=max(200000.0, income_default * 2),
            value=income_default,
            step=1000.0,
            key="income_a",
        )
        credit_score = st.slider(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=score_default,
            step=10,
            key="credit_score_a",
        )
        education = st.selectbox("Education", options=education_options, key="education_a")
    with col2:
        loan_amount = st.slider(
            "Loan Amount",
            min_value=0.0,
            max_value=max(500000.0, loan_default * 2),
            value=loan_default,
            step=5000.0,
            key="loan_amount_a",
        )
        married_status = st.selectbox(
            "Married Status", options=married_options, key="married_a"
        )

    current_inputs = {
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "Education": education,
        "Married": married_status,
    }
    credit_history = 1 if credit_score > 650 else 0
    input_payload = {
        "Loan_ID": "LP001",
        "Gender": "Male",
        "Married": married_status,
        "Dependents": "0",
        "Education": education,
        "Self_Employed": "No",
        "ApplicantIncome": income,
        "CoapplicantIncome": 0.0,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": 360.0,
        "Credit_History": float(credit_history),
        "Property_Area": "Urban",
    }
    baseline_credit_history = 1 if score_default > 650 else 0
    baseline_inputs = {
        "Loan_ID": "LP001",
        "Gender": "Male",
        "Married": married_options[0],
        "Dependents": "0",
        "Education": education_options[0],
        "Self_Employed": "No",
        "ApplicantIncome": income_default,
        "CoapplicantIncome": 0.0,
        "LoanAmount": loan_default,
        "Loan_Amount_Term": 360.0,
        "Credit_History": float(baseline_credit_history),
        "Property_Area": "Urban",
    }

    preprocessor = model.named_steps.get("preprocessor")
    expected_features = (
        list(preprocessor.feature_names_in_)
        if preprocessor is not None and hasattr(preprocessor, "feature_names_in_")
        else list(input_payload.keys())
    )
    defaults = default_feature_values(df, expected_features)

    try:
        input_df, prediction, approval_prob = evaluate_profile(
            model, df, defaults, input_payload
        )
        baseline_df, _, baseline_prob = evaluate_profile(
            model, df, defaults, baseline_inputs
        )
    except Exception:
        st.warning("Some inputs were invalid and have been replaced with safe defaults.")
        try:
            input_df, prediction, approval_prob = evaluate_profile(
                model, df, defaults, baseline_inputs
            )
            baseline_df, _, baseline_prob = evaluate_profile(
                model, df, defaults, baseline_inputs
            )
        except Exception as exc:
            st.error(f"Unable to generate prediction safely: {exc}")
            st.stop()

    comparison_mode = st.checkbox(
        "Enable comparison mode (duplicate and modify inputs)", value=False
    )
    comparison_input_df = None
    comparison_prediction = None
    comparison_prob = None
    if comparison_mode:
        st.subheader("Profile B")
        col3, col4 = st.columns(2)
        with col3:
            income_b = st.slider(
                "Income (B)",
                min_value=0.0,
                max_value=max(200000.0, income_default * 2),
                value=float(income),
                step=1000.0,
                key="income_b",
            )
            credit_score_b = st.slider(
                "Credit Score (B)",
                min_value=300,
                max_value=900,
                value=int(credit_score),
                step=10,
                key="credit_score_b",
            )
            education_b = st.selectbox(
                "Education (B)",
                options=education_options,
                index=education_options.index(education),
                key="education_b",
            )
        with col4:
            loan_amount_b = st.slider(
                "Loan Amount (B)",
                min_value=0.0,
                max_value=max(500000.0, loan_default * 2),
                value=float(loan_amount),
                step=5000.0,
                key="loan_amount_b",
            )
            married_b = st.selectbox(
                "Married Status (B)",
                options=married_options,
                index=married_options.index(married_status),
                key="married_b",
            )

        comparison_credit_history = 1 if credit_score_b > 650 else 0
        comparison_inputs = {
            "Loan_ID": "LP001",
            "Gender": "Male",
            "Married": married_b,
            "Dependents": "0",
            "Education": education_b,
            "Self_Employed": "No",
            "ApplicantIncome": income_b,
            "CoapplicantIncome": 0.0,
            "LoanAmount": loan_amount_b,
            "Loan_Amount_Term": 360.0,
            "Credit_History": float(comparison_credit_history),
            "Property_Area": "Urban",
        }
        try:
            comparison_input_df, comparison_prediction, comparison_prob = evaluate_profile(
                model, df, defaults, comparison_inputs
            )
        except Exception as exc:
            st.error(f"Unable to generate comparison prediction: {exc}")
            comparison_mode = False
    prob_delta = approval_prob - baseline_prob

    if is_approved_label(prediction):
       st.success("Final Ensemble Decision: Approved")
    else:
       st.error("Final Ensemble Decision: Rejected")

    st.metric(
        "Approval Probability",
        f"{approval_prob * 100:.2f}%",
        delta=f"{prob_delta * 100:+.2f}% vs baseline",
    )
    render_probability_gauge(approval_prob)
    st.caption(
        f"Baseline profile uses Income {income_default:.0f}, Loan Amount {loan_default:.0f}, "
        f"Credit Score {score_default}, Education {education_options[0]}, Married {married_options[0]}."
    )

    risk_level = get_risk_level(approval_prob)
    recommendation = get_recommendation(risk_level)
    if risk_level == "Low Risk":
        st.success(f"Risk Level: {risk_level}")
    elif risk_level == "Medium Risk":
        st.warning(f"Risk Level: {risk_level}")
    else:
        st.error(f"Risk Level: {risk_level}")
    st.info(f"Recommendation: {recommendation}")

    st.subheader("Individual Model Predictions")
    individual_rows = get_individual_model_predictions(model, input_df, prediction)
    render_individual_model_predictions(individual_rows)

    if comparison_mode and comparison_prob is not None and comparison_prediction is not None:
        st.subheader("Side-by-Side Comparison")
        left, right = st.columns(2)
        with left:
            st.markdown("**Profile A**")
            st.write(
    f"Prediction: {'Approved' if is_approved_label(prediction) else 'Rejected'}"
)
            st.metric("Approval Probability (A)", f"{approval_prob * 100:.2f}%")
            render_probability_gauge(approval_prob)
        with right:
            st.markdown("**Profile B**")
            st.write(
    f"Prediction: {'Approved' if str(comparison_prediction).lower() == 'approved' else 'Rejected'}"
)
            st.metric(
                "Approval Probability (B)",
                f"{comparison_prob * 100:.2f}%",
                delta=f"{(comparison_prob - approval_prob) * 100:+.2f}% vs A",
            )
            render_probability_gauge(comparison_prob)

        render_comparison_probability_chart(approval_prob, comparison_prob)
        if comparison_input_df is not None:
            st.subheader("Individual Model Predictions (Profile B)")
            comparison_rows = get_individual_model_predictions(
                model, comparison_input_df, comparison_prediction
            )
            render_individual_model_predictions(comparison_rows)

    st.subheader("SHAP Explanation")
    explanation, shap_plot_bytes = render_shap_explanation(model, input_df, df, prediction)
    if explanation is not None:
        st.subheader("Feature Contributions")
        render_feature_contribution_bar(explanation)
        shap_recommendations = build_shap_recommendations(explanation, current_inputs)
        render_shap_recommendations(shap_recommendations)
    else:
        shap_recommendations = [recommendation]

    st.subheader("Export Summary")
    if REPORTLAB_AVAILABLE:
        try:
            pdf_bytes = build_prediction_summary_pdf(
                user_inputs=current_inputs,
                final_prediction=prediction,
                approval_prob=approval_prob,
                risk_level=risk_level,
                recommendation=recommendation,
                shap_recommendations=shap_recommendations,
                shap_plot_image_bytes=shap_plot_bytes,
            )
            st.download_button(
                "Download Prediction Summary (PDF)",
                data=pdf_bytes,
                file_name="prediction_summary.pdf",
                mime="application/pdf",
            )
        except Exception as exc:
            st.warning(f"PDF export failed: {exc}")
    else:
        st.info("Install reportlab to enable PDF export (`pip install reportlab`).")


def main() -> None:
    st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
    st.title("Loan Approval Predictor")
    st.write("Predict loan approvals and inspect model performance.")

    try:
        model = load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    df = load_dataset()
    if df.empty:
        st.warning("Dataset not found at data/loan_dataset.csv. Running app with limited features.")
    prediction_tab, insights_tab = st.tabs(["Loan Prediction", "Model Insights"])
    with prediction_tab:
        render_prediction_tab(model, df)
    with insights_tab:
        render_model_insights(model, df)


if __name__ == "__main__":
    main()
