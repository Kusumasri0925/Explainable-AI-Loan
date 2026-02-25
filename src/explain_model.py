"""Generate SHAP-based explanations for the trained loan approval model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain trained classification model with SHAP."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/loan_model.pkl"),
        help="Path to trained model file.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/loan_data.csv"),
        help="Path to dataset CSV.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="Loan_Status",
        help="Target column to drop from dataset before explanation.",
    )
    parser.add_argument(
        "--class-index",
        type=int,
        default=1,
        help="Class index used when SHAP returns class-wise values.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum rows used for SHAP explanation.",
    )
    parser.add_argument(
        "--importance-plot-path",
        type=Path,
        default=Path("models/feature_importance.png"),
        help="Output path for SHAP feature importance image.",
    )
    parser.add_argument(
        "--summary-plot-path",
        type=Path,
        default=Path("models/shap_summary.png"),
        help="Output path for SHAP summary image.",
    )
    return parser.parse_args()


def pick_class_shap_values(
    shap_values: object, class_index: int
) -> np.ndarray:
    """Normalize SHAP output into a 2D array [n_samples, n_features]."""
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


def extract_features(
    df: pd.DataFrame, target_column: str, max_samples: int
) -> pd.DataFrame:
    """Prepare feature matrix for explanation."""
    X = df.drop(columns=[target_column]) if target_column in df.columns else df.copy()
    if max_samples > 0 and len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=42)
    return X


def explain_pipeline_model(
    model: Pipeline, X_raw: pd.DataFrame, class_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SHAP values for a sklearn Pipeline with tree classifier."""
    preprocessor = model.named_steps.get("preprocessor")
    classifier = model.named_steps.get("classifier")
    if preprocessor is None or classifier is None:
        raise ValueError("Expected pipeline with 'preprocessor' and 'classifier' steps.")

    X_processed = preprocessor.transform(X_raw)
    X_processed = np.asarray(X_processed)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_processed)
    shap_2d = pick_class_shap_values(shap_values, class_index)
    return X_processed, shap_2d, np.asarray(feature_names)


def save_feature_importance_plot(
    shap_values_2d: np.ndarray, feature_names: np.ndarray, output_path: Path
) -> None:
    """Save a bar chart with mean absolute SHAP importance."""
    mean_abs_shap = np.abs(shap_values_2d).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1]

    sorted_names = feature_names[order]
    sorted_values = mean_abs_shap[order]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_names[::-1], sorted_values[::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.title("Feature Importance (SHAP)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_summary_plot(
    shap_values_2d: np.ndarray,
    X_processed: np.ndarray,
    feature_names: np.ndarray,
    output_path: Path,
) -> None:
    """Save SHAP summary plot image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(
        shap_values_2d,
        X_processed,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data_path}")

    model = joblib.load(args.model_path)
    df = pd.read_csv(args.data_path)
    X_raw = extract_features(df, args.target_column, args.max_samples)

    if isinstance(model, Pipeline):
        X_processed, shap_values_2d, feature_names = explain_pipeline_model(
            model, X_raw, args.class_index
        )
    else:
        # Fallback for non-pipeline classifiers.
        explainer = shap.Explainer(model, X_raw)
        shap_result = explainer(X_raw)
        shap_values_2d = pick_class_shap_values(shap_result.values, args.class_index)
        X_processed = np.asarray(X_raw)
        feature_names = np.asarray(X_raw.columns)

    save_feature_importance_plot(
        shap_values_2d=shap_values_2d,
        feature_names=feature_names,
        output_path=args.importance_plot_path,
    )
    save_summary_plot(
        shap_values_2d=shap_values_2d,
        X_processed=X_processed,
        feature_names=feature_names,
        output_path=args.summary_plot_path,
    )

    print(f"Saved SHAP feature importance plot to: {args.importance_plot_path}")
    print(f"Saved SHAP summary plot to: {args.summary_plot_path}")


if __name__ == "__main__":
    main()
