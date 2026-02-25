"""Data preprocessing utilities for loan approval modeling."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return numeric and categorical feature column names."""
    numeric_features = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [
        col for col in df.columns if col not in set(numeric_features)
    ]
    return numeric_features, categorical_features


def build_preprocessing_pipeline(df: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing transformer for numeric and categorical columns."""
    numeric_features, categorical_features = split_feature_types(df)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
