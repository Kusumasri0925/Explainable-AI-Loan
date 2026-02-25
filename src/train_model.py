"""Train a loan approval model with preprocessing and save a sklearn pipeline."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

DATA_PATH = Path("data/loan_dataset.csv")
MODEL_PATH = Path("models/loan_pipeline.pkl")
TARGET_COLUMN = "Loan_Status"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Place loan_dataset.csv in data/."
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. Available: {list(df.columns)}"
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns
    numeric_columns = X.select_dtypes(exclude=["object", "category", "bool"]).columns

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "label_encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_columns),
            ("numeric", numeric_pipeline, numeric_columns),
        ]
    )

    ensemble_classifier = VotingClassifier(
        estimators=[
            ("random_forest", RandomForestClassifier(n_estimators=300, random_state=42)),
            ("gradient_boosting", GradientBoostingClassifier(random_state=42)),
            ("logistic_regression", LogisticRegression(max_iter=2000, random_state=42)),
        ],
        voting="soft",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", ensemble_classifier),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved sklearn Pipeline (preprocessor + model) to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
