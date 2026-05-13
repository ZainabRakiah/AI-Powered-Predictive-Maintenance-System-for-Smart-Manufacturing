"""Train on dataset/predictive_maintainence.csv; save model.pkl for app.py."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "dataset" / "predictive_maintainence.csv"
MODEL_PATH = ROOT / "model.pkl"

FEATURE_COLS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COL = "Machine failure"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = [c for c in FEATURE_COLS if c != "Type"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Type"]),
            ("num", "passthrough", num_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    random_state=42,
                    class_weight="balanced",
                    # Single process avoids loky semaphore warnings on Ctrl+C / reload
                    n_jobs=1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print("Accuracy:", round(accuracy, 4))

    joblib.dump(
        {
            "model": model,
            "feature_cols": FEATURE_COLS,
            "target_col": TARGET_COL,
        },
        MODEL_PATH,
    )
    print("Model saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
