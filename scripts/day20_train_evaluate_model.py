
from pathlib import Path
import json
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "crowd_training_labeled.csv"
)
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"

MODEL_FILE = MODEL_DIR / "crowd_risk_model_day20.joblib"
METRICS_FILE = MODEL_DIR / "day20_metrics.json"
REPORT_FILE = MODEL_DIR / "day20_evaluation.txt"
MATRIX_FILE = OUTPUT_DIR / "day20_confusion_matrix.png"

FEATURES = [
    "total_people_mean",
    "total_people_max",
    "total_people_min",
    "zone_a_mean",
    "zone_b_mean",
    "zone_c_mean",
    "zone_d_mean",
    "avg_movement_mean",
    "people_change",
    "rolling_people_mean",
    "people_change_per_minute",
    "observations",
]

LABELS = ["LOW", "MEDIUM", "HIGH"]


def main():
    print("=" * 65)
    print("CrowdSense AI - Day 20")
    print("Train and Evaluate Risk Prediction Model")
    print("=" * 65)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run scripts/day19_prepare_labeled_dataset.py first."
        )

    df = pd.read_csv(DATA_FILE)

    required = FEATURES + ["risk_label"]
    missing_columns = [
        col for col in required if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["risk_label"] = (
        df["risk_label"].astype("string").str.strip().str.upper()
    )

    df = df.dropna(subset=FEATURES + ["risk_label"]).copy()
    df = df[df["risk_label"].isin(LABELS)].copy()

    if df.empty:
        raise ValueError("No usable labeled rows remain.")

    class_counts = df["risk_label"].value_counts()

    print("\nDataset rows:", len(df))
    print("\nClass distribution:")
    print(class_counts.to_string())

    # Ensure each class can appear in both training and test sets.
    if len(class_counts) < 2 or class_counts.min() < 2:
        raise ValueError(
            "At least two classes, with at least two rows per class, "
            "are required for a stratified split."
        )

    X = df[FEATURES]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                min_samples_leaf=2,
            ),
        ),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(
        y_test, predictions
    )

    report = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=LABELS,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    metrics = {
        "dataset_rows": int(len(df)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_distribution": {
            str(k): int(v)
            for k, v in class_counts.items()
        },
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "labels": LABELS,
        "confusion_matrix": matrix.tolist(),
        "features": FEATURES,
        "label_source": (
            "Prototype-generated risk labels; "
            "not independently verified real-world safety labels."
        ),
    }

    METRICS_FILE.write_text(
        json.dumps(metrics, indent=4),
        encoding="utf-8",
    )

    report_text = (
        f"Dataset rows: {len(df)}\n"
        f"Training rows: {len(X_train)}\n"
        f"Test rows: {len(X_test)}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Balanced accuracy: {balanced_accuracy:.4f}\n\n"
        f"Classification report:\n{report}\n"
    )

    REPORT_FILE.write_text(report_text, encoding="utf-8")

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=LABELS,
    )
    display.plot(values_format="d")
    plt.title("CrowdSense AI - Day 20 Confusion Matrix")
    plt.tight_layout()
    plt.savefig(MATRIX_FILE, dpi=150)
    plt.close()

    print("\nEvaluation results")
    print("-" * 40)
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print("\nClassification report:")
    print(report)

    print("\nSaved model:", MODEL_FILE)
    print("Saved metrics:", METRICS_FILE)
    print("Saved report:", REPORT_FILE)
    print("Saved confusion matrix:", MATRIX_FILE)

    print(
        "\nNote: These metrics measure agreement with prototype-generated "
        "labels, not real-world emergency prediction accuracy."
    )


if __name__ == "__main__":
    main()
