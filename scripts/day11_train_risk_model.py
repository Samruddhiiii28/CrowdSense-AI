from pathlib import Path

import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "crowd_risk_labeled.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"

MODEL_FILE = MODEL_DIR / "crowd_risk_model.joblib"
REPORT_FILE = MODEL_DIR / "day11_evaluation.txt"
METRICS_FILE = MODEL_DIR / "day11_metrics.json"
MATRIX_FILE = OUTPUT_DIR / "day11_confusion_matrix.png"

LABELS = ["LOW", "MEDIUM", "HIGH"]


# Do not include risk_score or risk_label as input features.
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


def main():

    # ---------------------------------------------------------
    # 1. Check input file
    # ---------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT_FILE}\n"
            "Run Day 10 successfully before Day 11."
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("The input dataset is empty.")

    # ---------------------------------------------------------
    # 2. Check required columns
    # ---------------------------------------------------------

    required = ["timestamp", "risk_label"] + FEATURES

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            "Check your Day 9 and Day 10 output files."
        )

    # ---------------------------------------------------------
    # 3. Clean timestamp and labels
    # ---------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["timestamp", "risk_label"]
    ).copy()

    df["risk_label"] = (
        df["risk_label"]
        .astype(str)
        .str.upper()
    )

    df = df[
        df["risk_label"].isin(LABELS)
    ].copy()

    # Sort chronologically for consistent dataset ordering.
    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 4. Convert features to numeric
    # ---------------------------------------------------------

    for col in FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Remove rows where every feature is missing.
    df = df.dropna(
        subset=FEATURES,
        how="all"
    ).copy()

    # ---------------------------------------------------------
    # 5. Basic dataset validation
    # ---------------------------------------------------------

    if len(df) < 10:
        raise ValueError(
            f"Only {len(df)} usable rows found. "
            "Collect more real-time observations before training."
        )

    print("\nDataset size:", len(df))

    print("\nLabel distribution:")
    print(df["risk_label"].value_counts())

    if df["risk_label"].nunique() < 2:
        raise ValueError(
            "Only one risk class exists in the dataset. "
            "Collect observations with varied crowd conditions."
        )

    # ---------------------------------------------------------
    # 6. Separate features and target
    # ---------------------------------------------------------

    X = df[FEATURES]
    y = df["risk_label"]

    # ---------------------------------------------------------
    # 7. Stratified train-test split
    # ---------------------------------------------------------
    #
    # Stratification ensures LOW, MEDIUM and HIGH
    # are represented in both training and testing sets.
    #

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ---------------------------------------------------------
    # 8. Validate train/test classes
    # ---------------------------------------------------------

    if y_train.nunique() < 2:
        raise ValueError(
            "Training set contains fewer than two classes."
        )

    if y_test.nunique() < 2:
        raise ValueError(
            "Test set contains fewer than two classes."
        )

    # ---------------------------------------------------------
    # 9. Build Random Forest pipeline
    # ---------------------------------------------------------

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    # ---------------------------------------------------------
    # 10. Train model
    # ---------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # ---------------------------------------------------------
    # 11. Predictions
    # ---------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # ---------------------------------------------------------
    # 12. Evaluation
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        zero_division=0,
    )

    print("\nStratified test results")
    print("-----------------------")

    print(
        f"Training rows: {len(X_train)}"
    )

    print(
        f"Testing rows:  {len(X_test)}"
    )

    print(
        f"Accuracy: {accuracy:.3f}"
    )

    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy:.3f}"
    )

    print("\nClassification report:")
    print(report)

    # ---------------------------------------------------------
    # 13. Save model
    # ---------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "labels": LABELS,
            "label_source":
                "Day 18 prototype risk labels",
        },
        MODEL_FILE,
    )

    # ---------------------------------------------------------
    # 14. Save evaluation report
    # ---------------------------------------------------------

    REPORT_FILE.write_text(
        "CrowdSense AI - Day 11 Evaluation\n"
        "=================================\n"
        "Target: Day 18 prototype risk labels\n"
        "Split: stratified 80/20\n\n"
        f"Dataset rows: {len(df)}\n"
        f"Training rows: {len(X_train)}\n"
        f"Testing rows: {len(X_test)}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"Balanced accuracy: {balanced_accuracy:.4f}\n\n"
        f"Class distribution:\n"
        f"{df['risk_label'].value_counts()}\n\n"
        f"Test class distribution:\n"
        f"{y_test.value_counts()}\n\n"
        f"Classification report:\n"
        f"{report}",
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # 15. Save metrics JSON
    # ---------------------------------------------------------

    METRICS_FILE.write_text(
        json.dumps(
            {
                "dataset_rows": len(df),
                "training_rows": len(X_train),
                "testing_rows": len(X_test),
                "accuracy": accuracy,
                "balanced_accuracy":
                    balanced_accuracy,
                "evaluation":
                    "stratified_holdout",
                "target":
                    "Day 18 prototype labels",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # 16. Confusion matrix
    # ---------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=LABELS
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=LABELS,
    )

    display.plot()

    plt.title(
        "Day 11 - Crowd Risk Classification"
    )

    plt.tight_layout()

    plt.savefig(
        MATRIX_FILE,
        dpi=150
    )

    plt.close()

    # ---------------------------------------------------------
    # 17. Final output
    # ---------------------------------------------------------

    print("\nFiles saved:")

    print(
        f"Model: {MODEL_FILE}"
    )

    print(
        f"Report: {REPORT_FILE}"
    )

    print(
        f"Metrics: {METRICS_FILE}"
    )

    print(
        f"Confusion matrix: {MATRIX_FILE}"
    )


if __name__ == "__main__":
    main()