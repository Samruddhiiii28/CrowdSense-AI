from pathlib import Path
import sys

import pandas as pd

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.risk_predictor import CrowdRiskPredictor

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "crowd_features_5sec.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "risk_predictions.csv"
)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input data not found: {INPUT_FILE}"
        )

    # Load recent crowd-feature windows.
    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError(
            "The crowd-feature CSV is empty. "
            "Run your real-time pipeline to collect data."
        )

    predictor = CrowdRiskPredictor()

    # Predict one risk label for each time window.
    predictions = predictor.predict(df)

    # Keep the timestamp alongside each prediction.
    if "timestamp" in df.columns:
        predictions.insert(
            0, "timestamp", df["timestamp"].values
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )
    predictions.to_csv(OUTPUT_FILE, index=False)

    print("\nCrowdSense AI - Risk Prediction")
    print("=" * 38)
    print(f"Windows analyzed: {len(predictions)}")
    print(f"Predictions saved to: {OUTPUT_FILE}")

    print("\nPredicted risk distribution:")
    print(predictions["risk_label"].value_counts())

    print("\nLatest predictions:")
    print(predictions.tail(10).to_string(index=False))

    print(
        "\nNote: these are experimental predictions of "
        "Day 10 rule-based labels, not validated safety alerts."
    )


if __name__ == "__main__":
    main()
