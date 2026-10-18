
from pathlib import Path
import sys

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_FILE = (
    PROJECT_ROOT / "models" / "crowd_risk_model_day20.joblib"
)
LIVE_FILE = (
    PROJECT_ROOT / "data" / "processed" / "live_crowd_data.csv"
)
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "day21_live_predictions.csv"
)

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
    print("=" * 65)
    print("CrowdSense AI - Day 21")
    print("Live Data Risk Prediction Test")
    print("=" * 65)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}\n"
            "Complete Day 20 training successfully first."
        )

    if not LIVE_FILE.exists():
        raise FileNotFoundError(
            f"Live data not found: {LIVE_FILE}\n"
            "Run the Day 16 camera logger first."
        )

    # Load model and recorded camera observations.
    model = joblib.load(MODEL_FILE)
    live = pd.read_csv(LIVE_FILE)

    required = [
        "timestamp",
        "total_people",
        "zone_a",
        "zone_b",
        "zone_c",
        "zone_d",
        "avg_movement",
    ]

    missing = [col for col in required if col not in live.columns]
    if missing:
        raise ValueError(f"Live data is missing columns: {missing}")

    live["timestamp"] = pd.to_datetime(
        live["timestamp"], errors="coerce"
    )

    for col in required[1:]:
        live[col] = pd.to_numeric(live[col], errors="coerce")

    live = live.dropna(subset=required).copy()
    live = live.sort_values("timestamp")

    if live.empty:
        raise ValueError("No valid camera observations found.")

    # Use the same five-second aggregation as the live dataset builder.
    indexed = live.set_index("timestamp")

    features = indexed.resample("5s").agg({
        "total_people": ["mean", "max", "min"],
        "zone_a": "mean",
        "zone_b": "mean",
        "zone_c": "mean",
        "zone_d": "mean",
        "avg_movement": "mean",
    })

    features.columns = [
        "total_people_mean",
        "total_people_max",
        "total_people_min",
        "zone_a_mean",
        "zone_b_mean",
        "zone_c_mean",
        "zone_d_mean",
        "avg_movement_mean",
    ]

    features["observations"] = (
        indexed["total_people"].resample("5s").count()
    )

    features = features[
        features["observations"] > 0
    ].copy()

    features["people_change"] = (
        features["total_people_mean"].diff()
    )

    features["rolling_people_mean"] = (
        features["total_people_mean"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    gap_seconds = (
        features.index.to_series()
        .diff()
        .dt.total_seconds()
    )

    adjacent = gap_seconds.eq(5)
    features["people_change_per_minute"] = float("nan")

    features.loc[adjacent, "people_change_per_minute"] = (
        features["total_people_mean"].diff().loc[adjacent] * 12
    )

    # Do not invent zero values for unavailable temporal features.
    # The model's pipeline includes a median imputer.
    X = features[FEATURES].copy()

    predictions = model.predict(X)
    results = features.reset_index()

    results["predicted_risk"] = predictions

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = model.classes_

        for index, label in enumerate(classes):
            results[f"probability_{label.lower()}"] = (
                probabilities[:, index]
            )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"\nCamera observations: {len(live)}")
    print(f"Five-second windows: {len(results)}")

    print("\nPredicted risk distribution:")
    print(results["predicted_risk"].value_counts().to_string())

    print("\nSample predictions:")
    print(
        results[
            ["timestamp", "total_people_mean", "predicted_risk"]
        ].tail(10).to_string(index=False)
    )

    print(f"\nSaved predictions to:\n{OUTPUT_FILE}")
    print(
        "\nNote: These are model predictions on recorded camera data, "
        "not verified emergency-risk assessments."
    )


if __name__ == "__main__":
    main()
