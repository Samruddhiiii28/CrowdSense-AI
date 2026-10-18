
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = ROOT / "models" / "crowd_risk_model_day20.joblib"
DATA_FILE = ROOT / "data" / "processed" / "day21_live_predictions.csv"

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
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Predictions not found: {DATA_FILE}")

    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(DATA_FILE)

    missing = [feature for feature in FEATURES if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    classifier = model.named_steps["classifier"]

    importance = pd.Series(
        classifier.feature_importances_,
        index=FEATURES,
        name="importance",
    ).sort_values(ascending=False)

    print("=" * 60)
    print("CrowdSense AI - Day 22 Model Diagnostics")
    print("=" * 60)

    print("\nRandom Forest feature importance:")
    print(importance.round(4).to_string())

    print("\nRisk predictions by mean people count:")
    print(
        df.groupby(["total_people_mean", "predicted_risk"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print("\nFeature averages by predicted risk:")
    selected = [
        "total_people_mean",
        "avg_movement_mean",
        "people_change",
        "rolling_people_mean",
        "people_change_per_minute",
    ]
    print(
        df.groupby("predicted_risk")[selected]
        .mean()
        .round(3)
        .to_string()
    )

    print(
        "\nNote: Feature importance describes model behavior, "
        "not proof of real-world safety relevance."
    )


if __name__ == "__main__":
    main()
