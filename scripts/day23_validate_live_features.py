
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "processed" / "day21_live_predictions.csv"

ZONES = [
    "zone_a_mean",
    "zone_b_mean",
    "zone_c_mean",
    "zone_d_mean",
]

FEATURES = [
    "total_people_mean",
    "total_people_max",
    "avg_movement_mean",
    "observations",
    "people_change",
    "rolling_people_mean",
    "people_change_per_minute",
]


def main():
    if not FILE.exists():
        raise FileNotFoundError(f"File not found: {FILE}")

    df = pd.read_csv(FILE, parse_dates=["timestamp"])
    required = FEATURES + ZONES + ["predicted_risk"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("=" * 65)
    print("CrowdSense AI - Day 23 Feature Validation")
    print("=" * 65)

    print("\n1. Feature ranges")
    print(df[FEATURES + ZONES].describe().round(3).to_string())

    print("\n2. Invalid negative measurements")
    nonnegative = [
        "total_people_mean",
        "total_people_max",
        "avg_movement_mean",
        "observations",
        *ZONES,
    ]

    for col in nonnegative:
        count = int((df[col] < 0).sum())
        print(f"{col}: {count} negative rows")

    print("\n3. Zone sum compared with total people")
    df["zone_sum"] = df[ZONES].sum(axis=1)
    print(
        df[["total_people_mean", "zone_sum"]]
        .describe()
        .round(3)
        .to_string()
    )

    print("\n4. Movement and predictions")
    print(
        df.groupby("predicted_risk")[
            ["avg_movement_mean", "observations", "total_people_mean"]
        ]
        .agg(["count", "mean", "min", "max"])
        .round(3)
        .to_string()
    )

    print("\n5. Predictions over time")
    print(
        df[["timestamp", "total_people_mean",
            "avg_movement_mean", "predicted_risk"]]
        .sort_values("timestamp")
        .to_string(index=False)
    )

    print("\nValidation complete. No files were modified.")


if __name__ == "__main__":
    main()
