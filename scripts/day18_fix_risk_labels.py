from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FILE = (
    ROOT
    / "data"
    / "processed"
    / "crowd_training_dataset.csv"
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
    if not FILE.exists():
        raise FileNotFoundError(FILE)

    df = pd.read_csv(FILE)

    print("=" * 60)
    print("CrowdSense AI - Day 18 Dataset Validation")
    print("=" * 60)

    print(f"\nDataset shape: {df.shape}")

    print("\nRows by source:")
    print(df["data_source"].value_counts(dropna=False))

    print("\nMissing values in model features:")
    print(df[FEATURES].isna().sum())

    print("\nFeature summary:")
    print(df[FEATURES].describe().round(3).to_string())

    print("\nPeople-count checks:")
    count_cols = [
        "total_people_mean",
        "total_people_max",
        "total_people_min",
    ]
    print(df[count_cols].describe().round(3).to_string())

    print("\nRisk-label availability:")
    if "risk_label" in df.columns:
        print(df["risk_label"].value_counts(dropna=False))
    else:
        print("No risk_label column found.")

    print("\nDay 18 validation report completed.")

if __name__ == "__main__":
    main()