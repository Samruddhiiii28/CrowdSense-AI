
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "crowd_features_5sec.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "crowd_risk_labeled.csv"
)

PLOT_DIR = PROJECT_ROOT / "outputs" / "eda"

# Starter thresholds for testing only.
# Calibrate these for your actual camera and monitored area.
MEDIUM_COUNT_THRESHOLD = 8
HIGH_COUNT_THRESHOLD = 15

RAPID_INCREASE_THRESHOLD = 3.0  # people per minute
ZONE_IMBALANCE_THRESHOLD = 0.60
MIN_PEOPLE_FOR_ZONE_IMBALANCE = 5


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run scripts/day9_aggregation.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("The input CSV contains no rows.")

    required_columns = [
        "timestamp",
        "total_people_max",
        "zone_a_mean",
        "zone_b_mean",
        "zone_c_mean",
        "zone_d_mean",
        "people_change_per_minute",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}. "
            "Check your Day 9 output CSV."
        )

    numeric_columns = required_columns[1:]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Remove rows missing essential crowd counts.
    df = df.dropna(subset=["total_people_max"]).copy()

    if df.empty:
        raise ValueError("No valid crowd-count rows remain.")

    # Feature 1: Highest observed count in each time window.
    df["crowd_count"] = df["total_people_max"].clip(lower=0)

    # Feature 2: Highest zone share.
    zone_columns = [
        "zone_a_mean",
        "zone_b_mean",
        "zone_c_mean",
        "zone_d_mean",
    ]

    zone_values = df[zone_columns].clip(lower=0)
    zone_total = zone_values.sum(axis=1)

    df["max_zone_share"] = (
        zone_values.max(axis=1)
        .div(zone_total.where(zone_total > 0))
    )

    # Feature 3: Short-term increase in people.
    df["rapid_increase"] = (
        df["people_change_per_minute"]
        >= RAPID_INCREASE_THRESHOLD
    ).fillna(False)

    # Create a transparent prototype risk score.
    # This is a rule-based label, not a trained ML prediction.
    df["risk_score"] = 0

    df.loc[
        df["crowd_count"] >= MEDIUM_COUNT_THRESHOLD,
        "risk_score"
    ] += 1

    df.loc[
        df["crowd_count"] >= HIGH_COUNT_THRESHOLD,
        "risk_score"
    ] += 1

    df.loc[df["rapid_increase"], "risk_score"] += 1

    zone_imbalance = (
        (df["max_zone_share"] >= ZONE_IMBALANCE_THRESHOLD)
        & (df["crowd_count"] >= MIN_PEOPLE_FOR_ZONE_IMBALANCE)
    )

    df.loc[zone_imbalance, "risk_score"] += 1

    # Assign the final prototype labels.
    df["risk_label"] = "LOW"

    df.loc[
        df["risk_score"] == 1,
        "risk_label"
    ] = "MEDIUM"

    df.loc[
        df["risk_score"] >= 2,
        "risk_label"
    ] = "HIGH"

    # Save labeled data.
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nRisk-labeling completed.")
    print(f"Rows processed: {len(df)}")
    print(f"Output saved to: {OUTPUT_FILE}")

    print("\nRisk label distribution:")
    print(df["risk_label"].value_counts())

    print("\nSample results:")
    print(
        df[
            [
                "timestamp",
                "crowd_count",
                "max_zone_share",
                "rapid_increase",
                "risk_score",
                "risk_label",
            ]
        ].head(10).to_string(index=False)
    )

    # Create a risk distribution graph.
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    label_order = ["LOW", "MEDIUM", "HIGH"]
    counts = (
        df["risk_label"]
        .value_counts()
        .reindex(label_order, fill_value=0)
    )

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("CrowdSense AI - Prototype Risk Label Distribution")
    plt.xlabel("Risk label")
    plt.ylabel("Number of 5-second windows")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plot_file = PLOT_DIR / "risk_label_distribution.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()

    print(f"Risk distribution graph saved to: {plot_file}")


if __name__ == "__main__":
    main()
