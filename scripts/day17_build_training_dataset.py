
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "processed"

ORIGINAL_FILE = DATA_DIR / "crowd_features_5sec.csv"
LIVE_FILE = DATA_DIR / "live_crowd_data.csv"
OUTPUT_FILE = DATA_DIR / "crowd_training_dataset.csv"

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

RAW_LIVE_COLUMNS = [
    "total_people",
    "zone_a",
    "zone_b",
    "zone_c",
    "zone_d",
    "avg_movement",
]


def load_original_data():
    if not ORIGINAL_FILE.exists():
        raise FileNotFoundError(f"Missing file: {ORIGINAL_FILE}")

    original = pd.read_csv(ORIGINAL_FILE)
    original["timestamp"] = pd.to_datetime(
        original["timestamp"], errors="coerce"
    )

    missing = [c for c in FEATURES if c not in original.columns]
    if missing:
        raise ValueError(
            f"Original dataset is missing required features: {missing}"
        )

    original["data_source"] = "ORIGINAL_DATASET"
    return original


def load_live_data():
    if not LIVE_FILE.exists():
        raise FileNotFoundError(f"Missing file: {LIVE_FILE}")

    live = pd.read_csv(LIVE_FILE)

    required = ["timestamp"] + RAW_LIVE_COLUMNS
    missing = [c for c in required if c not in live.columns]
    if missing:
        raise ValueError(
            f"Live camera file is missing required columns: {missing}"
        )

    live["timestamp"] = pd.to_datetime(
        live["timestamp"], errors="coerce"
    )

    for col in RAW_LIVE_COLUMNS:
        live[col] = pd.to_numeric(live[col], errors="coerce")

    live = live.dropna(
        subset=["timestamp"] + RAW_LIVE_COLUMNS
    ).copy()

    live = live.sort_values("timestamp")
    live = live.drop_duplicates(
        subset=["timestamp"], keep="last"
    )

    if live.empty:
        raise ValueError("No usable live-camera observations remain.")

    return live


def build_live_features(live):
    live = live.set_index("timestamp").sort_index()

    # Create non-overlapping five-second windows.
    aggregated = live.resample("5s").agg({
        "total_people": ["mean", "max", "min"],
        "zone_a": "mean",
        "zone_b": "mean",
        "zone_c": "mean",
        "zone_d": "mean",
        "avg_movement": "mean",
    })

    aggregated.columns = [
        "total_people_mean",
        "total_people_max",
        "total_people_min",
        "zone_a_mean",
        "zone_b_mean",
        "zone_c_mean",
        "zone_d_mean",
        "avg_movement_mean",
    ]

    aggregated["observations"] = (
        live["total_people"].resample("5s").count()
    )

    aggregated = aggregated[
        aggregated["observations"] > 0
    ].copy()

    if aggregated.empty:
        raise ValueError("No non-empty five-second windows were created.")

    # Match Day 9's definitions.
    aggregated["people_change"] = (
        aggregated["total_people_mean"].diff()
    )

    aggregated["rolling_people_mean"] = (
        aggregated["total_people_mean"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    time_gap_seconds = (
        aggregated.index.to_series()
        .diff()
        .dt.total_seconds()
    )

    adjacent_windows = time_gap_seconds.eq(5)

    aggregated["people_change_per_minute"] = np.nan
    aggregated.loc[
        adjacent_windows, "people_change_per_minute"
    ] = (
        aggregated["total_people_mean"].diff()
        .loc[adjacent_windows] * 12
    )

    aggregated["gap_before_window"] = (
        time_gap_seconds.ne(5)
    )

    result = aggregated.reset_index()
    result["data_source"] = "REAL_CAMERA"

    return result


def main():
    print("=" * 65)
    print("CrowdSense AI - Day 17")
    print("Five-Second Training Dataset Builder")
    print("=" * 65)

    original = load_original_data()
    live = load_live_data()
    live_features = build_live_features(live)

    print(f"Original dataset rows: {len(original)}")
    print(f"Usable live camera rows: {len(live)}")
    print(f"Live five-second windows: {len(live_features)}")

    # Preserve original features and metadata.
    original_columns = [
        "timestamp"
    ] + FEATURES + ["data_source"]

    if "risk_label" in original.columns:
        original_columns.append("risk_label")

    if "gap_before_window" in original.columns:
        original_columns.append("gap_before_window")

    original_part = original[
        original_columns
    ].copy()

    live_columns = [
        "timestamp"
    ] + FEATURES + ["data_source", "gap_before_window"]

    live_part = live_features[live_columns].copy()

    # Combine by column name, not by position.
    combined = pd.concat(
        [original_part, live_part],
        ignore_index=True,
        sort=False,
    )

    combined = combined.sort_values(
        ["timestamp", "data_source"],
        na_position="last",
    ).reset_index(drop=True)

    # Check derived features before saving.
    missing_features = [
        c for c in FEATURES
        if c not in combined.columns
    ]

    if missing_features:
        raise ValueError(
            f"Combined dataset lacks features: {missing_features}"
        )

    print("\nMissing values by source:")
    print(
        combined.groupby("data_source")[FEATURES]
        .apply(lambda group: group.isna().sum())
        .to_string()
    )

    # Main count and zone features must never be missing.
    required_measurements = [
        "total_people_mean",
        "total_people_max",
        "total_people_min",
        "zone_a_mean",
        "zone_b_mean",
        "zone_c_mean",
        "zone_d_mean",
        "avg_movement_mean",
        "rolling_people_mean",
        "observations",
    ]

    bad_rows = combined[required_measurements].isna().any(axis=1)

    if bad_rows.any():
        raise ValueError(
            f"{int(bad_rows.sum())} rows have missing core features. "
            "Dataset was not saved; inspect the source data."
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    print("\nCombined dataset summary")
    print("-" * 40)
    print(f"Total rows: {len(combined)}")
    print(f"Total columns: {len(combined.columns)}")
    print("\nRows by source:")
    print(combined["data_source"].value_counts().to_string())

    print("\nMissing values in model features:")
    print(combined[FEATURES].isna().sum().to_string())

    if "risk_label" in combined.columns:
        print("\nRisk labels:")
        print(
            combined["risk_label"]
            .value_counts(dropna=False)
            .to_string()
        )
    else:
        print("\nRisk labels: not available in combined dataset.")

    print(f"\nSaved dataset:\n{OUTPUT_FILE}")
    print("\nDay 17 dataset preparation completed.")


if __name__ == "__main__":
    main()
