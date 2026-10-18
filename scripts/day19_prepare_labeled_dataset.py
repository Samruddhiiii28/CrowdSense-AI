
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = DATA_DIR / "crowd_risk_labeled.csv"
OUTPUT_FILE = DATA_DIR / "crowd_training_labeled.csv"

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

VALID_LABELS = {"LOW", "MEDIUM", "HIGH"}


def main():
    print("=" * 60)
    print("CrowdSense AI - Day 19")
    print("Prepare Labeled Training Dataset")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = FEATURES + ["risk_label", "timestamp"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["risk_label"] = (
        df["risk_label"].astype("string").str.strip().str.upper()
    )

    # Exclude rows without usable features or an approved prototype label.
    valid = (
        df["timestamp"].notna()
        & df[FEATURES].notna().all(axis=1)
        & df["risk_label"].isin(VALID_LABELS)
    )

    labeled = df.loc[
        valid, ["timestamp"] + FEATURES + ["risk_label"]
    ].copy()

    if labeled.empty:
        raise ValueError("No complete, valid labeled rows remain.")

    if labeled["risk_label"].nunique() < 2:
        raise ValueError("At least two risk classes are needed.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(OUTPUT_FILE, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Usable labeled rows: {len(labeled)}")
    print("\nLabel distribution:")
    print(labeled["risk_label"].value_counts().to_string())

    print("\nMissing values:")
    print(labeled[FEATURES].isna().sum().to_string())

    print(f"\nSaved to: {OUTPUT_FILE}")
    print("\nImportant: These are prototype-generated labels,")
    print("not independently verified real-world safety labels.")


if __name__ == "__main__":
    main()
