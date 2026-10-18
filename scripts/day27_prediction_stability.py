
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "day21_live_predictions.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed"
    / "day27_stability_comparison.csv"
)

LABEL_CANDIDATES = [
    "predicted_risk",
    "predicted_risk_label",
    "risk_prediction",
    "risk_label",
    "prediction",
]

WINDOW_SIZE = 3

def majority_filter(labels, window_size=3):
    """Causal majority filter; uses current and prior labels only."""
    result = []

    for i in range(len(labels)):
        start = max(0, i - window_size + 1)
        window = labels.iloc[start:i + 1].dropna()

        if window.empty:
            result.append(None)
            continue

        counts = window.value_counts()
        highest_count = counts.max()
        tied_labels = counts[counts == highest_count].index.tolist()

        # On a tie, preserve the current raw prediction.
        current = labels.iloc[i]

        if pd.notna(current) and current in tied_labels:
            result.append(current)
        else:
            result.append(tied_labels[0])

    return result


def main():
    print("=" * 60)
    print("DAY 27: PREDICTION STABILITY")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        raise ValueError("Prediction file contains no rows.")

    print(f"Rows loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    label_col = next(
        (col for col in LABEL_CANDIDATES if col in df.columns),
        None,
    )

    if label_col is None:
        raise ValueError(
            "Could not identify the predicted risk column. "
            "Use the printed columns to update LABEL_CANDIDATES."
        )

    # Preserve chronological order when a timestamp is available.
    if "timestamp" in df.columns:
        parsed_time = pd.to_datetime(
            df["timestamp"], errors="coerce"
        )

        if parsed_time.notna().any():
            df["_sort_time"] = parsed_time
            df = df.sort_values(
                "_sort_time", kind="stable", na_position="last"
            ).drop(columns="_sort_time").reset_index(drop=True)

    raw = df[label_col].astype("string").str.strip().str.upper()
    df["raw_risk_prediction"] = raw

    df["smoothed_risk_prediction"] = majority_filter(
        raw, WINDOW_SIZE
    )

    comparable = (
        df["raw_risk_prediction"].notna()
        & df["smoothed_risk_prediction"].notna()
    )

    df["prediction_changed_after_filter"] = (
        comparable
        & df["raw_risk_prediction"].ne(
            df["smoothed_risk_prediction"]
        )
    )

    raw_changes = int(
        (raw.ne(raw.shift(1)) & raw.notna() & raw.shift(1).notna()).sum()
    )

    smoothed = df["smoothed_risk_prediction"].astype("string")
    smooth_changes = int(
        (
            smoothed.ne(smoothed.shift(1))
            & smoothed.notna()
            & smoothed.shift(1).notna()
        ).sum()
    )

    changed_count = int(
        df["prediction_changed_after_filter"].sum()
    )

    print(f"\nPrediction column: {label_col}")
    print(f"Raw consecutive transitions: {raw_changes}")
    print(f"Filtered consecutive transitions: {smooth_changes}")
    print(f"Rows changed by filter: {changed_count}")

    print("\nRaw distribution:")
    print(raw.value_counts(dropna=False).to_string())

    print("\nFiltered distribution:")
    print(smoothed.value_counts(dropna=False).to_string())

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nComparison saved to: {OUTPUT_FILE}")
    print("Original prediction file was not modified.")
    print("\nDay 27 analysis complete.")


if __name__ == "__main__":
    main()
