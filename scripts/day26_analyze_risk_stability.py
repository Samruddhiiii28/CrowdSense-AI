
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data" / "processed"
    / "day21_live_predictions.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed"
    / "day26_risk_stability_report.txt"
)

print("=" * 60)
print("DAY 26: RISK PREDICTION STABILITY")
print("=" * 60)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Prediction file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

if df.empty:
    raise ValueError("The prediction file is empty.")

print(f"\nRows loaded: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Locate the predicted risk-label column.
possible_labels = [
    "predicted_risk",
    "predicted_risk_label",
    "risk_prediction",
    "risk_label",
    "prediction",
]

label_col = next(
    (col for col in possible_labels if col in df.columns),
    None,
)

if label_col is None:
    raise ValueError(
        "Could not identify the prediction column. "
        "Check the printed column names and update "
        "possible_labels to match your CSV."
    )

labels = df[label_col].astype("string").str.strip().str.upper()

valid_labels = labels.dropna()
counts = valid_labels.value_counts()

print(f"\nPrediction column: {label_col}")
print("\n--- Risk Distribution ---")

for label, count in counts.items():
    print(f"{label}: {count}")

# Compare each prediction with the previous row.
previous = labels.shift(1)

valid_pairs = labels.notna() & previous.notna()

changed = labels.ne(previous) & valid_pairs

pair_count = int(valid_pairs.sum())
change_count = int(changed.sum())

change_rate = (
    change_count / pair_count * 100
    if pair_count > 0
    else 0.0
)

print("\n--- Consecutive Prediction Changes ---")
print(f"Comparable consecutive pairs: {pair_count}")
print(f"Changed predictions: {change_count}")
print(f"Change rate: {change_rate:.2f}%")

transitions = pd.DataFrame({
    "previous": previous[changed],
    "current": labels[changed],
})

print("\n--- Observed Transitions ---")

if transitions.empty:
    print("No changes between comparable rows.")
else:
    transition_counts = (
        transitions.groupby(["previous", "current"])
        .size()
        .sort_values(ascending=False)
    )

    for (old, new), count in transition_counts.items():
        print(f"{old} -> {new}: {count}")

# Save the report.
report = [
    "DAY 26: RISK PREDICTION STABILITY REPORT",
    f"Input rows: {len(df)}",
    f"Prediction column: {label_col}",
    "",
    "Risk distribution:",
]

report.extend(
    f"- {label}: {count}"
    for label, count in counts.items()
)

report.extend([
    "",
    f"Comparable consecutive pairs: {pair_count}",
    f"Changed predictions: {change_count}",
    f"Change rate: {change_rate:.2f}%",
    "",
    "Observed transitions:",
])

if transitions.empty:
    report.append("- No transitions observed.")
else:
    for (old, new), count in transition_counts.items():
        report.append(f"- {old} -> {new}: {count}")

OUTPUT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8",
)

print(f"\nReport saved to: {OUTPUT_FILE}")
print("\nDay 26 analysis complete.")
