
from pathlib import Path

import pandas as pd
import numpy as np

# ----------------------------------------
# Paths
# ----------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "live_crowd_data.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "day25_validation_report.txt"

# ----------------------------------------
# Load data
# ----------------------------------------

print("=" * 60)
print("DAY 25: LIVE DATA VALIDATION")
print("=" * 60)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Live data file not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

if df.empty:
    raise ValueError("The live data CSV contains no rows.")

print(f"\nFile: {DATA_FILE.name}")
print(f"Total observations: {len(df)}")
print(f"Columns: {list(df.columns)}")

required_columns = [
    "timestamp",
    "total_people",
    "zone_a",
    "zone_b",
    "zone_c",
    "zone_d",
    "avg_movement",
    "entries",
    "exits",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# ----------------------------------------
# Convert measurement columns to numeric
# ----------------------------------------

numeric_columns = [
    "total_people",
    "zone_a",
    "zone_b",
    "zone_c",
    "zone_d",
    "avg_movement",
    "entries",
    "exits",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

issues = []

# ----------------------------------------
# Missing values
# ----------------------------------------

print("\n--- Missing Values ---")

for col in required_columns:
    missing = int(df[col].isna().sum())
    print(f"{col}: {missing}")

    if missing:
        issues.append(
            f"{col}: {missing} missing or non-numeric values"
        )

# ----------------------------------------
# Negative values
# ----------------------------------------

print("\n--- Negative Values ---")

for col in numeric_columns:
    negative = int((df[col] < 0).sum())
    print(f"{col}: {negative}")

    if negative:
        issues.append(
            f"{col}: {negative} negative values"
        )

# ----------------------------------------
# Count consistency
# ----------------------------------------

print("\n--- Count Consistency ---")

zone_columns = ["zone_a", "zone_b", "zone_c", "zone_d"]

valid_zone_rows = df[zone_columns].notna().all(axis=1)
valid_total_rows = df["total_people"].notna()

comparable = valid_zone_rows & valid_total_rows

zone_sum = df[zone_columns].sum(axis=1, min_count=4)

zone_mismatch = (
    comparable
    & ~np.isclose(
        zone_sum.fillna(0),
        df["total_people"].fillna(0),
    )
)

mismatch_count = int(zone_mismatch.sum())

print(f"Zone/total mismatches: {mismatch_count}")

if mismatch_count:
    issues.append(
        f"{mismatch_count} rows where zone counts do not sum "
        "to total_people"
    )

# ----------------------------------------
# Count and movement ranges
# ----------------------------------------

print("\n--- Measurement Summary ---")

for col in numeric_columns:
    print(f"\n{col}")
    print(df[col].describe().to_string())

# ----------------------------------------
# Abrupt count changes
# ----------------------------------------

print("\n--- Consecutive Count Changes ---")

people_change = df["total_people"].diff().abs()
large_changes = people_change > 5

large_change_count = int(large_changes.sum())

print(
    "Absolute count changes greater than 5 between "
    f"consecutive CSV rows: {large_change_count}"
)

if large_change_count:
    print(
        "These rows need inspection; they are not "
        "automatically classified as errors."
    )

# ----------------------------------------
# Timestamp checks
# ----------------------------------------

print("\n--- Timestamp Checks ---")

timestamps = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

bad_timestamps = int(timestamps.isna().sum())
duplicate_timestamps = int(timestamps.duplicated().sum())

print(f"Invalid timestamps: {bad_timestamps}")
print(f"Duplicate timestamps: {duplicate_timestamps}")

if bad_timestamps:
    issues.append(f"{bad_timestamps} invalid timestamps")

if duplicate_timestamps:
    issues.append(
        f"{duplicate_timestamps} duplicate timestamps"
    )

# ----------------------------------------
# Report
# ----------------------------------------

report_lines = [
    "DAY 25: LIVE DATA VALIDATION REPORT",
    f"Rows checked: {len(df)}",
    f"Zone/total mismatches: {mismatch_count}",
    f"Large consecutive count changes: {large_change_count}",
    f"Invalid timestamps: {bad_timestamps}",
    f"Duplicate timestamps: {duplicate_timestamps}",
    "",
    "Issues requiring review:",
]

if issues:
    report_lines.extend(f"- {issue}" for issue in issues)
else:
    report_lines.append(
        "No issues detected by the checks performed."
    )

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)

print(f"Report saved to: {OUTPUT_FILE}")

if issues:
    print(f"Items requiring review: {len(issues)}")
else:
    print("No issues detected by these checks.")
