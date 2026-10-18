
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# 1. PATHS
# ==================================================

# Project root is the parent of the scripts folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "crowd_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "crowd_features_5sec.csv"
)

PLOT_DIR = PROJECT_ROOT / "outputs" / "eda"

WINDOW = "5s"


# ==================================================
# 2. LOAD DATA
# ==================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input CSV not found: {INPUT_FILE}\n"
        "Run the real-time pipeline first."
    )

df = pd.read_csv(INPUT_FILE)

if df.empty:
    raise ValueError(
        "The input CSV is empty. Collect camera data first."
    )

print("\nOriginal dataset shape:", df.shape)
print("\nOriginal columns:")
print(df.columns.tolist())


# ==================================================
# 3. VALIDATE REQUIRED COLUMNS
# ==================================================

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
    "net_change",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in input CSV: {missing_columns}"
    )

df = df[required_columns].copy()


# ==================================================
# 4. CLEAN DATA TYPES
# ==================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

numeric_columns = [
    "total_people",
    "zone_a",
    "zone_b",
    "zone_c",
    "zone_d",
    "avg_movement",
    "entries",
    "exits",
    "net_change",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# Rows without usable timestamps cannot be time-aggregated.
df = df.dropna(subset=["timestamp"])

# Remove exact duplicate observations.
df = df.drop_duplicates()

# Sort chronologically.
df = df.sort_values("timestamp").reset_index(drop=True)

if df.empty:
    raise ValueError(
        "No valid timestamped rows remain after cleaning."
    )


# ==================================================
# 5. CHECK BASIC DATA QUALITY
# ==================================================

count_columns = [
    "total_people",
    "zone_a",
    "zone_b",
    "zone_c",
    "zone_d",
    "entries",
    "exits",
]

# Counts should be non-negative integers.
for column in count_columns:
    invalid = (
        df[column].notna()
        & (
            (df[column] < 0)
            | (df[column] % 1 != 0)
        )
    )

    if invalid.any():
        print(
            f"Warning: {invalid.sum()} invalid "
            f"values in {column}."
        )

# Report missing values without silently inventing data.
print("\nMissing values:")
print(df.isna().sum())

# Check consistency between zone totals and overall count.
zone_total = df[
    ["zone_a", "zone_b", "zone_c", "zone_d"]
].sum(axis=1, min_count=4)

mismatch_count = (
    zone_total.notna()
    & df["total_people"].notna()
    & (zone_total != df["total_people"])
).sum()

print("\nZone/count mismatches:", int(mismatch_count))


# ==================================================
# 6. HANDLE MISSING NUMERIC VALUES
# ==================================================

# Do not replace unknown occupancy with zero.
# Drop rows with missing essential count information.
essential_columns = [
    "total_people",
    "zone_a",
    "zone_b",
    "zone_c",
    "zone_d",
]

df = df.dropna(subset=essential_columns)

# Movement can be unavailable for an observation.
# Keep it missing here; aggregation will ignore missing
# movement values rather than treating them as zero.


# ==================================================
# 7. PREPARE THE TIME INDEX
# ==================================================

df = df.set_index("timestamp").sort_index()

# This script assumes one continuous recording session.
# Split recordings into separate sessions before using
# rolling features if there are large time gaps.


# ==================================================
# 8. AGGREGATE INTO FIVE-SECOND WINDOWS
# ==================================================

aggregated = df.resample(
    WINDOW,
    origin="start"
).agg(
    total_people_mean=("total_people", "mean"),
    total_people_max=("total_people", "max"),
    total_people_min=("total_people", "min"),

    zone_a_mean=("zone_a", "mean"),
    zone_b_mean=("zone_b", "mean"),
    zone_c_mean=("zone_c", "mean"),
    zone_d_mean=("zone_d", "mean"),

    avg_movement_mean=("avg_movement", "mean"),

    entries_last=("entries", "last"),
    exits_last=("exits", "last"),
    net_change_last=("net_change", "last"),

    observations=("total_people", "count"),
)

# Empty intervals should not become fake zero-crowd data.
aggregated = aggregated[
    aggregated["observations"] > 0
].copy()

if aggregated.empty:
    raise ValueError(
        "No non-empty time windows were produced."
    )


# ==================================================
# 9. ROUND OCCUPANCY VALUES FOR READABILITY
# ==================================================

mean_columns = [
    "total_people_mean",
    "zone_a_mean",
    "zone_b_mean",
    "zone_c_mean",
    "zone_d_mean",
    "avg_movement_mean",
]

aggregated[mean_columns] = aggregated[
    mean_columns
].round(2)

aggregated["total_people_max"] = (
    aggregated["total_people_max"].astype(int)
)

aggregated["total_people_min"] = (
    aggregated["total_people_min"].astype(int)
)


# ==================================================
# 10. CREATE TEMPORAL FEATURES
# ==================================================

# Change between consecutive non-empty windows.
aggregated["people_change"] = (
    aggregated["total_people_mean"].diff()
)
# Smooth crowd trend across three available windows.
aggregated["rolling_people_mean"] = (
    aggregated["total_people_mean"]
    .rolling(window=3, min_periods=1)
    .mean()
)

# Occupancy change per minute, assuming adjacent
# observations are exactly five seconds apart.
# Do not calculate this across large timestamp gaps.
time_gap_seconds = (
    aggregated.index.to_series().diff().dt.total_seconds()
)

adjacent_windows = time_gap_seconds.eq(5)

aggregated["people_change_per_minute"] = np.nan

aggregated.loc[
    adjacent_windows,
    "people_change_per_minute"
] = (
    aggregated["total_people_mean"].diff()
    .loc[adjacent_windows]
    * 12
)

# Mark windows separated by a gap in recorded data.
aggregated["gap_before_window"] = (
    time_gap_seconds.ne(5)
)


# ==================================================
# 11. SAVE THE DATASET
# ==================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

aggregated.to_csv(
    OUTPUT_FILE,
    index_label="timestamp"
)

print("\nAggregated dataset shape:", aggregated.shape)
print("\nFirst five aggregated rows:")
print(aggregated.head())

print("\nSaved dataset to:")
print(OUTPUT_FILE)


# ==================================================
# 12. CREATE OUTPUT PLOTS
# ==================================================

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Plot 1: average and maximum crowd count.
plt.figure(figsize=(12, 5))

plt.plot(
    aggregated.index,
    aggregated["total_people_mean"],
    label="Mean crowd count"
)

plt.plot(
    aggregated.index,
    aggregated["total_people_max"],
    label="Maximum crowd count"
)

plt.xlabel("Time")
plt.ylabel("Detected people")
plt.title("Crowd Occupancy by Five-Second Window")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "crowd_count_5sec.png",
    dpi=150
)

plt.close()


# Plot 2: zone occupancy trends.
plt.figure(figsize=(12, 5))

for column in [
    "zone_a_mean",
    "zone_b_mean",
    "zone_c_mean",
    "zone_d_mean",
]:
    plt.plot(
        aggregated.index,
        aggregated[column],
        label=column
    )

plt.xlabel("Time")
plt.ylabel("Average detected people")
plt.title("Zone Occupancy by Five-Second Window")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "zone_occupancy_5sec.png",
    dpi=150
)

plt.close()


# Plot 3: crowd change between adjacent windows.
plt.figure(figsize=(12, 5))

plt.plot(
    aggregated.index,
    aggregated["people_change"]
)

plt.axhline(0, linestyle="--")

plt.xlabel("Time")
plt.ylabel("Change in mean occupancy")
plt.title("Crowd Occupancy Change")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    PLOT_DIR / "crowd_change_5sec.png",
    dpi=150
)

plt.close()

print("\nPlots saved to:")
print(PLOT_DIR)
