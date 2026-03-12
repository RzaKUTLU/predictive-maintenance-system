import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_PATH   = "data/original_data.csv"
OUTPUT_FILE = "data/synthetic_statistical_50k.csv"
N_RECORDS   = 50000
TRUE_RATIO  = 0.40   # 40% failure cases
FALSE_RATIO = 1 - TRUE_RATIO

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
original_data = pd.read_csv(DATA_PATH)

# Split data by target label
true_data  = original_data[original_data["Failure_Within_7_Days"] == True]
false_data = original_data[original_data["Failure_Within_7_Days"] == False]

n_true  = int(N_RECORDS * TRUE_RATIO)
n_false = N_RECORDS - n_true

print(f"Generating {n_true} failure (True) and {n_false} normal (False) records...")

# ──────────────────────────────────────────────
# COMPUTE SENSOR STATISTICS
# ──────────────────────────────────────────────
# Calculate mean and std for each sensor column
# separately for True (failure) and False (normal) cases
EXCLUDE_COLS = ["Machine_ID", "Failure_Within_7_Days", "Machine_Type", "Installation_Year"]

sensor_stats = {}
for col in original_data.columns:
    if col not in EXCLUDE_COLS:
        sensor_stats[col] = {
            "true_mean":  true_data[col].mean(),
            "true_std":   true_data[col].std(),
            "false_mean": false_data[col].mean(),
            "false_std":  false_data[col].std(),
        }

# ──────────────────────────────────────────────
# GENERATE MACHINE METADATA
# ──────────────────────────────────────────────
# Generate unique machine IDs
machine_ids = [f"MC_{str(i).zfill(6)}" for i in range(100001)]

# Preserve original machine type distribution
machine_type_dist  = original_data["Machine_Type"].value_counts(normalize=True)
machine_types      = machine_type_dist.index.tolist()
machine_type_probs = machine_type_dist.values.tolist()

# Preserve original installation year distribution
installation_years      = original_data["Installation_Year"].unique()
installation_year_probs = (
    original_data["Installation_Year"]
    .value_counts(normalize=True)
    .reindex(installation_years)
    .values
)

# ──────────────────────────────────────────────
# GENERATE SYNTHETIC RECORDS
# ──────────────────────────────────────────────
def build_base_records(n, label):
    """Create metadata columns for n records with given failure label."""
    return {
        "Machine_ID":            random.choices(machine_ids, k=n),
        "Machine_Type":          np.random.choice(machine_types, size=n, p=machine_type_probs),
        "Installation_Year":     np.random.choice(installation_years, size=n, p=installation_year_probs),
        "Failure_Within_7_Days": [label] * n,
    }

true_records  = build_base_records(n_true,  True)
false_records = build_base_records(n_false, False)

# Generate sensor values using Gaussian distribution
# fitted to the corresponding failure/normal statistics
for col, stats in sensor_stats.items():
    true_records[col]  = np.random.normal(stats["true_mean"],  stats["true_std"],  n_true).round(2)
    false_records[col] = np.random.normal(stats["false_mean"], stats["false_std"], n_false).round(2)

# ──────────────────────────────────────────────
# COMBINE & FORMAT
# ──────────────────────────────────────────────
df = pd.concat(
    [pd.DataFrame(true_records), pd.DataFrame(false_records)],
    ignore_index=True
)

# Sort by Machine_ID and match original column order
df = df.sort_values("Machine_ID")
df = df[original_data.columns]
df["Installation_Year"] = df["Installation_Year"].astype(int)

# ──────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

# ──────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────
print(f"\n✅ {N_RECORDS} records saved to '{OUTPUT_FILE}'")
print(f"\nClass Distribution:")
print(f"  Failure (True) : {df['Failure_Within_7_Days'].mean():.2%}")
print(f"  Normal  (False): {(1 - df['Failure_Within_7_Days'].mean()):.2%}")

print(f"\nMachine Type Distribution:")
for mtype, prob in df["Machine_Type"].value_counts(normalize=True).items():
    print(f"  {mtype}: {prob:.2%}")

print(f"\nSensor Statistics (True vs False):")
for col, stats in sensor_stats.items():
    print(f"\n  {col}:")
    print(f"    Failure  → mean: {stats['true_mean']:.3f}, std: {stats['true_std']:.3f}")
    print(f"    Normal   → mean: {stats['false_mean']:.3f}, std: {stats['false_std']:.3f}")
