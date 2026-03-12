import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import os

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
DATA_PATH        = "data/original_data.csv"
OUTPUT_FILE      = "data/synthetic_gmm_pca.csv"
N_COMPONENTS_GMM = 3     # Number of Gaussian mixture components
TRUE_RATIO       = 0.40  # 40% failure / 60% normal

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
original_data = pd.read_csv(DATA_PATH)

# Split by target label
true_data  = original_data[original_data["Failure_Within_7_Days"] == True]
false_data = original_data[original_data["Failure_Within_7_Days"] == False]

n_records = len(original_data) - 1  # Exclude header row
n_true    = int(n_records * TRUE_RATIO)
n_false   = n_records - n_true

print(f"Target: {n_true} failure (True) + {n_false} normal (False) = {n_records} total records")

# ──────────────────────────────────────────────
# FEATURE SELECTION
# ──────────────────────────────────────────────
# Use only numeric columns, exclude the target label
feature_cols = [
    col for col in original_data.select_dtypes(include=[np.number]).columns
    if col != "Failure_Within_7_Days"
]

# ──────────────────────────────────────────────
# FIT PCA + GMM → TRUE (FAILURE) CASES
# ──────────────────────────────────────────────
print("Fitting PCA + GMM for failure cases (True)...")

pca_true      = PCA(n_components=2)
pca_true_data = pca_true.fit_transform(true_data[feature_cols])

gmm_true = GaussianMixture(n_components=N_COMPONENTS_GMM, random_state=42)
gmm_true.fit(pca_true_data)

# Sample from GMM and project back to original feature space
synthetic_true_pca, _ = gmm_true.sample(n_true)
synthetic_true         = pca_true.inverse_transform(synthetic_true_pca)

synthetic_true_df = pd.DataFrame(synthetic_true, columns=feature_cols)
synthetic_true_df["Failure_Within_7_Days"] = True

# ──────────────────────────────────────────────
# FIT PCA + GMM → FALSE (NORMAL) CASES
# ──────────────────────────────────────────────
print("Fitting PCA + GMM for normal cases (False)...")

pca_false      = PCA(n_components=2)
pca_false_data = pca_false.fit_transform(false_data[feature_cols])

gmm_false = GaussianMixture(n_components=N_COMPONENTS_GMM, random_state=42)
gmm_false.fit(pca_false_data)

# Sample from GMM and project back to original feature space
synthetic_false_pca, _ = gmm_false.sample(n_false)
synthetic_false         = pca_false.inverse_transform(synthetic_false_pca)

synthetic_false_df = pd.DataFrame(synthetic_false, columns=feature_cols)
synthetic_false_df["Failure_Within_7_Days"] = False

# ──────────────────────────────────────────────
# COMBINE
# ──────────────────────────────────────────────
synthetic_all = pd.concat(
    [synthetic_true_df, synthetic_false_df],
    ignore_index=True
)

# ──────────────────────────────────────────────
# FORMAT — MATCH ORIGINAL DTYPES
# ──────────────────────────────────────────────
for col in synthetic_all.columns:
    if col not in original_data.columns:
        continue

    orig_dtype = original_data[col].dtype

    if np.issubdtype(orig_dtype, np.integer):
        # Round floats to nearest integer
        synthetic_all[col] = np.round(synthetic_all[col]).astype(int)

    elif np.issubdtype(orig_dtype, np.floating):
        # Keep as float
        synthetic_all[col] = synthetic_all[col].astype(float)

    elif orig_dtype == bool:
        synthetic_all[col] = synthetic_all[col].astype(bool)

    elif orig_dtype == object:
        # For categorical columns (e.g. Machine_ID, Machine_Type),
        # randomly sample from original values if column is empty/uniform
        if synthetic_all[col].isnull().all() or synthetic_all[col].nunique() <= 1:
            synthetic_all[col] = np.random.choice(
                original_data[col].dropna().unique(),
                size=len(synthetic_all)
            )
        else:
            synthetic_all[col] = synthetic_all[col].astype(str)

# ──────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
synthetic_all.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ Synthetic data saved to '{OUTPUT_FILE}'")
print(f"   Total records : {len(synthetic_all)}")
print(f"   Failure (True): {synthetic_all['Failure_Within_7_Days'].mean():.2%}")
print(f"   Normal (False): {(1 - synthetic_all['Failure_Within_7_Days'].mean()):.2%}")
