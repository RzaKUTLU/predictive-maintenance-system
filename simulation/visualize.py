import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
import os

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
ORIGINAL_PATH  = "data/original_data.csv"
SYNTHETIC_PATH = "data/synthetic_gmm_pca.csv"
OUTPUT_DIR     = "outputs/visuals"

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
orj = pd.read_csv(ORIGINAL_PATH)
syn = pd.read_csv(SYNTHETIC_PATH)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use only numeric columns, exclude metadata and target
EXCLUDE_COLS = ["Machine_ID", "Machine_Type", "Failure_Within_7_Days"]
feature_cols = [
    col for col in orj.select_dtypes(include=[np.number]).columns
    if col not in EXCLUDE_COLS
]

# ──────────────────────────────────────────────
# PCA — Reduce to 2D for visualization
# ──────────────────────────────────────────────
pca     = PCA(n_components=2)
pca_orj = pca.fit_transform(orj[feature_cols])
pca_syn = pca.transform(syn[feature_cols])

# ──────────────────────────────────────────────
# VISUAL 1 — Original vs Synthetic (PCA 2D Scatter)
# ──────────────────────────────────────────────
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_orj[:, 0], y=pca_orj[:, 1], label="Original Data",  alpha=0.5, s=10)
sns.scatterplot(x=pca_syn[:, 0], y=pca_syn[:, 1], label="Synthetic Data", alpha=0.5, s=10)
plt.title("Original vs Synthetic Data — PCA 2D Distribution")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/visual_original_vs_synthetic.png", dpi=200)
plt.close()
print("✅ visual_original_vs_synthetic.png saved")

# ──────────────────────────────────────────────
# VISUAL 2 — Synthetic Data by Class Label
# ──────────────────────────────────────────────
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=pca_syn[:, 0], y=pca_syn[:, 1],
    hue=syn["Failure_Within_7_Days"],
    palette="coolwarm", alpha=0.5, s=10
)
plt.title("Synthetic Data — Class Distribution (Failure_Within_7_Days)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(title="Failure_Within_7_Days")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/visual_synthetic_classes.png", dpi=200)
plt.close()
print("✅ visual_synthetic_classes.png saved")

# ──────────────────────────────────────────────
# VISUAL 3 — Single Feature Histogram
# ──────────────────────────────────────────────
col = feature_cols[0]
plt.figure(figsize=(10, 6))
sns.histplot(orj[col], color="blue", label="Original",  kde=True, stat="density", alpha=0.5)
sns.histplot(syn[col], color="red",  label="Synthetic", kde=True, stat="density", alpha=0.5)
plt.title(f"Original vs Synthetic — {col} Distribution")
plt.xlabel(col)
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/visual_histogram_single.png", dpi=200)
plt.close()
print("✅ visual_histogram_single.png saved")

# ──────────────────────────────────────────────
# VISUAL 4 — Multi-Feature Histogram (first 5 features)
# ──────────────────────────────────────────────
plt.figure(figsize=(15, 8))
for i, col in enumerate(feature_cols[:5]):
    plt.subplot(2, 3, i + 1)
    sns.histplot(orj[col], color="blue", label="Original",  kde=True, stat="density", alpha=0.5)
    sns.histplot(syn[col], color="red",  label="Synthetic", kde=True, stat="density", alpha=0.5)
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel("Density")
    if i == 0:
        plt.legend()
plt.suptitle("Original vs Synthetic — Multi-Feature Histograms", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/visual_histogram_multi.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ visual_histogram_multi.png saved")

# ──────────────────────────────────────────────
# VISUAL 5 — Boxplot Comparison (first 5 features)
# ──────────────────────────────────────────────
plt.figure(figsize=(15, 8))
for i, col in enumerate(feature_cols[:5]):
    plt.subplot(2, 3, i + 1)
    temp_df = pd.DataFrame({
        "Value":     np.concatenate([orj[col], syn[col]]),
        "Data Type": ["Original"] * len(orj) + ["Synthetic"] * len(syn),
    })
    sns.boxplot(x="Data Type", y="Value", data=temp_df, palette=["blue", "red"])
    plt.title(col)
    plt.xlabel("")
    plt.ylabel(col)
plt.suptitle("Original vs Synthetic — Boxplot Comparison", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/visual_boxplot.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ visual_boxplot.png saved")

# ──────────────────────────────────────────────
# VISUAL 6 — Correlation Matrix Comparison
# ──────────────────────────────────────────────
corr_orj = orj[feature_cols].corr()
corr_syn = syn[feature_cols].corr()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(corr_orj, ax=axes[0], cmap="Blues", vmin=-1, vmax=1)
axes[0].set_title("Original Data — Correlation Matrix")
sns.heatmap(corr_syn, ax=axes[1], cmap="Reds",  vmin=-1, vmax=1)
axes[1].set_title("Synthetic Data — Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/visual_correlation.png", dpi=200)
plt.close()
print("✅ visual_correlation.png saved")

# ──────────────────────────────────────────────
# STATISTICS TABLE — Descriptive Comparison
# ──────────────────────────────────────────────
stats_orj = orj[feature_cols].describe().T
stats_syn = syn[feature_cols].describe().T

stats_orj["skew"]     = orj[feature_cols].skew()
stats_syn["skew"]     = syn[feature_cols].skew()
stats_orj["kurtosis"] = orj[feature_cols].kurtosis()
stats_syn["kurtosis"] = syn[feature_cols].kurtosis()

summary = pd.concat(
    [stats_orj.add_prefix("Orig_"), stats_syn.add_prefix("Synt_")],
    axis=1
)
for stat in ["mean", "std", "min", "max", "skew", "kurtosis"]:
    summary[f"diff_{stat}"] = summary[f"Orig_{stat}"] - summary[f"Synt_{stat}"]

summary.to_csv(f"{OUTPUT_DIR}/descriptive_stats_comparison.csv")
print("✅ descriptive_stats_comparison.csv saved")
print("\nDescriptive statistics (first 5 features):")
print(summary.head())

# ──────────────────────────────────────────────
# VISUAL 7 — GMM Components (Ellipses on PCA 2D)
# ──────────────────────────────────────────────
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(pca_syn)

plt.figure(figsize=(10, 6))
plt.scatter(pca_syn[:, 0], pca_syn[:, 1], s=10, alpha=0.4, label="Synthetic Data")

colors = ["red", "green", "blue"]
for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
    v, w  = np.linalg.eigh(covar)
    v     = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    u     = w[0] / np.linalg.norm(w[0])
    angle = np.degrees(np.arctan2(u[1], u[0]))
    ell   = Ellipse(
        xy=mean, width=v[0], height=v[1],
        angle=180.0 + angle,
        color=colors[i % len(colors)],
        alpha=0.3, lw=2,
        label=f"GMM Component {i + 1}"
    )
    plt.gca().add_patch(ell)

plt.title("GMM Components on Synthetic Data (PCA 2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/visual_gmm_ellipses.png", dpi=200)
plt.close()
print("✅ visual_gmm_ellipses.png saved")

print(f"\n🎉 All outputs saved to '{OUTPUT_DIR}/'")
