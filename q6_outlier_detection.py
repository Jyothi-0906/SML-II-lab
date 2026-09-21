"""
Q6: Detect and treat outliers using the Z-Score method and the IQR method
    -- applied to YOUR dataset (BSDS500 pixel features).

We check the continuous pixel-level features extracted from your images:
gray intensity, gradient magnitude, local texture (std), and Laplacian
response. Natural images legitimately contain a few extreme pixels (very
bright highlights, very sharp corners/textured spots), so this is a
realistic, non-synthetic outlier-detection exercise.

BACKGROUND
-----------
--- Z-SCORE METHOD ---
        z = (x - mean) / std
Rule of thumb: |z| > 3 is flagged as an outlier (assumes roughly normal
data; ~99.7% of values lie within 3 std devs of the mean under a normal
curve, so points beyond that are rare/extreme).

--- IQR (Interquartile Range) METHOD ---
More robust to skewed data (uses medians/quartiles, not mean/std, which
are themselves distorted by outliers):
        Q1 = 25th percentile, Q3 = 75th percentile
        IQR = Q3 - Q1
        Lower bound = Q1 - 1.5*IQR,  Upper bound = Q3 + 1.5*IQR
Any value outside these bounds is an outlier -- exactly the rule used to
draw whiskers on a boxplot.

--- TREATMENT ---
  1. Removal        - drop outlier rows.
  2. Capping/Winsorizing - clip to the boundary instead of deleting.
  3. Transformation  - e.g. log-transform to shrink right-skew effects.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

FEATURES_TO_CHECK = ["gray", "grad_mag", "local_std", "laplacian"]

df = pd.read_csv("/home/claude/work/pixel_dataset.csv")
data = df[FEATURES_TO_CHECK].copy()
print(f"Loaded {len(data)} pixel samples from YOUR BSDS500 images.")
print("Original data summary:\n", data.describe(), "\n")

# ---------------------------------------------------------------------------
# Z-SCORE method
# ---------------------------------------------------------------------------
z_scores = np.abs(stats.zscore(data))
z_outliers = (z_scores > 3).any(axis=1)
print(f"Z-score method flagged {z_outliers.sum()} outlier rows out of {len(data)} "
      f"({z_outliers.mean():.1%})")
print(data[z_outliers].describe(), "\n")

# ---------------------------------------------------------------------------
# IQR method
# ---------------------------------------------------------------------------
def iqr_bounds(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

iqr_outlier_mask = pd.Series(False, index=data.index)
bounds = {}
for col in FEATURES_TO_CHECK:
    lower, upper = iqr_bounds(data[col])
    bounds[col] = (lower, upper)
    col_outliers = (data[col] < lower) | (data[col] > upper)
    iqr_outlier_mask |= col_outliers
    print(f"{col:<12s}: IQR bounds = [{lower:8.2f}, {upper:8.2f}], "
          f"outliers found = {col_outliers.sum()} ({col_outliers.mean():.1%})")

print(f"\nIQR method flagged {iqr_outlier_mask.sum()} outlier rows total "
      f"({iqr_outlier_mask.mean():.1%})\n")

# ---------------------------------------------------------------------------
# TREATMENT: capping (Winsorizing) using IQR bounds
# ---------------------------------------------------------------------------
data_capped = data.copy()
for col in FEATURES_TO_CHECK:
    lower, upper = bounds[col]
    data_capped[col] = data_capped[col].clip(lower, upper)

print("Data summary AFTER capping outliers to IQR bounds:\n", data_capped.describe(), "\n")

# ---------------------------------------------------------------------------
# TREATMENT: removal
# ---------------------------------------------------------------------------
data_removed = data[~iqr_outlier_mask].reset_index(drop=True)
print(f"Data shape after removing IQR outliers: {data_removed.shape} (was {data.shape})\n")

# ---------------------------------------------------------------------------
# Visualize before/after with boxplots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].boxplot([data[c] for c in FEATURES_TO_CHECK], tick_labels=FEATURES_TO_CHECK)
axes[0].set_title("Original pixel features (with outliers)")
axes[0].tick_params(axis="x", rotation=20)

axes[1].boxplot([data_capped[c] for c in FEATURES_TO_CHECK], tick_labels=FEATURES_TO_CHECK)
axes[1].set_title("After capping (Winsorizing) at IQR bounds")
axes[1].tick_params(axis="x", rotation=20)

axes[2].boxplot([data_removed[c] for c in FEATURES_TO_CHECK], tick_labels=FEATURES_TO_CHECK)
axes[2].set_title("After removing IQR outlier rows")
axes[2].tick_params(axis="x", rotation=20)

plt.suptitle("Outlier Detection & Treatment on YOUR BSDS500 pixel features")
plt.tight_layout()
plt.savefig("outputs/q6_outlier_treatment.png", dpi=150)
print("Saved boxplot comparison to outputs/q6_outlier_treatment.png")
