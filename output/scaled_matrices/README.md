# output/scaled_matrices/

This directory contains the Z-score standardized versions of the four candidate feature matrices, produced in Stage 4 of the pipeline. These matrices were used as the primary input for all downstream exploratory and clustering analysis.

## Files

| File | Description |
| :--- | :--- |
| `A_RAW_scaled.csv` | Cohort A, raw scale, Z-score standardized (4,482 × 19) |
| `A_LOG_scaled.csv` | Cohort A, log1p-transformed, Z-score standardized (4,482 × 19) |
| `B_RAW_scaled.csv` | Cohort B, raw scale, Z-score standardized (967 × 24) |
| `B_LOG_scaled.csv` | Cohort B, log1p-transformed, Z-score standardized (967 × 24) |
| `scaled_matrices_metadata.json` | Per-matrix scaling parameters (means, standard deviations, feature lists) and provenance |

## Standardization

Each matrix was standardized independently using Z-score scaling:

- Each feature is centered to **mean ≈ 0** and scaled to **standard deviation ≈ 1**
- Scaling parameters (mean and standard deviation per feature) are stored in `scaled_matrices_metadata.json`

Scaling was applied **independently per matrix**. The mean and standard deviation of a feature in `A_RAW_scaled` were computed from Cohort A only, and the same feature in `B_RAW_scaled` used statistics computed from Cohort B only. No parameters were shared across cohorts or across RAW and LOG representations.

This independence ensures there is no data leakage between cohort or transformation comparisons.

## Usage

These standardized matrices were used directly as input to the clustering experiments in Stage 6 (K-Means, GMM, Spectral, Hierarchical, HDBSCAN) and the cross-model benchmark in Stage 7. Clustering was performed in the full standardized feature space — not in a truncated PCA space. See `output/pca/` for the PCA exploratory audit.

The scaling parameters in `scaled_matrices_metadata.json` can be used to transform new observations to the same scale for out-of-sample comparison, if needed.
