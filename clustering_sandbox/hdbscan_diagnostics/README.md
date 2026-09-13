# HDBSCAN Parameter-Calibration Diagnostic

**Project**: Bioinfo Bridge — Research 2 (NHANES 2017–2018)  
**Stage**: Pre-Experiment 5 — HDBSCAN Parameter-Calibration Diagnostic  
**Branch**: `experiment/clustering-sandbox`  
**Status**: COMPLETED & VALIDATED (100% Clean)

---

## Purpose

This diagnostic characterizes the density-based clustering behavior of **HDBSCAN** (`sklearn.cluster.HDBSCAN`) across a structured grid of density parameters (`min_cluster_size` and `min_samples`) for all four Z-score standardized candidate matrices (`A_RAW`, `A_LOG`, `B_RAW`, `B_LOG`).

Unlike partition-based algorithms (K-Means, GMM, Spectral Clustering), HDBSCAN does not take $K$ as an input. Its output depends on local density thresholds. This diagnostic establishes empirical baseline performance before designing Experiment 5.

**This directory does NOT contain**:
- Final model selection decisions
- Biological interpretation of noise vs clusters
- Modifications to the master methodology decision log

---

## Input Matrices

| Matrix | Cohort | Scale | N | p | File Path |
|:-------|:-------|:------|--:|--:|:----------|
| `A_RAW_scaled.csv` | Broad | Raw | 4,482 | 19 | `output/scaled_matrices/A_RAW_scaled.csv` |
| `A_LOG_scaled.csv` | Broad | Log1p | 4,482 | 19 | `output/scaled_matrices/A_LOG_scaled.csv` |
| `B_RAW_scaled.csv` | Fasting | Raw | 967 | 24 | `output/scaled_matrices/B_RAW_scaled.csv` |
| `B_LOG_scaled.csv` | Fasting | Log1p | 967 | 24 | `output/scaled_matrices/B_LOG_scaled.csv` |

---

## Parameter Grids Evaluated

- **Cohort A ($N=4,482$, $p=19$)**:
  - `min_cluster_size` $\in \{25, 50, 100, 200\}$ (~0.56% to 4.46% of N)
  - `min_samples` $\in \{\text{default (matches min\_cluster\_size)}, 5, 15\}$
- **Cohort B ($N=967$, $p=24$)**:
  - `min_cluster_size` $\in \{10, 25, 50, 100\}$ (~1.03% to 10.34% of N)
  - `min_samples` $\in \{\text{default (matches min\_cluster\_size)}, 5, 15\}$

Total configurations tested: **48** (12 per matrix).

---

## Key Diagnostic Findings

1. **High-Dimensional Continuous Density Diffusion**: Across almost all parameter configurations, HDBSCAN classifies **100% of observations as noise** (label `-1`, 0 clusters found).
2. **Minor Sub-Cluster Detection in Cohort B**: Only at `min_cluster_size=10, min_samples=5` in Cohort B (`B_RAW` and `B_LOG`) does HDBSCAN detect 2 small dense sub-clusters (48 points = 4.97% of sample in `B_RAW`; 63 points = 6.51% of sample in `B_LOG`), with 93.5%–95.0% classified as noise.
3. **Curse of Dimensionality in Standardized Feature Space**: In 19D (Cohort A) and 24D (Cohort B) Euclidean space, standardized biomarker values form a unimodal, continuously decaying density field without steep, isolated local density peaks. Consequently, HDBSCAN's mutual reachability distance algorithm treats the entire distribution as background variation around a single central mass.

---

## Directory Structure

```
clustering_sandbox/hdbscan_diagnostics/
├── README.md
├── run_hdbscan_calibration_diagnostic.py
├── validate_hdbscan_calibration_diagnostic.py
├── metadata/
│   └── hdbscan_calibration_metadata.json
├── metrics/
│   ├── A_LOG_hdbscan_calibration.csv
│   ├── A_RAW_hdbscan_calibration.csv
│   ├── B_LOG_hdbscan_calibration.csv
│   ├── B_RAW_hdbscan_calibration.csv
│   └── diag1_hdbscan_calibration_summary.csv
└── plots/
    ├── hdbscan_diag1_n_clusters_vs_params.png
    ├── hdbscan_diag2_noise_proportion_vs_params.png
    └── hdbscan_diag4_pca_projections_rep.png
```
