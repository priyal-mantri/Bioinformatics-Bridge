# Experiment 5: HDBSCAN Fine-Scale Density Clustering Sandbox

**Project**: Bioinfo Bridge — Research 2 (NHANES 2017–2018)  
**Stage**: Sandbox Experiment 5 — HDBSCAN Fine-Scale Sensitivity Analysis  
**Branch**: `experiment/clustering-sandbox`  
**Status**: COMPLETED & VALIDATED (100% Clean)

---

## Purpose

This experiment performs a fine-scale parameter sensitivity analysis of **HDBSCAN** (`sklearn.cluster.HDBSCAN`) across all four Z-score standardized candidate matrices (`A_RAW`, `A_LOG`, `B_RAW`, `B_LOG`).

Its objective is to determine whether smaller, persistent density structures exist in the standardized feature space when parameters are tuned to finer local neighborhood scales, and to evaluate cluster stability across parameter settings (`min_cluster_size` and `min_samples`) and RAW vs LOG representations.

---

## Methodological Boundaries & Constraints

1. **Exploratory Sandbox Only**: Isolated under `clustering_sandbox/hdbscan/`.
2. **HDBSCAN Implementation**: `sklearn.cluster.HDBSCAN` (scikit-learn `v1.4.1.post1`).
3. **Distance Metric**: Euclidean ($d_{ij} = \|x_i - x_j\|_2$).
4. **Cluster Selection Method**: `eom` (Excess of Mass).
5. **Clustering Space**: Full standardized feature space ($19\text{D}$ for Cohort A, $24\text{D}$ for Cohort B).
6. **PCA Role**: PCA is used **ONLY** for 2D post-hoc visualization (Decision 013).
7. **No Arbitrary Optimization**: We do NOT select a configuration because it produces a specific number of clusters (e.g. 2 or 3), nor do we interpret cluster counts as biological or Ayurvedic proof.

---

## Parameter Grid (48 Total Fits)

- **Cohort A ($N=4,482$, $p=19$)**:
  - `min_cluster_size` $\in \{10, 15, 25, 50\}$
  - `min_samples` $\in \{5, 10, \text{default (matches min\_cluster\_size)}\}$
  (12 configurations per matrix: `A_RAW`, `A_LOG`)
- **Cohort B ($N=967$, $p=24$)**:
  - `min_cluster_size` $\in \{5, 10, 15, 25\}$
  - `min_samples` $\in \{3, 5, 10\}$
  (12 configurations per matrix: `B_RAW`, `B_LOG`)

---

## Key Empirical Findings

1. **Sensitivity to `min_samples` vs `min_cluster_size`**:
   - At standard or conservative core settings (`min_samples` $\ge 10$ or `default`), HDBSCAN classifies **100% of observations as noise** ($0$ clusters) across almost all matrices.
   - At small `min_samples` ($3$ or $5$), HDBSCAN detects **small dense core sub-clusters**:
     - `A_RAW` ($mcs=10, ms=5$): $2$ clusters, $97.2\%$ noise ($125$ non-noise points, $2.8\%$).
     - `A_LOG` ($mcs=10, ms=5$): $2$ clusters, $99.0\%$ noise ($46$ non-noise points, $1.0\%$).
     - `B_RAW` ($mcs=5, ms=3$): $2$ clusters, $88.5\%$ noise ($111$ non-noise points, $11.5\%$).
     - `B_RAW` ($mcs=5, ms=5$): $3$ clusters, $95.1\%$ noise ($47$ non-noise points, $4.9\%$).
     - `B_LOG` ($mcs=5, ms=3$): $2$ clusters, $48.4\%$ noise ($499$ non-noise points, $51.6\%$).

2. **Stability & Persistence**:
   - Small core settings ($ms=3$ or $ms=5$) produce **highly parameter-sensitive, isolated sub-clusters** rather than broad, robust global partitions.
   - The detected sub-clusters represent tiny, high-density local core points surrounded by a vast continuous background noise distribution.

3. **Synthesis with Prior Experiments**:
   - Partition algorithms (K-Means, GMM, Spectral Clustering) partition the entire population by assigning 100% of participants to clusters.
   - HDBSCAN reveals that the standardized biomarker space of NHANES is a **continuous density field**, where only tiny dense cores survive thresholding under density-based criteria.

---

## Directory Structure

```
clustering_sandbox/hdbscan/
├── README.md
├── run_hdbscan_experiment.py
├── validate_hdbscan_experiment.py
├── cluster_assignments/
│   ├── A_LOG_hdbscan_assignments.csv
│   ├── A_RAW_hdbscan_assignments.csv
│   ├── B_LOG_hdbscan_assignments.csv
│   └── B_RAW_hdbscan_assignments.csv
├── metadata/
│   └── hdbscan_experiment_metadata.json
├── metrics/
│   ├── A_LOG_hdbscan_metrics.csv
│   ├── A_RAW_hdbscan_metrics.csv
│   ├── B_LOG_hdbscan_metrics.csv
│   ├── B_RAW_hdbscan_metrics.csv
│   ├── hdbscan_cross_matrix_summary.csv
│   └── hdbscan_stability_summary.csv
└── plots/
    ├── hdbscan_noise_and_clusters_grid.png
    ├── hdbscan_pca_projections_A_LOG.png
    ├── hdbscan_pca_projections_A_RAW.png
    ├── hdbscan_pca_projections_B_LOG.png
    └── hdbscan_pca_projections_B_RAW.png
```
