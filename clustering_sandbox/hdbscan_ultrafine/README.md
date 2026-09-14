# HDBSCAN Ultra-Fine Scale Sensitivity Diagnostic (Audited)

**Project**: Bioinfo Bridge — Research 2 (NHANES 2017–2018)  
**Stage**: Pre-HDBSCAN Sandbox — Ultra-Fine Scale Sensitivity Diagnostic  
**Branch**: `experiment/clustering-sandbox`  
**Status**: COMPLETED & VALIDATED (100% Clean) | AUDITED

---

## Purpose

This diagnostic evaluates HDBSCAN (`sklearn.cluster.HDBSCAN`) at ultra-fine parameter scales (`min_cluster_size` $\in \{3, 5, 7, 10\}$, `min_samples` $\in \{2, 3, 5\}$) across all four Z-score standardized candidate matrices (`A_RAW`, `A_LOG`, `B_RAW`, `B_LOG`).

Its objective is strictly to determine whether smaller density thresholds reveal persistent, reproducible local density structures or merely parameter-sensitive, fragmented micro-clusters.

**This directory does NOT contain**:
- Final model choices or biological phenotype claims
- Modifications to the master methodology decision log
- Git commits or branch merges

---

## Input Matrices

| Matrix | Cohort | Scale | N | p | File Path |
|:-------|:-------|:------|--:|--:|:----------|
| `A_RAW_scaled.csv` | Broad | Raw | 4,482 | 19 | `output/scaled_matrices/A_RAW_scaled.csv` |
| `A_LOG_scaled.csv` | Broad | Log1p | 4,482 | 19 | `output/scaled_matrices/A_LOG_scaled.csv` |
| `B_RAW_scaled.csv` | Fasting | Raw | 967 | 24 | `output/scaled_matrices/B_RAW_scaled.csv` |
| `B_LOG_scaled.csv` | Fasting | Log1p | 967 | 24 | `output/scaled_matrices/B_LOG_scaled.csv` |

---

## Parameter Grid (48 Total Fits)

- **`min_cluster_size`**: $\{3, 5, 7, 10\}$
- **`min_samples`**: $\{2, 3, 5\}$
- **Distance Metric**: Euclidean ($d_{ij} = \|x_i - x_j\|_2$)
- **Selection Method**: `eom` (Excess of Mass)

Total configurations tested: **48** (12 per matrix).

---

## Key Audited Empirical Findings

1. **Ultra-Permissive Core Setting ($ms=2$)**:
   - Setting $ms=2$ reduces noise dramatically ($0.1\% \dots 23.6\%$ noise in Cohort A; $7.2\% \dots 81.8\%$ noise in Cohort B), grouping $76.4\% \dots 99.9\%$ of participants into clusters.
   - **However, in Cohort A (`A_RAW` and `A_LOG`), this apparent multi-cluster structure is actually a single mega-cluster containing 90%–99.9% of the cohort, accompanied by 1 or 2 tiny micro-clusters of size 3 to 7.**
     - E.g., `A_RAW` ($mcs=3, ms=2$): Cluster 0 size = 4,475, Cluster 1 size = 3 (Noise = 4).
     - E.g., `A_LOG` ($mcs=5, ms=3$): Cluster 0 size = 4,046, Cluster 1 size = 5 (Noise = 431).

2. **B_LOG Structural Transition at `ms=3`**:
   - At `mcs=3` and `mcs=5` ($ms=3$), `B_LOG` yields 2 clusters with **499 non-noise participants (51.6% of cohort)** (Cluster 0 size = 494, Cluster 1 size = 5).
   - At `mcs=7` and `mcs=10` ($ms=3$), non-noise coverage **collapses from 499 down to 126 participants (13.0% of cohort)** (Cluster 0 size = 43, Cluster 1 size = 83), shedding 373 participants into noise.
   - Therefore, the 499-participant structure does **NOT** persist at $mcs \ge 7$; it undergoes a major structural collapse when minimum cluster size exceeds 5.

3. **High Parameter Sensitivity & Interaction**:
   - Increasing `min_cluster_size` and/or `min_samples` generally reduces non-noise coverage, but the magnitude depends strongly on their interaction (e.g., `A_RAW, mcs=7, ms=2` retains 77.73% coverage, whereas `ms=5` drops coverage to <3%).
   - The analysis did not identify persistent large-scale density-separated populations. Local stable regimes (such as `B_LOG mcs=5, ms=3` with 494+5) demonstrate extreme cluster imbalance and are not selected as representative biological partitions.

---


## Directory Structure

```
clustering_sandbox/hdbscan_ultrafine/
├── README.md
├── run_hdbscan_ultrafine_diagnostic.py
├── validate_hdbscan_ultrafine_diagnostic.py
├── cluster_assignments/
│   ├── A_LOG_hdbscan_ultrafine_assignments.csv
│   ├── A_RAW_hdbscan_ultrafine_assignments.csv
│   ├── B_LOG_hdbscan_ultrafine_assignments.csv
│   └── B_RAW_hdbscan_ultrafine_assignments.csv
├── metadata/
│   └── hdbscan_ultrafine_metadata.json
├── metrics/
│   ├── A_LOG_hdbscan_ultrafine_metrics.csv
│   ├── A_RAW_hdbscan_ultrafine_metrics.csv
│   ├── B_LOG_hdbscan_ultrafine_metrics.csv
│   ├── B_RAW_hdbscan_ultrafine_metrics.csv
│   ├── hdbscan_ultrafine_cross_matrix_summary.csv
│   └── hdbscan_ultrafine_stability_summary.csv
└── plots/
    ├── hdbscan_ultrafine_clusters_vs_params.png
    ├── hdbscan_ultrafine_noise_vs_params.png
    ├── hdbscan_ultrafine_non_noise_count.png
    ├── hdbscan_ultrafine_pca_projections_A_LOG.png
    ├── hdbscan_ultrafine_pca_projections_A_RAW.png
    ├── hdbscan_ultrafine_pca_projections_B_LOG.png
    └── hdbscan_ultrafine_pca_projections_B_RAW.png
```
