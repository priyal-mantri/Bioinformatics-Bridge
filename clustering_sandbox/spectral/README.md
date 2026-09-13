# Experiment 4: Spectral Clustering Exploratory Sandbox

**Project**: Bioinfo Bridge — Research 2 (NHANES 2017–2018)  
**Stage**: Sandbox Experiment 4 — Spectral Clustering  
**Branch**: `experiment/clustering-sandbox`  
**Status**: COMPLETED & VALIDATED (100% Clean)

---

## Purpose

This experiment evaluates the phenotypic structure found by probabilistic/graph-based **Spectral Clustering** operating on the full standardized feature space across all four candidate matrices (`A_RAW`, `A_LOG`, `B_RAW`, `B_LOG`) for cluster counts $K = 2, 3, 4, 5, 6, 7$.

Its objective is to determine what non-linear manifold/graph partition structure Spectral Clustering identifies, and to compare its behavior against the centroid-based K-Means (Experiments 1 & 2) and probabilistic GMM (Experiment 3) results.

---

## Methodological Boundaries & Rules

1. **Exploratory Sandbox Only**: Isolated under `clustering_sandbox/spectral/`.
2. **Fixed Graph Construction Parameter**: $k\_neighbors = 10$ across all matrices (informed by prior affinity calibration diagnostics).
3. **Fixed Symmetrization Rule**: $W = (A + A^T) / 2$ where $A$ is the directed binary kNN adjacency matrix.
4. **Full Feature Space**: All clustering is executed in the full standardized feature space (19D for Cohort A, 24D for Cohort B).
5. **PCA Role**: PCA is used **ONLY** for 2D post-hoc visualization (Decision 013).
6. **No Optimal K Selection**: We do NOT select a single winning K, declare a model winner, interpret biological phenotypes, or relate results to Ayurvedic Tridosha/Prakriti at this stage.

---

## Input Matrices

| Matrix | Cohort | Scale | N | p | File Path |
|:-------|:-------|:------|--:|--:|:----------|
| `A_RAW_scaled.csv` | Broad | Raw | 4,482 | 19 | `output/scaled_matrices/A_RAW_scaled.csv` |
| `A_LOG_scaled.csv` | Broad | Log1p | 4,482 | 19 | `output/scaled_matrices/A_LOG_scaled.csv` |
| `B_RAW_scaled.csv` | Fasting | Raw | 967 | 24 | `output/scaled_matrices/B_RAW_scaled.csv` |
| `B_LOG_scaled.csv` | Fasting | Log1p | 967 | 24 | `output/scaled_matrices/B_LOG_scaled.csv` |

---

## Implementation & Efficiency Design

- **Single Eigendecomposition per Matrix**: The kNN affinity graph $W$ and normalized symmetric graph Laplacian $L_{\text{sym}} = I - D^{-1/2} W D^{-1/2}$ are constructed once per matrix. The top 8 eigenvectors ($V_1 \dots V_8$) are computed once via `scipy.sparse.linalg.eigsh`.
- **Spectral Embedding Partitioning**: For each $K \in \{2..7\}$, the Ng-Jordan-Weiss (NJW) L2 row-normalized spectral embedding $Y_K \in \mathbb{R}^{N \times K}$ is formed and partitioned using `KMeans(n_clusters=K, random_state=42, n_init=25)`.
- **SEQN Integrity**: Participant SEQN identifiers are preserved 1-to-1 in all cluster assignment files.

---

## Directory Structure

```
clustering_sandbox/spectral/
├── README.md
├── run_spectral_experiment.py
├── validate_spectral_experiment.py
├── cluster_assignments/
│   ├── A_LOG_spectral_assignments.csv
│   ├── A_RAW_spectral_assignments.csv
│   ├── B_LOG_spectral_assignments.csv
│   └── B_RAW_spectral_assignments.csv
├── metadata/
│   └── spectral_experiment_metadata.json
├── metrics/
│   ├── A_LOG_spectral_metrics.csv
│   ├── A_RAW_spectral_metrics.csv
│   ├── B_LOG_spectral_metrics.csv
│   ├── B_RAW_spectral_metrics.csv
│   └── spectral_cross_matrix_summary.csv
└── plots/
    ├── spectral_metrics_vs_k.png
    ├── spectral_pca_projections_A_LOG.png
    ├── spectral_pca_projections_A_RAW.png
    ├── spectral_pca_projections_B_LOG.png
    └── spectral_pca_projections_B_RAW.png
```

---

## Summary of Results

### Cross-Matrix Metrics Summary (K=2..7)

| Matrix | K | Silhouette | CH Score | DB Index | Min Size | Min Prop | Max Size | Max Prop | Max/Min Ratio | Fiedler λ₂ | Eigengap Δ(K-1→K) | Eigengap Δ(K→K+1) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A_RAW** | 2 | 0.0935 | 419.8 | 3.076 | 1998 | 0.4458 | 2484 | 0.5542 | 1.2432 | 0.07798 | 0.07798 | 0.01281 |
| **A_RAW** | 3 | 0.0212 | 272.8 | 4.265 | 1453 | 0.3242 | 1541 | 0.3438 | 1.0606 | 0.07798 | 0.01281 | 0.00088 |
| **A_RAW** | 4 | 0.0299 | 260.4 | 3.756 | 613 | 0.1368 | 1406 | 0.3137 | 2.2936 | 0.07798 | 0.00088 | 0.00838 |
| **A_RAW** | 5 | 0.0379 | 251.7 | 3.422 | 340 | 0.0759 | 1350 | 0.3012 | 3.9706 | 0.07798 | 0.00838 | 0.02156 |
| **A_RAW** | 6 | 0.0425 | 243.4 | 2.864 | 355 | 0.0792 | 1184 | 0.2642 | 3.3352 | 0.07798 | 0.02156 | 0.01777 |
| **A_RAW** | 7 | 0.0406 | 224.2 | 2.851 | 324 | 0.0723 | 906 | 0.2021 | 2.7963 | 0.07798 | 0.01777 | 0.01318 |
| **A_LOG** | 2 | 0.0874 | 465.7 | 2.935 | 1922 | 0.4288 | 2560 | 0.5712 | 1.3319 | 0.08770 | 0.08770 | 0.02471 |
| **A_LOG** | 3 | 0.0742 | 368.0 | 2.876 | 1012 | 0.2258 | 1954 | 0.4360 | 1.9308 | 0.08770 | 0.02471 | 0.01053 |
| **A_LOG** | 4 | 0.0786 | 326.0 | 2.687 | 594 | 0.1325 | 1525 | 0.3402 | 2.5673 | 0.08770 | 0.01053 | 0.03456 |
| **A_LOG** | 5 | 0.0570 | 282.6 | 2.740 | 549 | 0.1225 | 1399 | 0.3121 | 2.5483 | 0.08770 | 0.03456 | 0.00788 |
| **A_LOG** | 6 | 0.0446 | 244.2 | 3.030 | 480 | 0.1071 | 953 | 0.2126 | 1.9854 | 0.08770 | 0.00788 | 0.00944 |
| **A_LOG** | 7 | 0.0462 | 227.3 | 2.811 | 394 | 0.0879 | 807 | 0.1801 | 2.0482 | 0.08770 | 0.00944 | 0.00970 |
| **B_RAW** | 2 | 0.1040 | 106.4 | 2.865 | 473 | 0.4891 | 494 | 0.5109 | 1.0444 | 0.09659 | 0.09659 | 0.00953 |
| **B_RAW** | 3 | 0.0894 | 100.8 | 2.471 | 285 | 0.2947 | 393 | 0.4064 | 1.3789 | 0.09659 | 0.00953 | **0.07961** |
| **B_RAW** | 4 | 0.0774 | 85.5 | 2.670 | 175 | 0.1810 | 294 | 0.3040 | 1.6800 | 0.09659 | **0.07961** | 0.01375 |
| **B_RAW** | 5 | 0.0459 | 69.6 | 2.956 | 173 | 0.1789 | 257 | 0.2658 | 1.4855 | 0.09659 | 0.01375 | 0.03585 |
| **B_RAW** | 6 | 0.0603 | 65.3 | 2.731 | 40 | 0.0414 | 246 | 0.2544 | 6.1500 | 0.09659 | 0.03585 | 0.01079 |
| **B_RAW** | 7 | 0.0472 | 59.4 | 2.623 | 39 | 0.0403 | 228 | 0.2358 | 5.8462 | 0.09659 | 0.01079 | 0.01641 |
| **B_LOG** | 2 | 0.1045 | 114.2 | 2.809 | 478 | 0.4943 | 489 | 0.5057 | 1.0230 | 0.09456 | 0.09456 | 0.01198 |
| **B_LOG** | 3 | 0.0945 | 108.0 | 2.435 | 255 | 0.2637 | 396 | 0.4095 | 1.5529 | 0.09456 | 0.01198 | **0.08792** |
| **B_LOG** | 4 | 0.0778 | 91.5 | 2.570 | 191 | 0.1975 | 266 | 0.2751 | 1.3927 | 0.09456 | **0.08792** | 0.02077 |
| **B_LOG** | 5 | 0.0648 | 78.2 | 2.682 | 127 | 0.1313 | 243 | 0.2513 | 1.9134 | 0.09456 | 0.02077 | 0.01228 |
| **B_LOG** | 6 | 0.0724 | 71.4 | 2.620 | 46 | 0.0476 | 220 | 0.2275 | 4.7826 | 0.09456 | 0.01228 | 0.02620 |
| **B_LOG** | 7 | 0.0609 | 64.5 | 2.576 | 45 | 0.0465 | 217 | 0.2244 | 4.8222 | 0.09456 | 0.02620 | 0.02616 |

---

## Validation Status

Validated by `validate_spectral_experiment.py`:
- ✅ All expected metric CSVs, assignment CSVs, plots, and metadata exist.
- ✅ All SEQN participant IDs match 1-to-1 with baseline cohort files.
- ✅ 100% graph connectivity confirmed ($N_{\text{components}} = 1$) for all candidate matrices.
- ✅ Zero missing/NaN values across all metrics and cluster assignments.
