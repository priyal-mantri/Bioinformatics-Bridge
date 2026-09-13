# Spectral Affinity Calibration Diagnostic

**Project**: Bioinfo Bridge — Research 2 (NHANES 2017–2018)  
**Stage**: Pre-Experiment 4 — Spectral Clustering Affinity Calibration  
**Branch**: `experiment/clustering-sandbox`  
**Status**: READ-ONLY DIAGNOSTIC — No clustering performed. No assignments created. No final decisions made.

---

## Purpose

This diagnostic was run prior to designing Experiment 4 (Spectral Clustering) to empirically characterize the pairwise distance geometry and kNN graph connectivity properties of the four Z-score standardized candidate matrices. Its outputs inform principled, data-driven hyperparameter selection for the affinity construction step of Spectral Clustering (RBF bandwidth `gamma` and kNN `n_neighbors`).

**This directory does NOT contain**:
- Cluster assignments
- K=2..7 Spectral Clustering results
- Final model-selection decisions
- Modifications to the master methodology decision log

---

## Input Matrices

| Matrix | Cohort | Transform | N | p |
|:-------|:-------|:----------|--:|--:|
| `A_RAW_scaled.csv` | Broad | Raw | 4,482 | 19 |
| `A_LOG_scaled.csv` | Broad | Log1p | 4,482 | 19 |
| `B_RAW_scaled.csv` | Fasting | Raw | 967 | 24 |
| `B_LOG_scaled.csv` | Fasting | Log1p | 967 | 24 |

All matrices were independently Z-score standardized (Decision 012, commit `6f5b3d4`). Main clustering operates on the full standardized feature space; PCA is for post-hoc visualization only (Decision 013).

---

## Diagnostic Scripts

| Script | Purpose |
|:-------|:--------|
| `run_affinity_diagnostics.py` | Main diagnostic pipeline (read-only) |

---

## Outputs

### metrics/
| File | Content |
|:-----|:--------|
| `diag1_distance_distribution.csv` | Full pairwise d² distribution statistics and gamma_ref per matrix |
| `diag2_knn_connectivity.csv` | kNN graph connectivity for k ∈ {5,7,10,12,15,20} per matrix |
| `diag3_laplacian_eigenvalues.csv` | First 10 normalized Laplacian eigenvalues and eigengaps per matrix |

### plots/
| File | Content |
|:-----|:--------|
| `diag1_distance_distribution.png` | Pairwise d² percentile bar chart across all matrices |
| `diag2_knn_connectivity.png` | Number of graph components vs k (connectivity sweep) |
| `diag3_laplacian_eigenvalues.png` | Laplacian eigenvalue spectrum and eigengap bar charts |

### metadata/
| File | Content |
|:-----|:--------|
| `affinity_diagnostics_metadata.json` | Methodological notes, hyperparameter rules, results summary |

---

## Methodological Notes

### Pairwise Distance Computation
Full exact pairwise squared Euclidean distances computed via `scipy.spatial.distance.pdist` (condensed form). No random sampling used. For N=4,482: 10,041,921 unique pairs → ~80 MB condensed float64.

### RBF Reference Gamma
`gamma_ref = 1 / (2 * median_d²)`.  
**Important**: Under this gamma, pairs at the MEDIAN distance receive affinity exp(−0.5) ≈ 0.607 — NOT 0.5. The median heuristic sets gamma so that half of all pairwise distances produce affinity above exp(−0.5) ≈ 0.607.

### Graph Symmetrization Rule
`W = (A + A^T) / 2`. Applied consistently for all k and all matrices. Preserves weight scale of the directed kNN graph.

### Diagnostic k Selection Rule for Eigenvalues
The smallest k in {5, 7, 10, 12, 15, 20} that yields a fully connected graph (n_components = 1) for that matrix. If no tested k achieves connectivity, the largest k is used. This rule is deterministic and does not optimize for any eigenvalue profile.

### Eigengap Interpretation
Eigengaps of the normalized graph Laplacian are evidence about graph partition structure. They are NOT proof of the biologically correct number of clusters.
