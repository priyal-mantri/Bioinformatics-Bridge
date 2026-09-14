# Experiment 5: Hierarchical / Agglomerative Clustering Sandbox

**Project**: Bioinfo Bridge — Research 2 (NHANES 2017–2018)  
**Stage**: Sandbox Experiment 5 — Hierarchical / Agglomerative Clustering  
**Branch**: `experiment/clustering-sandbox`  
**Status**: COMPLETED & AUDIT VALIDATED (100% Clean)

---

## Executive Summary

This experiment evaluates the phenotypic partition structure identified by **Agglomerative Hierarchical Cluster Analysis (AHCA)** operating directly on the full standardized Euclidean feature spaces across all four candidate matrices (`A_RAW`, `A_LOG`, `B_RAW`, `B_LOG`) for cluster counts $K = 2 \dots 7$.

It represents the **FINAL model family benchmark** in the project's unsupervised machine learning series, complementing previous evaluations of **K-Means** (centroid variance minimization), **GMM** (soft probabilistic density), **Spectral Clustering** (graph Laplacian manifold embedding), and **HDBSCAN** (density-connected core/noise separation).

---

## Methodological Setup & Parameters

1. **Feature Space**: All clustering was executed directly on full Z-score standardized feature spaces ($p=19$ for Cohort A, $p=24$ for Cohort B).
2. **PCA Role**: PCA coordinate projection was used **STRICTLY for 2D post-hoc visualization** (Decision 013).
3. **Distance Metric**: Continuous Euclidean distance ($L_2$ norm) across standardized Z-scores ($\mu=0, \sigma=1$).
4. **Linkage Evaluated**:
   - **Primary Linkage**: **Ward (`ward`)** — Directly minimizes total within-cluster variance ($\text{ESS}$).
   - **Sensitivity Analysis Linkages**: **Complete (`complete`)** (maximum pairwise distance) and **Average (`average`)** (mean pairwise distance).
5. **Cluster Range**: $K \in \{2, 3, 4, 5, 6, 7\}$.
6. **Subsampling Stability Protocol**: Participant-overlap Adjusted Rand Index ($\text{ARI}$) across $B=100$ iterations with $80\%$ random subsamples without replacement. Overlapping participant cluster labels were compared between baseline full-dataset cuts and subsample cuts.

---

## Cross-Matrix Numerical Results Summary

### Primary Ward Linkage Solutions ($K=2 \dots 7$)

| Matrix | K | Silhouette | CH Score | DB Index | Within WSS | Min Size | Min Prop | Max Size | Max Prop | Mean ARI (Stability) | 5th-95th %ile ARI | Small Cluster Flag |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| **A_RAW** | 2 | 0.0593 | 277.5 | 3.781 | 72266.3 | 2002 | 0.4467 | 2480 | 0.5533 | **0.2013** | [0.1587, 0.2642] | False |
| **A_RAW** | 3 | 0.0603 | 229.1 | 2.794 | 70337.8 | 1342 | 0.2994 | 1629 | 0.3635 | **0.1901** | [0.1345, 0.2488] | False |
| **A_RAW** | 4 | 0.0609 | 210.5 | 2.720 | 67909.1 | 754 | 0.1682 | 1488 | 0.3320 | **0.1938** | [0.1388, 0.2512] | False |
| **A_RAW** | 5 | 0.0518 | 197.7 | 2.854 | 65860.5 | 450 | 0.1004 | 1445 | 0.3224 | **0.1919** | [0.1412, 0.2490] | False |
| **A_RAW** | 6 | 0.0143 | 186.5 | 2.948 | 64287.4 | 437 | 0.0975 | 1195 | 0.2666 | **0.1715** | [0.1301, 0.2210] | False |
| **A_RAW** | 7 | 0.0196 | 176.8 | 2.703 | 63013.9 | 370 | 0.0826 | 1081 | 0.2412 | **0.1690** | [0.1311, 0.2185] | False |
| **A_LOG** | 2 | 0.0552 | 307.7 | 3.565 | 71830.4 | 2004 | 0.4471 | 2478 | 0.5529 | **0.2424** | [0.1852, 0.3105] | False |
| **A_LOG** | 3 | 0.0500 | 260.2 | 3.352 | 69408.2 | 1046 | 0.2334 | 1856 | 0.4141 | **0.2056** | [0.1498, 0.2678] | False |
| **A_LOG** | 4 | 0.0524 | 232.9 | 3.094 | 67215.1 | 682 | 0.1522 | 1634 | 0.3646 | **0.2218** | [0.1584, 0.2812] | False |
| **A_LOG** | 5 | 0.0257 | 202.8 | 3.484 | 65809.8 | 586 | 0.1307 | 1512 | 0.3373 | **0.1932** | [0.1415, 0.2468] | False |
| **A_LOG** | 6 | 0.0185 | 182.8 | 3.372 | 64756.2 | 473 | 0.1055 | 1146 | 0.2557 | **0.1908** | [0.1402, 0.2420] | False |
| **A_LOG** | 7 | 0.0164 | 169.3 | 3.541 | 63842.1 | 425 | 0.0948 | 1022 | 0.2280 | **0.1875** | [0.1390, 0.2378] | False |
| **B_RAW** | 2 | 0.0818 | 81.2 | 3.283 | 21081.7 | 454 | 0.4695 | 513 | 0.5305 | **0.1401** | [0.0812, 0.2085] | False |
| **B_RAW** | 3 | 0.0486 | 73.5 | 2.808 | 19782.5 | 240 | 0.2482 | 390 | 0.4033 | **0.2782** | [0.1925, 0.3680] | False |
| **B_RAW** | 4 | 0.0513 | 67.5 | 2.928 | 18764.1 | 182 | 0.1882 | 315 | 0.3257 | **0.3357** | [0.2250, 0.4410] | False |
| **B_RAW** | 5 | 0.0566 | 62.9 | 2.721 | 17935.4 | 148 | 0.1531 | 275 | 0.2844 | **0.3481** | [0.2315, 0.4590] | False |
| **B_RAW** | 6 | 0.0417 | 56.2 | 2.823 | 17390.8 | 98 | 0.1013 | 252 | 0.2606 | **0.3340** | [0.2280, 0.4375] | False |
| **B_RAW** | 7 | 0.0338 | 51.2 | 2.774 | 16962.3 | 74 | 0.0765 | 239 | 0.2472 | **0.3313** | [0.2215, 0.4350] | False |
| **B_LOG** | 2 | 0.0957 | 97.0 | 2.982 | 20856.2 | 473 | 0.4891 | 494 | 0.5109 | **0.1945** | [0.1250, 0.2710] | False |
| **B_LOG** | 3 | 0.0645 | 87.6 | 2.778 | 19376.1 | 236 | 0.2441 | 398 | 0.4116 | **0.3180** | [0.2185, 0.4120] | False |
| **B_LOG** | 4 | 0.0656 | 76.2 | 2.715 | 18482.4 | 185 | 0.1913 | 302 | 0.3123 | **0.3294** | [0.2240, 0.4285] | False |
| **B_LOG** | 5 | 0.0532 | 67.0 | 2.940 | 17855.9 | 129 | 0.1334 | 268 | 0.2771 | **0.3678** | [0.2510, 0.4780] | False |
| **B_LOG** | 6 | 0.0571 | 60.9 | 2.748 | 17290.4 | 94 | 0.0972 | 248 | 0.2565 | **0.3676** | [0.2485, 0.4760] | False |
| **B_LOG** | 7 | 0.0462 | 54.9 | 2.952 | 16865.0 | 85 | 0.0879 | 231 | 0.2389 | **0.3503** | [0.2360, 0.4550] | False |

---

### Sensitivity Analyses: Complete vs Average Linkage Behavior

#### Complete Linkage (`complete`)
- High apparent Silhouette scores at $K=2$ ($\text{SC} \approx 0.76$ for Cohort A, $\text{SC} \approx 0.61$ for Cohort B).
- **Extreme Degeneracy**: However, this high silhouette score is an mathematical artifact caused by severe cluster imbalance. Complete linkage isolates 1 or 2 extreme outlier participants into singleton clusters ($N_1 = 1$, $N_{\text{min}} = 0.02\%$) while grouping $99.98\%$ of the dataset into a single giant mass.
- Marked as `small_cluster_flag = True` across $K=4 \dots 7$.

#### Average Linkage (`average` / UPGMA)
- Shows moderate-to-high Silhouette scores ($\text{SC} \approx 0.43 \dots 0.58$) and high subsampling stability ($\text{ARI} \approx 0.60 \dots 0.88$).
- **Severe Degeneracy**: Similar to Complete linkage, Average linkage suffers from outlier-peeling artifacts. At $K=2 \dots 7$, it peels off small peripheral sub-branches ($N_1 = 1 \dots 5$ participants) while leaving $99.8\%$ of participants in Cluster 0.

---

## Methodological Key Findings

1. **Low Structural Separation under Ward Linkage**:
   - Primary Ward linkage yields near-zero Silhouette scores ($\text{SC} \approx 0.015 \dots 0.095$) across all matrices and all $K \in \{2 \dots 7\}$.
   - This indicates continuous, unimodal distribution lacking hyper-spherical boundaries in the standardized feature space.

2. **Severe Subsampling Instability under Primary Ward Linkage**:
   - The participant-overlap subsampling stability ($\text{ARI}$) for Ward linkage remains extremely low across all configurations ($\text{Mean ARI} \approx 0.14 \dots 0.36$).
   - This proves that cutting the agglomerative dendrogram produces arbitrary, non-reproducible partitions that fluctuate dramatically under $80\%$ dataset perturbation.

3. **Degenerate Tree Splitting under Sensitivity Linkages**:
   - Complete and Average linkage methods exhibit mathematical "outlier peeling" behavior, isolating single participants ($N_1=1$) to maximize inter-cluster distance bounds rather than identifying multi-participant phenotypic clusters.

---

## Conclusion & Benchmark Recommendation

- **Conclusion**: Agglomerative Hierarchical Clustering does **NOT** reveal robust, reproducible, or scientifically compelling participant cluster structure in the continuous NHANES phenotype feature space.
- **Cross-Model Benchmark Recommendation**:
  - Hierarchical clustering should **NOT** be selected as a winning model for biological downstream interpretation.
  - However, its complete baseline assignment tables, diagnostic metrics, and stability values must be preserved in `clustering_sandbox/hierarchical/` as part of the rigorous, auditable cross-model comparative benchmark.

---

## Validation Status

Validated by `validate_hierarchical_experiment.py`:
- ✅ 100% directory structure, CSV outputs, PNG plot visual artifacts, and JSON metadata present.
- ✅ 100% participant SEQN alignment verified against baseline analysis cohort files.
- ✅ 72 out of 72 experimental runs verified clean with valid numeric ranges and stability bounds.
