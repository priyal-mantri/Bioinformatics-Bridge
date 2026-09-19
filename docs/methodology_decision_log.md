# Methodology Decision Log

### Bioinfo Bridge — Research 2 (2026)
*Mapping Ayurvedic Dosha Phenotypes to Human Biology Using Unsupervised Machine Learning*

---

## Overview

This document records key methodological decisions made during the feature matrix preparation, standardization, principal component analysis (PCA), and unsupervised clustering stages of the pipeline. It serves as an audit trail for peer-review defense and methods documentation.

---

## Decision 011 — Candidate Feature Matrix Preparation & Feature Selection

### Context
Construct four candidate feature matrices from preprocessed NHANES 2017–2018 data:
1. `A_RAW`: Broad Cohort, Raw Scale ($N = 4,482$, $p = 19$)
2. `A_LOG`: Broad Cohort, Log1p Scale ($N = 4,482$, $p = 19$)
3. `B_RAW`: Fasting Cohort, Raw Scale ($N = 967$, $p = 24$)
4. `B_LOG`: Fasting Cohort, Log1p Scale ($N = 967$, $p = 24$)

### Rationale
- **Dual Cohort Strategy**: Cohort A maximizes sample size and statistical power across non-fasting serum biomarkers. Cohort B provides deep phenotyping including fasting glucose, fasting insulin, triglycerides, and DEXA body composition metrics.
- **Log Transformation Criterion**: Features with skewness $|S| > 1.0$ (e.g., insulin `LBXIN`, triglycerides `LBXSTR`, uric acid `LBXSUA`) were evaluated for `log1p` transformation to stabilize variance.
- **Exclusion of Derived Ratios**: Derived variables (`HOMA_IR`, `TC_HDL_ratio`, `TG_HDL_ratio`) calculated during preprocessing (Decision 006) were **explicitly excluded** from the candidate feature matrices to prevent collinearity and mathematical redundancy with their constituent primary biomarkers (`LBXGLU`, `LBXIN`, `LBXTC`, `LBDHDD`, `LBXSTR`).

### Status
✅ **Implemented** — Locked at commit `f789516`. Outputs saved in `output/feature_matrices/`.

---

## Decision 012 — Independent Z-Score Standardization

### Context
Standardize each of the four candidate feature matrices prior to dimensionality reduction and unsupervised clustering.

### Rationale
- Biomarkers are measured in disparate physical units ($\text{mg/dL}$, $\text{mmol/L}$, $\text{mmHg}$, $\text{kg/m}^2$, $\text{g/cm}^2$). Unscaled distance-based or variance-based algorithms would be dominated by features with large absolute scales.
- Each matrix was standardized **100% independently** to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$).
- No parameter sharing or data leakage occurred between Cohort A and Cohort B, or between RAW and LOG representations.

### Status
✅ **Implemented** — Locked at commit `6f5b3d4`. Outputs saved in `output/scaled_matrices/`.

---

## Decision 013 — Exploratory PCA & Non-Truncation for Main Clustering

### Context
Compute complete Principal Component Analysis (PCA) solutions for all four scaled matrices ($19$ PCs for Analysis A; $24$ PCs for Analysis B) to evaluate variance distribution, scree profiles, and feature loadings.

### Empirical Findings
- **Variance Distribution**: Total variance is broadly distributed across many components without a single dominant axis:
  - `A_RAW`: PC1 = 14.53%, PC2 = 11.39%, PC3 = 10.03% (Top 3 PCs = 35.95% cumulative variance)
  - `A_LOG`: PC1 = 15.37%, PC2 = 11.64%, PC3 = 10.13% (Top 3 PCs = 37.13% cumulative variance)
  - `B_RAW`: PC1 = 16.90%, PC2 = 13.76%, PC3 = 8.29% (Top 3 PCs = 38.95% cumulative variance)
  - `B_LOG`: PC1 = 18.42%, PC2 = 14.08%, PC3 = 8.38% (Top 3 PCs = 40.89% cumulative variance)
- **Kaiser Rule ($\lambda \ge 1.0$)**: Retains 7 components for Analysis A (~61.7%–62.7% variance) and 8–9 components for Analysis B (~66.8%–69.1% variance).

### Decision & Rationale
> **Decision**: PCA will **NOT** be used as the primary input space for main downstream clustering.
>
> **Rationale**: PCA results demonstrate that variance is broadly distributed across many components. Reducing the data to a small 2D/3D PCA representation would retain only a limited fraction of the total variance. Therefore, downstream clustering will proceed on the **full standardized feature representations**, while complete PCA outputs are preserved as an exploratory audit trail and for post-hoc visualization.

### Status
✅ **Implemented** — Locked at commit `ec526d8` and merged to `master`. Outputs saved in `output/pca/`.

---

## Decision 014 — Cross-Model Clustering Structure and K Selection

### Context

Following the completion of five clustering model families across all four standardized feature representations (A_RAW, A_LOG, B_RAW, B_LOG), a cross-model benchmark was performed to determine whether the data provide convergent evidence for a particular number of clusters (K). This constituted Experiment 6 in the clustering sandbox.

The five model families evaluated were:
1. K-Means (K=2–7; participant-overlap subsampling stability, B=100, 80% subsamples)
2. Gaussian Mixture Model (GMM; K=2–7; full covariance; BIC model selection)
3. Spectral Clustering (K=2–7; kNN k=10 affinity; eigengap diagnostics)
4. HDBSCAN (min_cluster_size sweep; density-structure sanity check)
5. Hierarchical / Agglomerative Clustering (Ward linkage, Euclidean distance, K=2–7; subsampling stability B=100; Complete and Average linkage sensitivity; Euclidean and Manhattan distance-metric sensitivity)

The benchmark was run on the `experiment/clustering-sandbox` branch. The full computational evidence is preserved in `clustering_sandbox/cross_model/` and referenced in PR #4.

### Evidence Considered

The cross-model K assessment incorporated the following classes of evidence:

- **Internal separation metrics**: silhouette score across K=2–7 for all five models and all four matrices
- **K-Means subsampling stability**: adjusted Rand index (ARI) across B=100 participant-overlap subsamples, per K, per matrix
- **Hierarchical-Ward subsampling stability**: ARI across B=100 subsamples, per K, per matrix
- **Cross-model pairwise ARI**: agreement of cluster assignments between all pairs of model families at each K, for each matrix
- **Cross-model pairwise NMI**: normalized mutual information between model-family assignments at each K, for each matrix
- **Cluster-size balance**: ratio of smallest to largest cluster across K values and models
- **RAW vs. LOG comparison**: consistency of evidence across raw-scale and log1p-scale representations within each cohort
- **Cohort A vs. Cohort B comparison**: consistency of evidence across the broad cohort (A; N=4,482; p=19) and the fasting cohort (B; N=967; p=24)
- **GMM BIC diagnostics**: BIC model-selection profile across K=2–7 for full covariance structures
- **Spectral eigengap diagnostics**: spectral gap profile $\lambda_k - \lambda_{k-1}$ across K=2–7, evaluated per matrix
- **HDBSCAN density-structure diagnostics**: presence or absence of large stable density-separated groups across the min_cluster_size parameter sweep
- **Hierarchical linkage sensitivity**: Ward vs. Complete vs. Average linkage; behavior of cophenetic structure and dendrogram stability
- **Hierarchical distance-metric sensitivity**: Euclidean vs. Manhattan distance; reproducibility of partition quality across metrics

### Findings

#### Cohort A (A_RAW, A_LOG; N=4,482; p=19)

- **K-Means stability**: K=2 exhibited exceptionally high subsampling stability (A_RAW: ~0.965; A_LOG: ~0.951). This was the strongest individual reproducibility signal observed across all model families in Cohort A.
- **Cluster balance**: K=2 partitions were reasonably balanced in both A_RAW and A_LOG.
- **Silhouette**: Silhouette scores were weak throughout Cohort A across all K values and all models (approximately 0.06–0.10). No K achieved strong internal separation.
- **Cross-model ARI**: Cross-model pairwise ARI at K=2 was low (approximately 0.24 in A_RAW; 0.26 in A_LOG), indicating that the K-Means K=2 partition was not independently recovered with comparable fidelity by other model families.
- **GMM BIC**: BIC did not consistently favour K=2 in Cohort A. BIC minimum was observed at K=4 for A_RAW and K=3 for A_LOG. GMM therefore does not provide independent support for K=2.
- **Spectral**: No strong primary eigengap at K=2 was observed in Cohort A. No clear convergent spectral signal supported a two-group partition.
- **HDBSCAN**: The HDBSCAN min_cluster_size sweep did not recover large stable density-separated groups in Cohort A at any min_cluster_size setting. HDBSCAN does not independently support a two-group density structure.
- **Hierarchical Ward stability**: Ward linkage subsampling stability was low throughout Cohort A (approximately 0.14–0.37 across K values), and did not independently confirm any K.
- **RAW vs. LOG consistency**: K-Means K=2 stability was similarly high in both A_RAW and A_LOG, providing some internal cross-representation consistency. However, this consistency did not extend across model families.

#### Cohort B (B_RAW, B_LOG; N=967; p=24)

- **K-Means stability**: K=3 showed particularly strong stability in B_RAW (~0.877), exceeding B_RAW K=2 (~0.755). B_LOG showed a more mixed pattern: K=2 was somewhat more stable (~0.842) than K=3 (~0.783). The B_RAW K=3 stability was the strongest individual argument for K=3 in Cohort B.
- **Cluster balance**: K=3 cluster sizes were reasonably balanced in B_RAW.
- **Silhouette**: Silhouette scores remained weak throughout Cohort B across all models and K values. No K achieved strong internal separation.
- **Cross-model ARI**: Cross-model pairwise ARI at K=3 was somewhat higher than in Cohort A K=2 (approximately 0.26–0.34), suggesting marginally more model-independent convergence, though the level of agreement remained moderate rather than strong.
- **Spectral eigengap**: A secondary eigengap signal was observed around the third spectral cutoff (B_RAW: $\lambda_3 \to \lambda_4$ gap $\approx$ 0.080; B_LOG: $\lambda_3 \to \lambda_4$ gap $\approx$ 0.088). This constitutes weak-to-secondary evidence for a possible K=3 spectral structure. The primary eigengap in Cohort B remained at K=2, and the secondary signal does not overcome the low silhouette.
- **GMM BIC**: BIC minimum favoured K=2 in both B_RAW and B_LOG. GMM therefore opposes K=3 in Cohort B on model-selection grounds.
- **HDBSCAN**: The HDBSCAN parameter sweep did not independently recover a stable three-group density structure in Cohort B at any calibration setting.
- **Hierarchical Ward stability**: Hierarchical Ward stability was low throughout Cohort B and did not independently support K=3.
- **RAW vs. LOG consistency**: B_RAW favoured K=3 on stability grounds; B_LOG weakly favoured K=2. This cross-representation inconsistency constitutes a limitation on the strength of the K=3 inference.

### Formal Decision

> **No single K is sufficiently supported consistently across clustering methods, representations, and cohorts. Therefore, no universal cluster count is adopted as a definitive number of phenotype groups.**

Two cohort-specific exploratory candidate partitions are retained for possible downstream characterization:

- **Cohort A: K=2 exploratory candidate partition**
- **Cohort B: K=3 exploratory candidate partition**

These candidate partitions are **exploratory** and do **not** constitute evidence that the corresponding K represents a validated biological population structure. They are retained because they showed comparatively stronger evidence within their respective cohorts relative to other tested K values. They are not proposed as definitive natural cluster counts, nor as validated boundaries of distinct biological subpopulations.

### Rationale for Retaining Candidates

- K=2 has the strongest individual reproducibility signal in Cohort A (K-Means subsampling stability ~0.95–0.97), and K=2 partitions are reasonably balanced. Despite the failure of other model families to converge on this partition, K=2 provides a reproducible exploratory starting point for downstream phenotypic characterization in Cohort A.
- K=3 has comparatively stronger secondary structural and reproducibility evidence in Cohort B (B_RAW K-Means stability ~0.877; secondary spectral eigengap signal; moderate cross-model ARI). These constitute the strongest secondary signals observed in Cohort B, even though the GMM and HDBSCAN evidence does not support K=3.
- Neither candidate achieves sufficient method-independent convergence to be declared a definitive number of phenotype groups. Retaining both candidates allows downstream phenotype characterization to proceed without prematurely asserting a natural cluster count that is not robustly established.

### Status

✅ **Implemented** — Locked at merge of PR #5 into `master`. Formal K-selection decision following Experiment 6 (Cross-Model Clustering Benchmark). The full computational evidence base is preserved in the `experiment/clustering-sandbox` branch under `clustering_sandbox/cross_model/` and referenced in PR #4 (`Experiment 6: Cross-model clustering benchmark and K assessment`).

---

## Summary Matrix

| Decision ID | Stage | Topic | Decision Outcome |
| :--- | :--- | :--- | :--- |
| **Decision 011** | Feature Matrices | Cohort & Transformation Strategy | Established 4 candidate matrices (A_RAW, A_LOG, B_RAW, B_LOG); excluded derived ratios to prevent collinearity |
| **Decision 012** | Scaling | Standardization | Independent Z-score scaling ($\mu=0, \sigma=1$) across all 4 candidate matrices |
| **Decision 013** | PCA Exploration | Input Space for Clustering | PCA performed for exploratory audit only; main clustering to run on full standardized feature space |
| **Decision 014** | Clustering — K Selection | Cross-Model Benchmark | No universal K identified; Cohort A K=2 and Cohort B K=3 retained as exploratory candidate partitions only |
