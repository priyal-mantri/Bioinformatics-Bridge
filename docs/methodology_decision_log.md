# Methodology Decision Log

### Bioinfo Bridge — Research 2 (2026)
*Mapping Ayurvedic Dosha Phenotypes to Human Biology Using Unsupervised Machine Learning*

---

## Overview

This document records key methodological decisions made during the feature matrix preparation, standardization, and principal component analysis (PCA) stages of the pipeline. It serves as an audit trail for peer-review defense and methods documentation.

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

## Summary Matrix

| Decision ID | Stage | Topic | Decision Outcome |
| :--- | :--- | :--- | :--- |
| **Decision 011** | Feature Matrices | Cohort & Transformation Strategy | Established 4 candidate matrices (A_RAW, A_LOG, B_RAW, B_LOG); excluded derived ratios to prevent collinearity |
| **Decision 012** | Scaling | Standardization | Independent Z-score scaling ($\mu=0, \sigma=1$) across all 4 candidate matrices |
| **Decision 013** | PCA Exploration | Input Space for Clustering | PCA performed for exploratory audit only; main clustering to run on full standardized feature space |
