# Bioinfo Bridge

### Mapping Ayurvedic Dosha Phenotypes to Human Biology Using Unsupervised Machine Learning
*Research 2 · 2026*

---

## Overview

This repository contains the end-to-end bioinformatics pipeline for an exploratory population health study applying **unsupervised machine learning** to the **NHANES 2017–2018** health survey dataset.

### Core Research Questions

- **Primary Research Question**:
  > *Can unsupervised machine learning discover natural biological constitutions / phenotypic groupings in humans using population-scale health and biomarker data, without being given predefined constitutional labels?*

- **Secondary Research Question**:
  > *If natural biological groupings are discovered, how do their biological profiles compare with characteristics described in Ayurvedic constitutional (Prakriti / Tridosha) literature?*

### Methodological Principles

1. **Independent Variable Selection**: Biomarkers were selected solely because they represent major physiological systems (body composition, cardiovascular, glucose metabolism, lipid profile, hepatic, renal, electrolyte balance) — never because they match expected Ayurvedic descriptions.
2. **Dual Cohort Strategy**:
   - **Primary Analysis A** (Broad Cohort, $N = 4,482$, 19 features): Maximizes sample size and statistical power across non-fasting serum biomarkers.
   - **Secondary Analysis B** (Fasting Cohort, $N = 967$, 24 features): Incorporates fasting glucose, fasting insulin, triglycerides, and DEXA body composition metrics. *(Note: Derived features such as HOMA-IR and lipid ratios were calculated during preprocessing for exploratory audit but explicitly excluded from candidate feature matrices to prevent collinearity with constituent biomarkers).*
3. **Data-Driven Skewness Criteria**: Candidate log-transformations (`log1p`) are evaluated against empirical cohort skewness thresholds ($|S| > 1.0$) rather than arbitrary hardcoding.
4. **Strict Audit Trail & Stage-Gating**: Every stage (Preprocessing, Matrix Preparation, Scaling, PCA, Clustering Sandbox, Cross-Model Benchmark) is independently locked, validated, and recorded in version control and methodology decision logs.

---

## Repository Structure

```
Bioinformatics-Bridge/
│
├── pipeline/                            # Core Python package
│   ├── __init__.py
│   ├── config.py                        # Single source of truth: variables, paths, merge schemas
│   ├── merge.py                         # Stage 1: Merge orchestration
│   ├── utils.py                         # Shared helper functions & data IO
│   ├── preprocess/                      # Stage 2: Preprocessing sub-package
│   │   ├── __init__.py
│   │   ├── bp_averaging.py              # BP averaging across up to 3 readings
│   │   ├── filters.py                   # Age (20–80), DEXA validity, insulin LOD filtering
│   │   └── derived_features.py          # HOMA-IR, TC/HDL, TG/HDL ratio calculations (audit only)
│   ├── feature_matrices.py              # Stage 3: Candidate feature matrix construction
│   ├── scaling.py                       # Stage 4: Z-score standardization pipeline
│   └── pca.py                           # Stage 5: Exploratory PCA analysis & validation suite
│
├── output/                              # Pipeline outputs & data artifacts
│   ├── merged_raw.csv                   # Stage 1 merged raw dataset (9,254 rows × 34 cols)
│   ├── preprocessed.csv                 # Stage 2 preprocessed cohort (5,569 rows × 39 cols)
│   ├── analysis_cohort_a_broad.csv      # Primary Cohort A (4,482 rows × 26 cols)
│   ├── analysis_cohort_b_fasting.csv    # Secondary Cohort B (967 rows × 31 cols)
│   ├── feature_matrices/                # Stage 3 candidate matrices (A_RAW, A_LOG, B_RAW, B_LOG)
│   ├── scaled_matrices/                 # Stage 4 Z-score standardized matrices
│   └── pca/                             # Stage 5 PCA exploratory outputs & diagnostics
│
├── clustering_sandbox/                  # Stage 6 & 7: Experimental clustering sandbox (on experiment/clustering-sandbox branch & PR #4)
│   ├── kmeans/                          # Experiment 1: K-Means & subsampling stability (B=100)
│   ├── gmm/                             # Experiment 2: Gaussian Mixture Models & BIC selection
│   ├── spectral/                        # Experiment 3: Spectral Clustering & eigengap diagnostics
│   ├── hdbscan/                          # Experiment 4: HDBSCAN density-structure diagnostics
│   ├── hierarchical/                    # Experiment 5: Hierarchical Clustering & distance/linkage sensitivity
│   └── cross_model/                     # Experiment 6: Cross-model benchmark & K assessment
│
├── docs/
│   └── methodology_decision_log.md      # Formal decision audit log (Decisions 011, 012, 013, 014)
├── preprocessing_decision_log.md        # Detailed rationale for every preprocessing step
├── variable_selection_analysis.md       # Empirical skewness audit & transformation rationale
├── run_pipeline.py                      # Stage 1 entry point
├── run_preprocess.py                    # Stage 2 entry point
├── run_feature_matrices.py              # Stage 3 entry point
├── run_scaling.py                       # Stage 4 entry point
├── run_pca.py                           # Stage 5 entry point
├── requirements.txt                     # Python dependencies
└── README.md
```

---

## Biological Systems Covered

The feature selection covers **7 major physiological systems** using 24 primary NHANES biomarkers:

| Biological System | Primary Biomarkers | Included in Cohort A (19) | Included in Cohort B (24) |
| :--- | :--- | :---: | :---: |
| **Body Composition** | BMI (`BMXBMI`), Waist (`BMXWAIST`), Total Fat (`DXDTOPF`), Total Lean (`DXDTOLE`), Bone Density (`DXDTOBMD`) | BMI, Waist | All 5 |
| **Cardiovascular** | Avg Systolic BP, Avg Diastolic BP, Resting Pulse (`BPXPLS`) | BP (Sys, Dia) | All 3 |
| **Glucose Metabolism** | Fasting Glucose (`LBXGLU`), Fasting Insulin (`LBXIN`) *(HOMA-IR derived in preprocessing but excluded from feature matrices to avoid collinearity)* | — | Glucose, Insulin |
| **Lipid Metabolism** | Total Cholesterol (`LBXTC`), HDL (`LBDHDD`), Triglycerides (`LBXSTR`) *(Ratios derived in preprocessing but excluded from feature matrices to avoid collinearity)* | TC, HDL | All 3 |
| **Hepatic Function** | ALT (`LBXSATSI`), Albumin (`LBXSAL`), Total Protein (`LBXSTP`), Total Bilirubin (`LBXSTB`) | Albumin, Protein, Bilirubin | All 4 |
| **Renal Function** | Serum Creatinine (`LBXSCR`), Uric Acid (`LBXSUA`), BUN (`LBXSBU`) | All 3 | All 3 |
| **Electrolytes/Minerals** | Sodium (`LBXSNASI`), Potassium (`LBXSKSI`), Calcium (`LBXSCA`), Phosphorus (`LBXSPH`) | All 4 | All 4 |

---

## Pipeline Execution & Workflow Stages

### Stage 1 — Dataset Merging (`run_pipeline.py`)
Merges 9 NHANES 2017–2018 SAS/CSV files on `SEQN` using left-joins anchored on Demographics (`DEMO_J`).
- **Input**: Raw NHANES CSV files in `DATASET/`
- **Output**: `output/merged_raw.csv` ($9,254$ rows × $34$ columns)

### Stage 2 — Preprocessing & Cohort Filtering (`run_preprocess.py`)
Applies blood pressure averaging, age filtering ($20 \le \text{Age} \le 80$), DEXA validity filtering, insulin limit-of-detection (LOD) handling, and ratio derivations.
- **Input**: `output/merged_raw.csv`
- **Output**: `output/preprocessed.csv` ($5,569$ rows × $39$ columns)

### Stage 3 — Candidate Feature Matrix Construction (`run_feature_matrices.py`)
Constructs four candidate feature matrices with complete case analysis and empirical skewness checks:
- **Outputs**:
  - `output/feature_matrices/A_RAW.csv` ($N=4,482$, $p=19$)
  - `output/feature_matrices/A_LOG.csv` ($N=4,482$, $p=19$)
  - `output/feature_matrices/B_RAW.csv` ($N=967$, $p=24$)
  - `output/feature_matrices/B_LOG.csv` ($N=967$, $p=24$)

### Stage 4 — Z-Score Scaling & Standardization (`run_scaling.py`)
Independently standardizes each feature matrix to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$).
- **Input**: Candidate feature matrices in `output/feature_matrices/`
- **Output**: Scaled feature matrices in `output/scaled_matrices/`

### Stage 5 — Exploratory PCA & Methodological Audit (`run_pca.py`)
Fits complete, independent Principal Component Analysis solutions for all 4 candidate representations ($19$ PCs for A; $24$ PCs for B).
- **Outputs**: Transformed score matrices, loading matrices, variance tables, scree plots, cumulative variance curves, and `output/pca/pca_exploration_metadata.json`.
- **Recorded Methodological Decision (Decision 013)**:
  > *PCA results demonstrate that variance is broadly distributed across many components (PC1 explains only ~14.5%–18.4% of total variance). Therefore, reducing the data to a small 2D/3D PCA representation would retain only a limited fraction of the total variance. PCA will **NOT** be used as the primary input space for downstream clustering; clustering will proceed on the full standardized feature representations, while the complete PCA outputs are preserved as an exploratory audit trail.*

### Stage 6 — Unsupervised Clustering Model Families (`clustering_sandbox/`)
Five clustering model families were evaluated across $K=2\text{--}7$ on the full standardized feature representations (`A_RAW`, `A_LOG`, `B_RAW`, `B_LOG`):
1. **K-Means**: $K=2\text{--}7$, participant-overlap subsampling stability ($B=100$, 80% subsamples).
2. **Gaussian Mixture Models (GMM)**: $K=2\text{--}7$, full covariance structures, BIC model selection.
3. **Spectral Clustering**: $K=2\text{--}7$, kNN $k=10$ affinity graph, Laplacian eigengap diagnostics.
4. **HDBSCAN**: `min_cluster_size` parameter sweep as a density-structure sanity check.
5. **Hierarchical / Agglomerative Clustering**: Ward linkage baseline ($K=2\text{--}7$), Complete and Average linkage sensitivity, Euclidean and Manhattan distance metric sensitivity.

### Stage 7 — Cross-Model Clustering Benchmark & K Assessment (`clustering_sandbox/cross_model/`)
A cross-model benchmark evaluated convergent evidence across internal separation metrics (silhouette score), subsampling stability (ARI), cross-model pairwise agreement (ARI/NMI), cluster-size balance, RAW vs. LOG transformation robustness, Cohort A vs. Cohort B differences, and algorithm-specific diagnostics.

- **Formal Methodological Conclusion (Decision 014)**:
  > *No single cluster count was sufficiently supported consistently across clustering methods, feature representations, and cohorts. Therefore, no universal cluster count was adopted as the definitive number of biological phenotype groups.*

- **Exploratory Candidate Partitions**:
  - **Cohort A**: $K=2$ candidate partition (retained due to high K-Means subsampling stability $\approx 0.95\text{--}0.97$).
  - **Cohort B**: $K=3$ candidate partition (retained due to strong B_RAW K-Means stability $\approx 0.877$, secondary spectral eigengap, and moderate cross-model ARI).
  - *Methodological Note*: These candidate partitions are exploratory starting points for downstream characterization only. They are not treated as validated biological clusters or natural population counts.

*Note: All computational scripts, metrics, figures, and artifacts for Stages 6 and 7 are preserved in the dedicated `experiment/clustering-sandbox` branch and referenced in PR #4 to maintain `master` as the stable research state.*

### Stage 8 — Exploratory Phenotypic Characterization *(Planned Next Stage)*
Subsequent research will characterize the retained Cohort A ($K=2$) and Cohort B ($K=3$) candidate partitions by evaluating participant-level biomarker distributions, standardized mean differences, and domain-specific feature profiles without asserting validated biological cluster boundaries.

---

## Installation & Running the Pipeline

### 1. Requirements

Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Data Setup

Download the 9 NHANES 2017–2018 XPT datasets from CDC NHANES and convert them to CSV in `DATASET/`:
- `DEMO_J.csv`, `BMX_J.csv`, `BPX_J.csv`, `DXX_J.csv`, `GLU_J.csv`, `INS_J.csv`, `HDL_J.csv`, `TCHOL_J.csv`, `BIOPRO_J.csv`

### 3. Pipeline Execution Commands

Run core pipeline stages sequentially on `master`:

```bash
# Stage 1: Merge raw datasets
python run_pipeline.py

# Stage 2: Execute preprocessing & cohort filters
python run_preprocess.py

# Stage 3: Construct candidate feature matrices
python run_feature_matrices.py

# Stage 4: Standardize feature matrices using Z-score scaling
python run_scaling.py

# Stage 5: Run exploratory PCA & compute variance metrics
python run_pca.py
```

*Clustering sandbox experiments (Stages 6–7) and cross-model benchmark evaluations are maintained on the `experiment/clustering-sandbox` branch.*

---

## Documentation Links

- **Methodology Decision Log**: [`docs/methodology_decision_log.md`](docs/methodology_decision_log.md) *(Decisions 011–014: feature matrices, scaling, PCA, and cross-model K selection)*
- **Preprocessing Rationale**: [`preprocessing_decision_log.md`](preprocessing_decision_log.md) *(Decisions 001–010: variable selection, BP averaging, filters, ratios, skewness)*
- **Variable Selection & Skewness Audit**: [`variable_selection_analysis.md`](variable_selection_analysis.md)
- **Scaling Metadata**: [`output/scaled_matrices/scaled_matrices_metadata.json`](output/scaled_matrices/scaled_matrices_metadata.json)
- **PCA Metadata & Recorded Decisions**: [`output/pca/pca_exploration_metadata.json`](output/pca/pca_exploration_metadata.json)
- **Cross-Model Benchmark & Evidence**: Available on branch `experiment/clustering-sandbox` under `clustering_sandbox/cross_model/` and referenced in PR #4.

---

## License

MIT License — see `LICENSE` for details.
