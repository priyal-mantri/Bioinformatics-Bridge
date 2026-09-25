# Bioinfo Bridge

### Mapping Ayurvedic Dosha Phenotypes to Human Biology Using Unsupervised Machine Learning
*Research 2 · 2026*

---

## Overview

This repository contains the complete bioinformatics pipeline for an exploratory population health study applying **unsupervised machine learning** to the **NHANES 2017–2018** health survey dataset. The project investigates whether natural phenotypic groupings discoverable from large-scale biomarker data show directional overlap with physiological descriptions in Ayurvedic constitutional (Prakriti / Tridosha) literature.

The analysis pipeline (Stages 1–8) is fully complete. All outputs are preserved in `output/`.

## Research Questions

**Primary**: Can unsupervised machine learning discover natural phenotypic groupings in a human population using biomarker data, without being given predefined constitutional labels?

**Secondary**: If natural groupings are discoverable, how do their physiological profiles compare directionally with characteristics described in Ayurvedic constitutional literature?

> **Scope note**: This project does not claim to prove that Dosha groups exist as biological categories, nor does it assign cluster labels to specific Doshas. Any comparison with Ayurvedic literature is exploratory and directional only.

---

## Methodological Principles

1. **Independent variable selection**: Biomarkers were selected because they represent major physiological systems — never because they match expected Ayurvedic descriptions.
2. **Dual cohort strategy**:
   - **Cohort A** (Broad, N = 4,482, 19 features): Maximises sample size across non-fasting serum biomarkers.
   - **Cohort B** (Fasting, N = 967, 24 features): Adds fasting glucose, fasting insulin, triglycerides, and DEXA body composition metrics.
3. **Data-driven skewness criterion**: log1p transformations are applied only where empirical skewness exceeds |S| > 1.0.
4. **Exclusion of derived ratios**: HOMA-IR, TC/HDL, and TG/HDL are calculated during preprocessing for audit purposes but excluded from all feature matrices to prevent collinearity.
5. **Stage-gated audit trail**: Every stage is independently validated and recorded; preprocessing snapshots are preserved in `output/`.

---

## Repository Structure

```
Bioinformatics-Bridge/
│
├── pipeline/                            # Core Python package
│   ├── config.py                        # Single source of truth: variables, paths, merge schemas
│   ├── merge.py                         # Stage 1: Dataset merge orchestration
│   ├── utils.py                         # Shared helper functions and data I/O
│   ├── preprocess/                      # Stage 2: Preprocessing sub-package
│   │   ├── bp_averaging.py              # BP averaging across up to 3 readings
│   │   ├── filters.py                   # Age (20–80), DEXA validity, insulin LOD filtering
│   │   └── derived_features.py          # HOMA-IR, TC/HDL, TG/HDL calculations (audit only)
│   ├── feature_matrices.py              # Stage 3: Candidate feature matrix construction
│   ├── scaling.py                       # Stage 4: Z-score standardization pipeline
│   ├── pca.py                           # Stage 5: Exploratory PCA analysis and validation
│   └── characterization/               # Stage 8: Phenotype characterization modules
│
├── scripts/                             # Executable entry points (run these)
│   ├── run_pipeline.py                  # Stage 1
│   ├── run_preprocess.py                # Stage 2
│   ├── run_feature_matrices.py          # Stage 3
│   ├── run_scaling.py                   # Stage 4
│   ├── run_pca.py                       # Stage 5
│   ├── run_eda.py                       # Exploratory Data Analysis (audit)
│   └── run_characterization.py          # Stage 8
│
├── output/                              # All pipeline outputs and data artifacts
│   ├── merged_raw.csv                   # Stage 1 merged dataset (9,254 rows × 34 cols)
│   ├── preprocessed.csv                 # Stage 2 preprocessed cohort (5,569 rows × 39 cols)
│   ├── analysis_cohort_a_broad.csv      # Primary Cohort A (4,482 rows × 26 cols)
│   ├── analysis_cohort_b_fasting.csv    # Secondary Cohort B (967 rows × 31 cols)
│   ├── snapshot_decision_*.csv          # Preprocessing audit snapshots (Decisions 002–010)
│   ├── feature_matrices/                # Stage 3: A_RAW, A_LOG, B_RAW, B_LOG
│   ├── scaled_matrices/                 # Stage 4: standardized matrices + metadata JSON
│   ├── pca/                             # Stage 5: scores, loadings, variance tables, plots
│   ├── eda/                             # EDA audit outputs
│   └── phenotype_characterization/      # Stage 8: all characterization outputs
│       ├── cohort_a/                    # Cluster profiles and statistical tests, Cohort A
│       ├── cohort_b/                    # Cluster profiles and statistical tests, Cohort B
│       ├── literature_concordance/      # Directional concordance tables (A, B, and combined)
│       ├── sensitivity/                 # Cross-representation sensitivity analysis
│       ├── plots/                       # Figures 1–6
│       └── metadata.json               # Stage 8 run provenance and join-integrity record
│
├── docs/                                # Methodology documentation and decision logs
│   ├── preprocessing_decision_log.md    # Decisions 001–010: preprocessing rationale
│   ├── variable_selection_analysis.md   # Skewness audit and transformation rationale
│   ├── methodology_decision_log.md      # Decisions 011–014: matrices, scaling, PCA, K-selection
│   └── Research2_Final_Operationalization.md  # Ayurvedic variable operationalization mapping
│
├── tests/                               # Dataset integrity tests
├── DATASET/                             # Raw NHANES source files (not tracked by Git)
├── requirements.txt
└── README.md
```

> **Clustering experiments**: Experiments 1–6 (K-Means, GMM, Spectral, Hierarchical, HDBSCAN, cross-model benchmark) were conducted on the `experiment/clustering-sandbox` branch. The full computational evidence is preserved there under `clustering_sandbox/`. Only the formal decision record (Decision 014) and the cluster assignments used in Stage 8 are maintained on `master`.

---

## Biological Systems and Features

Feature selection spans **7 major physiological systems**. Derived ratios were excluded from all feature matrices to prevent collinearity with their constituent biomarkers.

| Biological System | Primary Variables | Cohort A (p=19) | Cohort B (p=24) |
| :--- | :--- | :---: | :---: |
| Body Composition | BMI, Waist, Total fat %, Total lean mass, Bone mineral density | BMI, Waist | All 5 |
| Cardiovascular | Avg. systolic BP, Avg. diastolic BP, Resting pulse | Sys, Dia BP | All 3 |
| Glucose Metabolism | Fasting glucose, Fasting insulin | — | Both |
| Lipid Profile | Total cholesterol, HDL cholesterol, Triglycerides | TC, HDL | All 3 |
| Hepatic Function | ALT, Albumin, Total protein, Total bilirubin | Albumin, Protein, Bilirubin | All 4 |
| Renal Function | Serum creatinine, Uric acid, BUN | All 3 | All 3 |
| Electrolytes / Minerals | Sodium, Potassium, Calcium, Phosphorus | All 4 | All 4 |

---

## Pipeline Stages

### Stage 1 — Dataset Merging

Merges 9 NHANES 2017–2018 XPT/CSV files on `SEQN` using left-joins anchored on the Demographics file (`DEMO_J`).

- **Input**: Raw NHANES CSVs in `DATASET/`
- **Output**: `output/merged_raw.csv` (9,254 rows × 34 columns)
- **Entry point**: `scripts/run_pipeline.py`

### Stage 2 — Preprocessing and Cohort Filtering

Applies blood pressure averaging (up to 3 readings), age filtering (20–80 years), DEXA validity filtering, insulin limit-of-detection handling, and derived-ratio calculations (audit only).

- **Input**: `output/merged_raw.csv`
- **Output**: `output/preprocessed.csv` (5,569 rows × 39 columns); audit snapshots in `output/snapshot_decision_*.csv`
- **Entry point**: `scripts/run_preprocess.py`
- **Decision log**: `docs/preprocessing_decision_log.md`

### Stage 3 — Candidate Feature Matrix Construction

Constructs four candidate matrices using complete-case analysis and empirical skewness evaluation (|S| > 1.0 threshold for log1p):

| Matrix | Cohort | Transformation | N | p |
| :--- | :--- | :--- | ---: | ---: |
| A_RAW | Broad | None | 4,482 | 19 |
| A_LOG | Broad | log1p | 4,482 | 19 |
| B_RAW | Fasting | None | 967 | 24 |
| B_LOG | Fasting | log1p | 967 | 24 |

- **Output**: `output/feature_matrices/`
- **Entry point**: `scripts/run_feature_matrices.py`

### Stage 4 — Z-Score Standardization

Each candidate matrix is independently standardized to zero mean (μ = 0) and unit variance (σ = 1). No parameters are shared across cohorts or representations.

- **Output**: `output/scaled_matrices/`
- **Entry point**: `scripts/run_scaling.py`

### Stage 5 — Exploratory PCA

Complete PCA solutions computed for all four scaled matrices (19 PCs for Cohort A; 24 PCs for Cohort B).

Key finding: variance is broadly distributed with no dominant axis (PC1 explains ~14.5%–18.4% of total variance). **Decision 013**: PCA is retained as an exploratory audit tool only; downstream clustering proceeds on the full standardized feature space.

- **Output**: `output/pca/` — scores, loadings, variance tables, scree plots, cumulative variance curves, `pca_exploration_metadata.json`
- **Entry point**: `scripts/run_pca.py`

### Stage 6 — Clustering Experiments (`experiment/clustering-sandbox`)

Five clustering algorithm families were evaluated across K = 2–7 on all four standardized representations.

| Experiment | Method | Key Diagnostics |
| :--- | :--- | :--- |
| Experiment 1 | K-Means | Silhouette, subsampling stability ARI (B=100, 80% subsamples) |
| Experiment 2 | K-Means (extended) | Distance-metric sensitivity, elbow curves |
| Experiment 3 | Gaussian Mixture Models | Full covariance; BIC, AIC, log-likelihood profiles |
| Experiment 4 | Spectral Clustering | kNN affinity (k=10); normalised Laplacian eigengap |
| Experiment 5 | Agglomerative Hierarchical | Ward/Average/Complete/Single linkage; Euclidean and Manhattan sensitivity |
| Experiment 6 (synthesis) | HDBSCAN + Cross-Model Benchmark | min_cluster_size sweep; cross-model ARI/NMI; GMM BIC; spectral eigengap |

All experiments are preserved on the `experiment/clustering-sandbox` branch under `clustering_sandbox/`.

### Stage 7 — Cross-Model Clustering Benchmark

A structured benchmark evaluated K-selection agreement across all model families:

**Formal conclusion (Decision 014)**: No single K achieves consistent, method-independent convergence across all representations and both cohorts.

- **Cohort A**: K=2 is a weak candidate — strongest K-Means subsampling stability (~0.95–0.97), but low cross-model ARI (~0.24–0.26) and weak silhouette (~0.06–0.10). GMM BIC, Spectral, and HDBSCAN do not independently support K=2.
- **Cohort B**: K=3 is a weak candidate — stronger B_RAW K-Means stability (~0.877), secondary spectral eigengap signal, moderate cross-model ARI (~0.26–0.34); GMM BIC favours K=2 and HDBSCAN does not corroborate K=3.

> Neither K=2 (Cohort A) nor K=3 (Cohort B) represents a validated or biologically established cluster count. They are pragmatic candidates for downstream exploratory characterisation only.

### Stage 8 — Phenotype Characterisation and Literature Concordance

The candidate partitions (Cohort A K=2; Cohort B K=3) were characterised using participant-level phenotypic variables. Join integrity was verified at 100% match for both cohorts (N=4,482 and N=967 respectively).

**Outputs** (`output/phenotype_characterization/`):

| Output | Description |
| :--- | :--- |
| `cohort_a/cohort_a_primary_profiles.csv` | Mean ± SD biomarker profiles per cluster, Cohort A |
| `cohort_a/cohort_a_statistical_tests.csv` | Between-cluster statistical tests, Cohort A |
| `cohort_a/cohort_a_membership.csv` | Cluster membership sizes, Cohort A |
| `cohort_b/cohort_b_primary_profiles.csv` | Mean ± SD biomarker profiles per cluster, Cohort B |
| `cohort_b/cohort_b_statistical_tests.csv` | Between-cluster statistical tests, Cohort B |
| `cohort_b/cohort_b_membership.csv` | Cluster membership sizes, Cohort B |
| `literature_concordance/cohort_a_literature_concordance.csv` | Directional concordance with Dosha literature, Cohort A |
| `literature_concordance/cohort_b_literature_concordance.csv` | Directional concordance with Dosha literature, Cohort B |
| `literature_concordance/literature_concordance_summary.csv` | Combined concordance summary (both cohorts) |
| `sensitivity/cross_model_pairwise_ari.csv` | Cross-representation ARI sensitivity |
| `sensitivity/raw_vs_log_feature_persistence.csv` | RAW vs. LOG feature characterisation consistency |
| `plots/fig1_cohort_a_zscore_heatmap.png` | Z-score heatmap — Cohort A clusters |
| `plots/fig2_cohort_b_zscore_heatmap.png` | Z-score heatmap — Cohort B clusters |
| `plots/fig3_cohort_a_discriminating_features_boxplots.png` | Top discriminating features, Cohort A |
| `plots/fig4_cohort_b_discriminating_features_boxplots.png` | Top discriminating features, Cohort B |
| `plots/fig5_derived_metabolic_ratios.png` | Derived metabolic ratio profiles (audit; excluded from clustering) |
| `plots/fig6_literature_concordance_matrix.png` | Concordance summary matrix across Doshas and cohorts |

- **Entry point**: `scripts/run_characterization.py`

---

## Installation and Execution

### Requirements

Python 3.10+ is required.

```bash
pip install -r requirements.txt
```

### Data Setup

Download the 9 NHANES 2017–2018 XPT datasets from [CDC NHANES](https://www.cdc.gov/nchs/nhanes/) and convert them to CSV in `DATASET/`:

```
DEMO_J.csv   BMX_J.csv   BPX_J.csv   DXX_J.csv   GLU_J.csv
INS_J.csv    HDL_J.csv   TCHOL_J.csv BIOPRO_J.csv
```

### Running the Pipeline

Run each stage sequentially from the repository root:

```bash
# Stage 1: Merge raw NHANES datasets
python scripts/run_pipeline.py

# Stage 2: Preprocessing and cohort filtering
python scripts/run_preprocess.py

# Stage 3: Construct candidate feature matrices
python scripts/run_feature_matrices.py

# Stage 4: Z-score standardization
python scripts/run_scaling.py

# Stage 5: Exploratory PCA
python scripts/run_pca.py

# Stage 8: Phenotype characterisation and literature concordance
python scripts/run_characterization.py
```

> **Note**: Stages 6 and 7 (clustering experiments and cross-model benchmark) are preserved on the `experiment/clustering-sandbox` branch and are not re-runnable from `master`. The formal outcome (Decision 014) and the K-Means cluster assignments used in Stage 8 are committed to `master`.

---

## Documentation

| Document | Location | Contents |
| :--- | :--- | :--- |
| Preprocessing decision log | `docs/preprocessing_decision_log.md` | Rationale for every preprocessing step (Decisions 001–010) |
| Variable selection and transformation | `docs/variable_selection_analysis.md` | Empirical skewness audit and log-transformation rationale |
| Methodology decision log | `docs/methodology_decision_log.md` | Feature matrices, scaling, PCA, and K-selection decisions (011–014) |
| Operationalization reference | `docs/Research2_Final_Operationalization.md` | Ayurvedic variable-to-biomarker mapping methodology |
| Scaling metadata | `output/scaled_matrices/scaled_matrices_metadata.json` | Per-matrix scaling parameters and provenance |
| PCA metadata | `output/pca/pca_exploration_metadata.json` | PCA variance records and recorded decision |
| Stage 8 metadata | `output/phenotype_characterization/metadata.json` | Stage 8 run provenance and join-integrity record |

---

## License

MIT License — see `LICENSE` for details.
