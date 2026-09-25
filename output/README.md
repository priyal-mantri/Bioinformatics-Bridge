# output/

This directory contains all generated research artifacts from the analytical pipeline. Files here are produced programmatically from raw NHANES source data in `DATASET/` and should not be manually edited.

## Contents

```
output/
├── merged_raw.csv                       # Stage 1 merged dataset
├── preprocessed.csv                     # Stage 2 preprocessed cohort
├── analysis_cohort_a_broad.csv          # Cohort A analysis file (broad, non-fasting)
├── analysis_cohort_b_fasting.csv        # Cohort B analysis file (fasting)
├── snapshot_decision_002_bp_averaged.csv
├── snapshot_decision_003_age_filtered.csv
├── snapshot_decision_004_dexa_filtered.csv
├── snapshot_decision_005_insulin_lod.csv
├── snapshot_decision_006_derived_features.csv
├── snapshot_decision_007_outlier_bp_invalid.csv
├── snapshot_decision_008_009_cohorts.csv
├── snapshot_decision_010_skewness_evaluated.csv
├── feature_matrices/                    # Stage 3: candidate feature matrices
├── scaled_matrices/                     # Stage 4: Z-score standardized matrices
├── pca/                                 # Stage 5: PCA exploratory outputs
├── eda/                                 # Exploratory data analysis outputs
└── phenotype_characterization/          # Stage 8: cluster characterization outputs
```

## Output Categories

### Pipeline outputs

These are the primary intermediate and final datasets produced at each stage of the analytical pipeline.

| File | Stage | Description |
| :--- | :---: | :--- |
| `merged_raw.csv` | 1 | All 9 NHANES files joined on SEQN (9,254 rows × 34 cols) |
| `preprocessed.csv` | 2 | After BP averaging, age filtering, DEXA filtering, LOD handling, and derived ratio calculation (5,569 rows × 39 cols) |
| `analysis_cohort_a_broad.csv` | 2 | Cohort A: non-fasting participants with complete core biomarkers (4,482 rows) |
| `analysis_cohort_b_fasting.csv` | 2 | Cohort B: fasting participants with complete extended biomarkers (967 rows) |
| `feature_matrices/` | 3 | Four candidate feature matrices (A_RAW, A_LOG, B_RAW, B_LOG) |
| `scaled_matrices/` | 4 | Z-score standardized versions of each candidate matrix |
| `pca/` | 5 | PCA scores, loadings, variance tables, diagnostic plots |
| `phenotype_characterization/` | 8 | Cluster profiles, statistical tests, literature concordance, figures |

### Preprocessing audit snapshots

The `snapshot_decision_*.csv` files capture the dataset state immediately after each preprocessing decision (002–010). These are preserved for reproducibility and audit purposes — they allow any step of the preprocessing chain to be verified independently. They are not used as inputs to downstream analysis; `preprocessed.csv` is the canonical preprocessed output.

### EDA outputs

The `eda/` directory contains exploratory data analysis outputs (distributions, missingness, correlations, demographics, outliers). EDA was conducted before feature matrix construction as an audit and orientation step, not as primary clustering evidence.

