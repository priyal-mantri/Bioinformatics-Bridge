# output/feature_matrices/

This directory contains the four candidate feature matrices produced in Stage 3 of the pipeline. Each matrix is a complete-case subset of the preprocessed cohort, filtered to participants with non-missing values for all features in that representation.

## Files

| File | Description |
| :--- | :--- |
| `A_RAW.csv` | Cohort A, raw scale |
| `A_LOG.csv` | Cohort A, log1p-transformed |
| `B_RAW.csv` | Cohort B, raw scale |
| `B_LOG.csv` | Cohort B, log1p-transformed |
| `feature_matrices_metadata.json` | Construction parameters and provenance record |

## Matrix Dimensions

| Matrix | Cohort | N | Features (p) | Transformation |
| :--- | :--- | ---: | ---: | :--- |
| A_RAW | Broad (non-fasting) | 4,482 | 19 | None |
| A_LOG | Broad (non-fasting) | 4,482 | 19 | log1p |
| B_RAW | Fasting | 967 | 24 | None |
| B_LOG | Fasting | 967 | 24 | log1p |

## Cohort Distinction

**Cohort A** (Broad) includes participants with complete values across the 19 core non-fasting biomarkers: body composition (BMI, waist), cardiovascular (systolic BP, diastolic BP), lipids (total cholesterol, HDL), hepatic (albumin, total protein, bilirubin), renal (creatinine, uric acid, BUN), and electrolytes (sodium, potassium, calcium, phosphorus), plus resting pulse.

**Cohort B** (Fasting) adds fasting glucose, fasting insulin, triglycerides, ALT, and all five DEXA body composition metrics (total fat %, total lean mass, bone mineral density, and arm/leg body composition), requiring a fasting blood draw and valid DEXA scan.

## RAW vs. LOG

The RAW matrices contain values on their original measurement scales. The LOG matrices apply `log1p` to features whose empirical skewness exceeded the |S| > 1.0 threshold in this cohort. The transformation is applied per-feature and per-cohort; the skewness evaluation is documented in `docs/variable_selection_analysis.md`.

Both representations are retained to allow downstream comparison and sensitivity analysis.

## Excluded Variables

Derived ratios (HOMA-IR, TC/HDL ratio, TG/HDL ratio) were calculated during Stage 2 preprocessing for audit purposes but were **not included** in these feature matrices. Including derived ratios alongside their constituent biomarkers would introduce mathematical collinearity into the distance and covariance structures used by downstream clustering algorithms. The rationale is recorded in `docs/methodology_decision_log.md` (Decision 011).
