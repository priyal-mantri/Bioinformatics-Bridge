# Variable Selection Analysis
## Research 2: The Bioinformatics Bridge — NHANES Feature Set

---

## Context and Status of This Document

This document was written during the early planning phase of the project, when variables were being explored through a Dosha-proxy lens to assess whether NHANES biomarkers could theoretically map to the three Ayurvedic constitutional types. That framing helped identify which physiological domains to include.

**The Dosha-proxy approach was subsequently rejected as the primary selection criterion** before the pipeline was built. Selecting variables to match expected Dosha patterns would load the expected outcome into the input features, which is circular reasoning for an unsupervised analysis.

The final implemented methodology selected variables based on comprehensive biological system coverage, independently of any expected Dosha association. The rationale for that decision is documented in `preprocessing_decision_log.md` under Decision 001.

This document is preserved as a historical record. Sections 1 and 3 below reflect early planning thinking. The final feature set, exclusion decisions, and merge strategy (Sections 4-8) remain accurate.

---

## 1. Early Planning Hypothesis (not the final methodology)

During initial scoping, the hypothesis was that running unsupervised clustering on NHANES biomarkers might reveal groups whose profiles resemble the three Ayurvedic constitutional types (Doshas):

| Dosha | Hypothesised Biomarker Profile |
|-------|-------------------------------|
| Vata | Low BMI, low lean mass, variable BP, fast metabolism |
| Pitta | High BP, high fasting glucose, high uric acid, elevated liver enzymes |
| Kapha | High bone density, high body fat, high insulin, low resting HR, high waist |

This was a hypothesis to be tested, not a design constraint. The final methodology does not preselect variables to match these profiles.

---

## 2. Final Feature Set

The pipeline uses the following variables, selected for biological system coverage. The "Dosha Proxy" label used in the original planning table has been replaced with the biological system each variable belongs to, which is the actual selection criterion.

| # | Variable Code | Description | Biological System | Source File | Null % |
|---|--------------|-------------|-------------------|-------------|--------|
| 1 | `BMXBMI` | Body Mass Index | Body Composition | `BMX_J` | 8% |
| 2 | `BMXWAIST` | Waist circumference | Body Composition | `BMX_J` | 13% |
| 3 | `BPXPLS` | Resting pulse / heart rate | Cardiovascular | `BPX_J` | 23% |
| 4 | `DXDTOBMD` | Total Body Bone Mineral Density | Body Composition (DEXA) | `DXX_J` | 28% |
| 5 | `DXDTOPF` | Total Body % Fat | Body Composition (DEXA) | `DXX_J` | 29% |
| 6 | `DXDTOLE` | Total Lean Mass (excl. bone) | Body Composition (DEXA) | `DXX_J` | 25% |
| 7 | `LBXGLU` | Fasting Glucose | Glucose Metabolism | `GLU_J` | 5% |
| 8 | `LBXIN` | Fasting Insulin | Glucose Metabolism | `INS_J` | 7% |
| 9 | `LBDHDD` | HDL Cholesterol | Lipid Metabolism | `HDL_J` | 9% |
| 10 | `LBXTC` | Total Cholesterol | Lipid Metabolism | `TCHOL_J` | 9% |
| 11 | `LBXSTR` | Triglycerides | Lipid Metabolism | `BIOPRO_J` | 8% |
| 12 | `LBXSAL` | Albumin | Hepatic Function | `BIOPRO_J` | 8% |
| 13 | `LBXSCR` | Creatinine | Renal Function | `BIOPRO_J` | 8% |
| 14 | `LBXSUA` | Uric Acid | Renal Function | `BIOPRO_J` | 8% |
| 15 | `LBXSATSI` | ALT (liver enzyme) | Hepatic Function | `BIOPRO_J` | 8% |
| 16 | `LBXSTP` | Total Protein | Hepatic Function | `BIOPRO_J` | 8% |
| 17 | `LBXSTB` | Total Bilirubin | Hepatic Function | `BIOPRO_J` | 8% |
| 18 | `LBXSBU` | Blood Urea Nitrogen | Renal Function | `BIOPRO_J` | 8% |
| 19 | `LBXSCA` | Total Calcium | Electrolyte/Mineral | `BIOPRO_J` | 8% |
| 20 | `LBXSPH` | Phosphorus | Electrolyte/Mineral | `BIOPRO_J` | 8% |
| 21 | `LBXSNASI` | Sodium | Electrolyte/Mineral | `BIOPRO_J` | 8% |
| 22 | `LBXSKSI` | Potassium | Electrolyte/Mineral | `BIOPRO_J` | 8% |
| 23 | `Avg_Systolic_BP` | Averaged Systolic BP | Cardiovascular | `BPX_J` | ~27% |
| 24 | `Avg_Diastolic_BP` | Averaged Diastolic BP | Cardiovascular | `BPX_J` | ~27% |

Note: Items 1-19 (no DEXA or fasting variables) constitute Cohort A (N=4,482, 19 features). All 24 constitute Cohort B (N=967). The averaged BP columns replace the six raw BP readings in the feature matrices.

### Control / Confounder Variables (NOT in the feature matrix)

| Variable | Use |
|----------|-----|
| `SEQN` | Participant ID — merge key |
| `RIAGENDR` | Gender — for post-hoc confounder analysis |
| `RIDAGEYR` | Age — for filtering (adults 20-80) |
| `RIDRETH3` | Race/ethnicity — for confounder analysis |

---

## 3. Reasoning Behind Each Variable (historical planning notes)

This section reflects the original per-variable reasoning written during early planning. The Dosha references below are the original hypothesis framing, not a claim about what the clustering found. No Dosha assignments have been made — Stage 8 (downstream characterisation) has not yet been performed.

### Variables primarily associated with body composition in the planning hypothesis

- `BMXBMI` — BMI captures overall adiposity. In the planning hypothesis, low BMI was considered a possible Vata indicator and high BMI a possible Kapha indicator. Selected because it covers Body Composition (System 1).
- `DXDTOLE` (Lean Mass) — DEXA-measured lean mass is more specific than BMI for distinguishing muscular from adipose phenotypes. Selected for Body Composition coverage.
- `LBXSAL` (Albumin) — Low albumin reflects poor nutritional status or high catabolic rate. Selected for Hepatic Function (System 5).

### Variables primarily associated with cardiovascular function in the planning hypothesis

- `Avg_Systolic_BP` and `Avg_Diastolic_BP` — Averaged across readings 2 and 3 following CDC NHANES protocol. Selected for Cardiovascular coverage (System 2).
- `BPXPLS` (Resting HR) — Reflects autonomic tone and metabolic rate. Selected for Cardiovascular coverage.

### Variables primarily associated with metabolic and lipid function in the planning hypothesis

- `LBXGLU` — Fasting glucose (not serum glucose `LBXSGL`) is the standard metabolic marker. Selected for Glucose Metabolism (System 3).
- `LBXSUA` (Uric Acid) — An inflammatory and metabolic marker with relevance to gout, kidney function, and cardiovascular risk. Selected for Renal Function (System 6).
- `LBXSATSI` (ALT) — A liver enzyme reflecting hepatic metabolic activity. Selected for Hepatic Function (System 5).
- `LBDHDD` (HDL) — Anti-atherogenic cholesterol, relevant to lipid metabolism efficiency. Selected for Lipid Metabolism (System 4).

### Variables primarily associated with body composition / metabolic in the planning hypothesis

- `BMXWAIST` — Waist circumference captures central adiposity independently of overall BMI. Selected for Body Composition (System 1).
- `DXDTOBMD` — Total body bone mineral density from DEXA. Selected for Body Composition (System 1) and its role in distinguishing lean vs. dense phenotypes.
- `DXDTOPF` (% Body Fat) — Normalised fat percentage from DEXA. Selected for Body Composition (System 1).
- `LBXIN` (Fasting Insulin) — Reflects pancreatic function and insulin sensitivity. Selected for Glucose Metabolism (System 3).
- `LBXSTR` (Triglycerides) — Key lipid metabolism marker. Selected for Lipid Metabolism (System 4).

---

## 4. Derived Features (computed, not in raw CSV)

These variables are computed from raw measurements during preprocessing (Decision 006). They are included in `preprocessed.csv` for post-hoc interpretation but are explicitly excluded from the clustering feature matrices to avoid collinearity with their constituent biomarkers.

| Derived Feature | Formula | Biological meaning |
|----------------|---------|-------------------|
| HOMA-IR | `(LBXGLU * LBXIN) / 405` | Homeostatic model assessment of insulin resistance |
| TC/HDL ratio | `LBXTC / LBDHDD` | Total cholesterol to HDL ratio — cardiovascular risk index |
| TG/HDL ratio | `LBXSTR / LBDHDD` | Triglyceride to HDL ratio — metabolic syndrome proxy |

---

## 5. Variables Excluded and Why

| Variable | Reason to Exclude |
|----------|------------------|
| `BMXWT`, `BMXHT` | Redundant with BMI |
| `BMXHIP`, `BMXARMC` | Lower information value; waist circumference is more specific for central adiposity |
| `DXDTOBMC` | Highly correlated with `DXDTOBMD`; only one retained |
| `DXDTOFAT` (raw fat grams) | `DXDTOPF` (% fat) is preferred as it is normalised for body size |
| `DXDTOTOT` | Mathematical sum of lean + fat — completely redundant with its components |
| All regional BMD/fat variables | Use only total measures to avoid collinearity from regional sub-components |
| `LBXSGL` (non-fasting glucose in BIOPRO_J) | `LBXGLU` (fasting glucose in GLU_J) is used — fasting values are more diagnostically reliable |
| `LBXSCH` / `LBDSCHSI` in BIOPRO_J | Duplicate of `LBXTC` from TCHOL_J — not included to avoid double-counting |
| All `*SI` and `*LC` suffix columns | Unit conversions of primary variables — one unit per variable |
| `WTSAF2YR` | Survey weight for population-level inference, not an individual biomarker |
| `DMDEDUC2`, `INDHHIN2`, `INDFMPIR` | Socioeconomic confounders — kept for sensitivity analysis, not clustering features |
| `HOMA_IR`, `TC_HDL_ratio`, `TG_HDL_ratio` | Derived ratios — excluded from clustering matrices to prevent collinearity with constituent biomarkers |

---

## 6. Notes on the NHANES_Variable_Dictionary.docx

**Overall verdict: mostly correct, with several important caveats noted at the time.**

### Confirmed correct
- All key variable codes exist in the actual CSV files.
- File descriptions and participant counts are accurate.
- The merge strategy (left join on SEQN anchored from DEMO_J) is correct.

### Errors and issues found

| Issue | Details | Severity |
|-------|---------|----------|
| `LBXSCH` listed as Total Cholesterol in BIOPRO_J | This is the same measurement as `LBXTC` in TCHOL_J. Using both would be duplication. | Medium |
| `BPXSY1` null rate at 28% | Reflects children being included. Filtering to adults 20+ reduces this significantly. | Clarification |
| `DXAHEBV` column | Present in DXX_J but not mentioned in the dictionary. DEXA validity flags should be checked via `DXAEXSTS` before using DXX_J rows. | Missing context |
| `LBDINLC` in INS_J | Not mentioned in dictionary. This below-detection-limit flag for insulin is important for data cleaning (Decision 005). | Missing — important |

### Variables not in the dictionary that were added to the final feature set

| Variable | File | Why added |
|----------|------|-----------
| `LBXSCA` (Calcium) | `BIOPRO_J` | Electrolyte/mineral system; only 8% nulls |
| `LBXSPH` (Phosphorus) | `BIOPRO_J` | Electrolyte/mineral system |
| `LBXSNASI` (Sodium) | `BIOPRO_J` | Electrolyte/mineral system |
| `LBXSKSI` (Potassium) | `BIOPRO_J` | Electrolyte/mineral system |
| `LBXSTP` (Total Protein) | `BIOPRO_J` | Hepatic function |
| `LBXSTB` (Total Bilirubin) | `BIOPRO_J` | Hepatic function |
| `LBXSBU` (BUN) | `BIOPRO_J` | Renal function; included in final set despite earlier exclusion note |

---

## 7. Final Merge Strategy

The actual merge is implemented in `pipeline/merge.py` and configured via `pipeline/config.py`. The code below shows the conceptual structure:

```python
import pandas as pd

# Load all files
demo  = pd.read_csv('DEMO_J.csv')
bmx   = pd.read_csv('BMX_J.csv')
bpx   = pd.read_csv('BPX_J.csv')
dxx   = pd.read_csv('DXX_J.csv')
glu   = pd.read_csv('GLU_J.csv')
ins   = pd.read_csv('INS_J.csv')
hdl   = pd.read_csv('HDL_J.csv')
tchol = pd.read_csv('TCHOL_J.csv')
bio   = pd.read_csv('BIOPRO_J.csv')

# Select only the columns needed
demo_cols  = ['SEQN', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3']
bmx_cols   = ['SEQN', 'BMXBMI', 'BMXWAIST']
bpx_cols   = ['SEQN', 'BPXSY1', 'BPXDI1', 'BPXPLS', 'BPXSY2', 'BPXSY3']
dxx_cols   = ['SEQN', 'DXDTOBMD', 'DXDTOPF', 'DXDTOLE']
glu_cols   = ['SEQN', 'LBXGLU']
ins_cols   = ['SEQN', 'LBXIN']
hdl_cols   = ['SEQN', 'LBDHDD']
tchol_cols = ['SEQN', 'LBXTC']
bio_cols   = ['SEQN', 'LBXSTR', 'LBXSAL', 'LBXSCR', 'LBXSUA', 'LBXSATSI']

# Merge all on SEQN (left join from DEMO)
df = demo[demo_cols]
for sub, cols in [(bmx, bmx_cols), (bpx, bpx_cols), (dxx, dxx_cols),
                  (glu, glu_cols), (ins, ins_cols), (hdl, hdl_cols),
                  (tchol, tchol_cols), (bio, bio_cols)]:
    df = df.merge(sub[cols], on='SEQN', how='left')

# Filter: adults 20-80 only
df = df[df['RIDAGEYR'] >= 20]

# Add averaged BP (CDC NHANES protocol — see Decision 002)
# Actual implementation in pipeline/preprocess/bp_averaging.py

# Add derived features (audit/interpretation only — excluded from clustering matrices)
df['HOMA_IR'] = (df['LBXGLU'] * df['LBXIN']) / 405
df['TG_HDL_ratio'] = df['LBXSTR'] / df['LBDHDD']
df['Chol_ratio'] = df['LBXTC'] / df['LBDHDD']
```

---

## 8. Summary

| Category | Count |
|----------|-------|
| Total raw variables in feature matrices | 19 (Cohort A) / 24 (Cohort B) |
| Derived variables computed (audit/interpretation only) | 3 (HOMA_IR, TC_HDL_ratio, TG_HDL_ratio) |
| Files used | 9 (all available NHANES 2017-2018 files) |
| Final sample — Cohort A (broad, 19 features) | 4,482 participants |
| Final sample — Cohort B (fasting/DEXA, 24 features) | 967 participants |

The K-selection question (how many clusters to use) was addressed through a cross-model benchmark in Experiment 6, documented in `docs/methodology_decision_log.md` under Decision 014. No specific K was predetermined or forced. No Dosha assignments have been made; downstream characterisation (Stage 8) has not yet been performed.
