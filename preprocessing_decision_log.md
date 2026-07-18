# Data Preprocessing Decision Log
## Research 2: The Bioinformatics Bridge
### NHANES 2017–2018 · Unsupervised Clustering Pipeline

---

> **Maintenance Rules**
> - Do NOT delete existing entries.
> - If a decision changes later, add a NEW entry referencing the original Decision Number.
> - Each entry must be self-contained: a researcher should be able to reproduce the step without reading the code.
> - Status values: `Proposed` | `Implemented` | `Modified` | `Removed`

---

| # | Step | Pipeline Stage | Status |
|---|------|---------------|--------|
| [001](#decision-001) | Variable Selection by Biological System | Merge | ✅ Implemented |
| [002](#decision-002) | Blood Pressure Averaging | Preprocess | ✅ Implemented |
| [003](#decision-003) | Participant Age Filter | Preprocess | 🔵 Proposed |
| [004](#decision-004) | DEXA Scan Validity Filter | Preprocess | 🔵 Proposed |
| [005](#decision-005) | Insulin Below-Detection-Limit Handling | Preprocess | 🔵 Proposed |
| [006](#decision-006) | Derived Feature Engineering | Preprocess | 🔵 Proposed |

---

## Decision 001

**Variable Selection by Biological System**

---

**Preprocessing Step**
Select which NHANES variables to include in the feature matrix before any clustering is performed.

**Why It Was Needed**
The dataset contains hundreds of variables across 9 CSV files. Only a curated subset can be meaningfully used in unsupervised clustering. A principled selection criterion is required so that the choice of variables is scientifically defensible and not circular.

**Biological Reasoning**
Human physiology is organised into distinct but interacting biological systems. A comprehensive metabolic phenotype requires measurement across all major systems rather than any single domain. The seven systems represented are:

| System | Biological Role |
|--------|----------------|
| Body Composition | Structural makeup: fat mass, lean mass, bone density, adiposity distribution |
| Cardiovascular | Circulatory function: arterial pressure, cardiac output at rest |
| Glucose Metabolism | Insulin-glucose axis: pancreatic function, insulin sensitivity |
| Lipid Metabolism | Fat transport: cholesterol, triglycerides, cardiovascular risk |
| Hepatic Function | Metabolic processing: liver enzymes, protein synthesis, detoxification |
| Renal Function | Waste filtration: kidney clearance rate, nitrogenous waste products |
| Electrolyte/Mineral Balance | Cellular signalling: calcium, phosphorus, sodium, potassium |

**Statistical Reasoning**
Variables were NOT selected because they resemble expected Dosha patterns. Selecting features to match a pre-defined hypothesis produces circular reasoning: the clustering will trivially recover the expected groups because the input features were pre-loaded to distinguish them. The correct scientific workflow is:

```
Select by comprehensive system coverage
         ↓
Run unsupervised clustering
         ↓
Interpret cluster profiles
         ↓
Observe whether profiles resemble Dosha patterns
```

This approach allows the data to generate its own groupings, which is the methodological requirement for a valid unsupervised result.

**Implementation Details**
- File: `pipeline/config.py`
- Variables are declared in `SELECTED_VARIABLES` dictionary, keyed by logical dataset name
- SEQN (participant ID) is added automatically by `load_csv()` — it is not listed in the variable definitions
- A runtime assertion ensures `FILE_MAP`, `SELECTED_VARIABLES`, and `MERGE_ORDER` are always in sync
- Adding or removing a variable requires editing only `config.py`

**Final Selected Variables (26 total)**

| Variable | Description | System | Source File |
|----------|-------------|--------|-------------|
| BMXBMI | Body Mass Index (kg/m²) | Body Composition | BMX_J |
| BMXWAIST | Waist circumference (cm) | Body Composition | BMX_J |
| DXDTOBMD | Total Body Bone Mineral Density (g/cm²) | Body Composition | DXX_J |
| DXDTOPF | Total Body Percent Fat (%) | Body Composition | DXX_J |
| DXDTOLE | Total Lean Mass excl. bone (g) | Body Composition | DXX_J |
| BPXSY1 | Systolic BP — 1st reading (mm Hg) | Cardiovascular | BPX_J |
| BPXSY2 | Systolic BP — 2nd reading (mm Hg) | Cardiovascular | BPX_J |
| BPXSY3 | Systolic BP — 3rd reading (mm Hg) | Cardiovascular | BPX_J |
| BPXDI1 | Diastolic BP — 1st reading (mm Hg) | Cardiovascular | BPX_J |
| BPXDI2 | Diastolic BP — 2nd reading (mm Hg) | Cardiovascular | BPX_J |
| BPXDI3 | Diastolic BP — 3rd reading (mm Hg) | Cardiovascular | BPX_J |
| BPXPLS | Resting pulse / heart rate (beats/min) | Cardiovascular | BPX_J |
| LBXGLU | Fasting Plasma Glucose (mg/dL) | Glucose Metabolism | GLU_J |
| LBXIN | Fasting Insulin (uU/mL) | Glucose Metabolism | INS_J |
| LBXTC | Total Cholesterol (mg/dL) | Lipid Metabolism | TCHOL_J |
| LBDHDD | HDL Cholesterol (mg/dL) | Lipid Metabolism | HDL_J |
| LBXSTR | Triglycerides (mg/dL) | Lipid Metabolism | BIOPRO_J |
| LBXSATSI | ALT — liver enzyme (U/L) | Hepatic Function | BIOPRO_J |
| LBXSAL | Albumin (g/dL) | Hepatic Function | BIOPRO_J |
| LBXSTP | Total Protein (g/dL) | Hepatic Function | BIOPRO_J |
| LBXSTB | Total Bilirubin (mg/dL) | Hepatic Function | BIOPRO_J |
| LBXSCR | Creatinine (mg/dL) | Renal Function | BIOPRO_J |
| LBXSUA | Uric Acid (mg/dL) | Renal Function | BIOPRO_J |
| LBXSBU | Blood Urea Nitrogen (mg/dL) | Renal Function | BIOPRO_J |
| LBXSCA | Total Calcium (mg/dL) | Electrolyte/Mineral | BIOPRO_J |
| LBXSPH | Phosphorus (mg/dL) | Electrolyte/Mineral | BIOPRO_J |
| LBXSNASI | Sodium (mmol/L) | Electrolyte/Mineral | BIOPRO_J |
| LBXSKSI | Potassium (mmol/L) | Electrolyte/Mineral | BIOPRO_J |

**Auxiliary variables retained for cleaning/validation (NOT in feature matrix)**

| Variable | Purpose |
|----------|---------|
| SEQN | Participant ID — merge key |
| RIAGENDR | Gender — for post-hoc confounder analysis |
| RIDAGEYR | Age — for filtering (see Decision 003) |
| RIDRETH3 | Race/ethnicity — for confounder analysis |
| DXAEXSTS | DEXA scan validity flag — for filtering (see Decision 004) |
| LBDINLC | Insulin below-detection flag — for filtering (see Decision 005) |

**Alternative Methods Considered**

| Alternative | Why Rejected |
|-------------|-------------|
| Select variables as Dosha proxies | Circular reasoning — pre-loading expected outcomes into features invalidates unsupervised analysis |
| Include all available variables | Curse of dimensionality; highly correlated regional DEXA variables would dominate PCA artificially |
| Use only lipid + glucose panel | Too narrow; misses body composition and kidney/liver context |
| Use factor analysis to select variables | Requires large complete-case sample first; not appropriate for the selection stage |

**Advantages**
- Scientifically defensible: variable choice is independent of the expected outcome
- Reproducible: the full variable list is in one location (`config.py`)
- Modifiable: adding or removing variables requires editing one dictionary

**Disadvantages**
- Excludes variables that may individually be strong Dosha discriminators (acceptable trade-off for methodological integrity)
- Requires domain knowledge of biological systems to construct the categories

**Effect on Sample Size**
No direct effect — this decision determines columns, not rows.

**Effect on Variables**
Reduced from hundreds of available NHANES columns to 28 selected + 6 auxiliary = 34 total columns in `merged_raw.csv`.

**Reproducibility Notes**
- Configuration file: `pipeline/config.py`
- Run `python run_pipeline.py` to regenerate `output/merged_raw.csv`
- Run `python run_pipeline.py --explore` to see all available columns in every raw CSV

**Pipeline Stage**
Merge (`run_pipeline.py` → `pipeline/merge.py`)

**Status**
✅ Implemented — `output/merged_raw.csv` generated with 9,254 rows × 34 columns

---

## Decision 002

**Blood Pressure Averaging**

---

**Preprocessing Step**
Convert three repeated systolic and diastolic blood pressure readings (BPXSY1, BPXSY2, BPXSY3, BPXDI1, BPXDI2, BPXDI3) into a single representative value per participant per direction.

New columns created:
- `Avg_Systolic_BP` = nanmean(BPXSY1, BPXSY2, BPXSY3)
- `Avg_Diastolic_BP` = nanmean(BPXDI1, BPXDI2, BPXDI3)

**Why It Was Needed**
NHANES takes up to three blood pressure readings at a single examination. Including all three raw readings as separate features would introduce artificial multicollinearity into the feature matrix — BPXSY1, BPXSY2, and BPXSY3 measure the same underlying physiological quantity and are highly correlated. The clustering algorithm would over-weight blood pressure simply because it is represented by three near-duplicate columns. A single averaged value eliminates this artefact.

**Biological Reasoning**
Blood pressure is not a fixed quantity — it fluctuates continuously with respiration, emotion, and physical state. A single reading samples one moment of this fluctuation. Three readings, taken a few minutes apart, capture a short window of that fluctuation. Their mean is a more representative estimate of the participant's resting blood pressure than any individual reading.

The first reading is systematically elevated relative to subsequent readings in clinical examination settings. This is well-documented as the white-coat effect: participants experience mild stress at the start of a medical examination, raising their blood pressure transiently. Readings two and three are taken after the participant has had time to acclimatise. Averaging dampens this transient artifact and produces a value that more closely approximates the participant's habitual blood pressure.

**Statistical Reasoning**
1. **Variance reduction**: For independent measurements with the same true mean μ and variance σ², the mean of n readings has variance σ²/n. Even if readings are not fully independent, averaging reduces measurement noise relative to any single reading.
2. **Bias reduction**: The first reading is upward-biased due to the white-coat effect. Averaging introduces readings taken at lower stress states, pulling the estimate toward the true resting level.
3. **Sample size preservation**: Using only BPXSY1 as the BP variable loses all participants where the first reading is missing but a later reading is available. The nanmean approach retains those participants.
4. **Alignment with clinical standards**: AHA guidelines and JNC-7 both specify that the average of two or more readings at a single visit should be used for clinical decision-making. NHANES analytic guidelines recommend using the average of available readings.

**Implementation Details**
- File: `pipeline/preprocess/bp_averaging.py`
- Function: `compute_bp_averages(df)` — accepts a DataFrame, returns a new DataFrame with two new columns appended; input is never mutated
- Missing value handling: `pandas.DataFrame.mean(axis=1, skipna=True)` — ignores NaN in the rowwise mean. If ALL three readings are NaN for a participant, the result is NaN (not zero).
- Original columns BPXSY1, BPXSY2, BPXSY3, BPXDI1, BPXDI2, BPXDI3 are preserved in the output file
- Run: `python run_preprocess.py` → reads `output/merged_raw.csv` → saves `output/preprocessed.csv`

**Measured Results from Live Run**

| Metric | Systolic | Diastolic |
|--------|---------|---------|
| Participants: 1st reading only valid | 6,302 | 6,302 |
| Participants: average valid | **6,714** | **6,714** |
| Coverage gain | **+412** | **+412** |
| Missing % before | 31.9% | 31.9% |
| Missing % after | **27.4%** | **27.4%** |
| Mean (average) | 121.7 mm Hg | 68.3 mm Hg |
| 3 readings used | 6,077 participants | 6,077 participants |
| 2 readings used | 535 participants | 535 participants |
| 1 reading used | 102 participants | 102 participants |
| All missing → NaN | 2,540 participants | 2,540 participants |

**Alternative Methods Considered**

| Alternative | Why Rejected |
|-------------|-------------|
| Use only first reading (BPXSY1/BPXDI1) | Upward-biased (white-coat effect); loses 412 participants with valid later readings |
| Use only the last available reading | No clinical or statistical justification; introduces different missingness pattern |
| Use readings 2 and 3 only (skip first) | Some participants only have reading 1; this would lose more data than averaging all |
| Use the median of available readings | With only 3 measurements, the median equals the middle value exactly — provides no statistical advantage over the mean and is less commonly used in NHANES literature |
| Impute missing readings before averaging | Imputation before the averaging step would compound uncertainty; NHANES guidelines recommend averaging available readings only |

**Advantages**
- Reduces measurement error and white-coat bias
- Recovers 412 participants who would have been lost using first-reading-only
- Reduces missing rate from 31.9% to 27.4%
- Aligns with AHA, JNC-7, and NHANES analytic guidelines
- Preserves all original raw readings for full traceability

**Disadvantages**
- Mixing readings taken under different stress states (reading 1 is biased high); however, averaging is still preferable to using the biased first reading alone
- If a participant only had one reading because of an abnormal value (e.g., extremely high), that single reading becomes the average — not smoothed

**Effect on Sample Size**
No rows removed. 412 additional participants gain a valid `Avg_Systolic_BP` and `Avg_Diastolic_BP` value compared to using only the first reading.

**Effect on Variables**
Two new columns added: `Avg_Systolic_BP`, `Avg_Diastolic_BP`.
Six raw columns preserved: BPXSY1, BPXSY2, BPXSY3, BPXDI1, BPXDI2, BPXDI3.
Net: +2 columns. Output file: 9,254 rows × 36 columns.

**Reproducibility Notes**
- Module: `pipeline/preprocess/bp_averaging.py`
- Entry point: `python run_preprocess.py`
- Input: `output/merged_raw.csv`
- Output: `output/preprocessed.csv`
- The averaging constants (`SYSTOLIC_COLS`, `DIASTOLIC_COLS`, `AVG_SYSTOLIC_COL`, `AVG_DIASTOLIC_COL`) are defined at the top of `bp_averaging.py` so column names can be changed in one place

**Pipeline Stage**
Preprocess (`run_preprocess.py` → `pipeline/preprocess/bp_averaging.py`)

**Status**
✅ Implemented — `output/preprocessed.csv` generated with 9,254 rows × 36 columns

---

## Decision 003

**Participant Age Filter**

---

**Preprocessing Step**
Filter the dataset to retain only adult participants aged 20 to 80 years inclusive.

**Why It Was Needed**
NHANES covers the full age range from infancy through old age. The biological systems being measured respond differently across the lifespan. Children and adolescents have fundamentally different body composition profiles, blood pressure norms, lipid levels, and bone density trajectories than adults. Including them in a clustering intended to detect adult metabolic phenotypes would introduce age-driven clusters that reflect developmental stage rather than habitual phenotype.

**Biological Reasoning**
- **Children/adolescents**: BMI, bone density (DXDTOBMD), blood pressure, and lipid levels are all age-dependent in ways that do not reflect the adult metabolic variation the study is examining. The bone-density–based Kapha phenotype hypothesis, for example, is meaningless before peak bone mass is reached (~age 25–30).
- **Very elderly (80+)**: NHANES codes age as 80 for all participants 80 and older. These participants cannot be distinguished by exact age, and extreme age introduces sarcopenia, polypharmacy effects, and other confounders.
- **Lower bound 20**: Standard cutoff for adult NHANES analyses. Consistent with NIH, CDC, and most NHANES analytic papers.

**Statistical Reasoning**
- Blood pressure reference ranges are age-stratified; including children would create spurious low-BP clusters
- Blood pressure null rates drop significantly when age ≥ 20 (children often do not have full BP measurements)
- Keeping the analysis within a single developmental stage reduces confounder dimensionality
- Consistent with the NHANES analytic and reporting guidelines

**Implementation Details**
- Filter criterion: `RIDAGEYR >= 20` AND `RIDAGEYR <= 80`
- Applied after BP averaging (Decision 002)
- RIDAGEYR is already present in the merged file (retained as a control variable in Decision 001)
- No imputation or row modification — rows outside the age range are simply dropped

**Alternative Methods Considered**

| Alternative | Why Not Preferred |
|-------------|-----------------|
| Include all ages | Introduces developmental confounders; blood pressure and BMI in children are non-comparable to adults |
| Age ≥ 18 | Common legal-adult cutoff but NHANES analytic convention uses 20; some measurements (DEXA) are also adult-only |
| Stratify by age group and cluster separately | Valid approach but increases complexity; proposed as a sensitivity analysis in the paper, not the primary pipeline |

**Advantages**
- Removes developmental confounders
- Reduces blood pressure missingness (children often lack full BP data)
- Consistent with published NHANES analytic practice

**Disadvantages**
- Reduces sample size; estimated loss of ~3,400 rows (those aged 0–19)
- Excludes potentially interesting adolescent metabolic phenotypes

**Effect on Sample Size**
Estimated: 9,254 rows → ~5,800 rows (loss of ~3,454 participants under 20 or over 80)
*Exact number to be confirmed when implemented.*

**Effect on Variables**
None — only rows are filtered, no columns added or removed.

**Reproducibility Notes**
- Filter expression: `df = df[(df['RIDAGEYR'] >= 20) & (df['RIDAGEYR'] <= 80)]`
- `RIDAGEYR` must be retained in the dataset (already included as per Decision 001)

**Pipeline Stage**
Preprocess (next step after Decision 002)

**Status**
🔵 Proposed — not yet implemented

---

## Decision 004

**DEXA Scan Validity Filter**

---

**Preprocessing Step**
Retain only DEXA scan records where the examination status flag `DXAEXSTS` equals 1 (valid, complete scan).

**Why It Was Needed**
The DXX_J dataset contains a flag variable `DXAEXSTS` indicating whether the DEXA scan was successfully completed. Participants with status codes other than 1 have partial, failed, or invalid scans. Using bone density or body fat values from invalid scans would introduce measurement error into the body composition features.

**Biological Reasoning**
An incomplete DEXA scan does not measure the full body. A partial scan will underestimate total bone mineral density (`DXDTOBMD`), total lean mass (`DXDTOLE`), and total body fat percentage (`DXDTOPF`). Including these values as if they represent whole-body measurements is biologically incorrect.

**Statistical Reasoning**
Partial-scan values are not missing at random — they are missing because the scan could not be completed (e.g., participant was too large for the scanner, implant interference, pregnancy). Using these values would introduce systematic downward bias in body composition estimates for a specific subset of the population.

**Implementation Details**
- Filter criterion: set DEXA variables to NaN for rows where `DXAEXSTS != 1`
- Variables affected: `DXDTOBMD`, `DXDTOPF`, `DXDTOLE`
- `DXAEXSTS` column is retained in the dataset for reference
- This is a column-level NaN assignment, NOT row deletion: the participant remains in the dataset with other valid measurements

**DXAEXSTS codes in NHANES**

| Code | Meaning |
|------|---------|
| 1 | Whole body scan completed |
| 2 | Whole body scan completed, but invalid data |
| 3 | Whole body scan not completed, fat tissue invalid |
| 4 | Whole body scan not completed, lean tissue invalid |
| 5 | Whole body scan not completed, bone tissue invalid |
| 6 | Whole body scan not completed, all invalid |

**Alternative Methods Considered**

| Alternative | Why Not Preferred |
|-------------|-----------------|
| Drop entire rows where DXAEXSTS != 1 | Wastes valid non-DEXA measurements for the same participant |
| Keep all DEXA values regardless of flag | Uses biologically invalid partial measurements |
| Impute DEXA values for invalid scans | Imputing bone density from demographics would lose the very signal being studied |

**Advantages**
- Eliminates systematically biased DEXA measurements
- Preserves participants for other biological system measurements
- Follows NHANES analytic guidelines on DEXA data quality

**Disadvantages**
- Increases missingness in DEXA variables beyond the current 60%
- Exact number of additional NaNs depends on how many participants have DXAEXSTS != 1

**Effect on Sample Size**
Rows: no change (row-level operation is NOT performed)
Variables: DXDTOBMD, DXDTOPF, DXDTOLE set to NaN for invalid-scan participants

**Effect on Variables**
`DXDTOBMD`, `DXDTOPF`, `DXDTOLE` — values replaced with NaN for non-valid scans

**Reproducibility Notes**
- `DXAEXSTS` must be retained in the dataset (already included as per Decision 001)
- Filter expression: `df.loc[df['DXAEXSTS'] != 1, ['DXDTOBMD','DXDTOPF','DXDTOLE']] = np.nan`

**Pipeline Stage**
Preprocess (after Decision 003)

**Status**
🔵 Proposed — not yet implemented

---

## Decision 005

**Insulin Below-Detection-Limit Handling**

---

**Preprocessing Step**
Set `LBXIN` (Fasting Insulin) to NaN for participants where the value was flagged as below the laboratory's minimum detection limit (`LBDINLC == 1`).

**Why It Was Needed**
NHANES laboratory data includes a detection-limit flag (`LBDINLC`) that indicates whether a measured value fell below the instrument's reliable detection threshold. When `LBDINLC == 1`, the reported insulin value is not a real measurement — it is an imputed fill value (typically the detection limit divided by the square root of 2, per CDC convention). Using this as a true insulin measurement would introduce a systematic artefact in the lower tail of the insulin distribution.

**Biological Reasoning**
Very low fasting insulin values (below the detection limit) are physiologically rare and are typically observed in individuals with Type 1 diabetes or insulin deficiency states. The specific numerical value assigned by the laboratory is an artefact of the imputation convention, not the participant's true insulin level. Treating this value as a genuine measurement could distort the low-insulin cluster profile.

**Statistical Reasoning**
- Detection-limit imputation creates a point mass at one specific value in the lower tail, artificially inflating frequency at that point
- This artefact would be amplified by HOMA-IR computation (Glucose × Insulin), propagating the error into a derived feature
- Setting to NaN is the standard NHANES analytic recommendation for below-detection-limit values

**Implementation Details**
- Condition: `LBDINLC == 1`
- Action: set `LBXIN` to `np.nan` for those rows
- `LBDINLC` column is retained in the dataset for reference
- Fasting Insulin has approximately 7% nulls before this step; this step will increase that slightly

**Alternative Methods Considered**

| Alternative | Why Not Preferred |
|-------------|-----------------|
| Keep the imputed value as-is | Introduces a non-physiological point mass at detection limit / sqrt(2) |
| Substitute with zero | Physiologically incorrect; zero insulin is not possible in living participants |
| Impute from other variables | Detection-limit values are not missing at random — they represent the lowest quantile; standard imputation methods will not recover the true undetected level |

**Advantages**
- Removes laboratory artefact from the insulin distribution
- Prevents artefact propagation into derived features (HOMA-IR)
- Consistent with CDC and NHANES analytic guidelines for laboratory data

**Disadvantages**
- Slightly increases missingness in `LBXIN`
- Removes participants at the extreme low end of insulin distribution; if very-low-insulin is clinically important, this information is lost

**Effect on Sample Size**
Rows: no change
Variables: `LBXIN` set to NaN for rows where `LBDINLC == 1`

**Effect on Variables**
`LBXIN` — small number of values replaced with NaN

**Reproducibility Notes**
- `LBDINLC` must be retained in the dataset (already included as per Decision 001)
- Filter expression: `df.loc[df['LBDINLC'] == 1, 'LBXIN'] = np.nan`

**Pipeline Stage**
Preprocess (after Decision 004)

**Status**
🔵 Proposed — not yet implemented

---

## Decision 006

**Derived Feature Engineering**

---

**Preprocessing Step**
Compute three new variables from existing columns:

| New Variable | Formula | Biological Meaning |
|-------------|---------|-------------------|
| `HOMA_IR` | `(LBXGLU × LBXIN) / 405` | Homeostatic Model Assessment of Insulin Resistance |
| `TC_HDL_ratio` | `LBXTC / LBDHDD` | Total Cholesterol to HDL ratio — cardiovascular risk index |
| `TG_HDL_ratio` | `LBXSTR / LBDHDD` | Triglyceride to HDL ratio — metabolic syndrome proxy |

**Why It Was Needed**
The three raw variables (glucose, insulin, lipids) each carry partial information. Clinical biostatistics has established that certain ratio-derived indices carry more discriminatory information than any individual component. HOMA-IR in particular is the standard clinical index for quantifying insulin resistance and is more informative than either glucose or insulin alone.

**Biological Reasoning**
- **HOMA-IR**: The relationship between fasting glucose and fasting insulin is non-linear. A participant with moderately elevated glucose AND elevated insulin is insulin-resistant. A participant with low glucose and low insulin is not. Neither glucose nor insulin alone captures this — only their product (approximated by HOMA-IR) does. Derived by Matthews et al. (1985); validated in thousands of studies.
- **TC/HDL ratio**: Total cholesterol alone is a poor cardiovascular risk predictor. HDL is protective. The ratio captures the balance between atherogenic and anti-atherogenic cholesterol. Endorsed by AHA and used in Framingham risk scoring.
- **TG/HDL ratio**: A surrogate for LDL particle size and metabolic syndrome. High TG combined with low HDL indicates insulin resistance and dyslipidaemia. A TG/HDL > 3.0 is used as a clinical screening criterion.

**Statistical Reasoning**
- Ratio features may have stronger separation between metabolic phenotype clusters than the raw components
- They reduce the effective dimensionality of the lipid and metabolic systems (three ratios replace or complement six raw variables)
- Must be computed AFTER filtering (Decisions 003–005) so that biologically invalid values do not propagate into the derived features

**Implementation Details**
- Computed after: Decision 002 (BP averaging), Decision 003 (age filter), Decision 004 (DEXA validity), Decision 005 (insulin detection limit)
- If any component is NaN, the derived feature is NaN (standard pandas arithmetic propagation)
- No participants are dropped due to missing derived features at this stage

**Alternative Methods Considered**

| Alternative | Why Not Preferred |
|-------------|-----------------|
| Use raw glucose + raw insulin as separate features | Loses the interaction information captured by HOMA-IR |
| Use eGFR instead of raw creatinine | eGFR requires age + sex in the formula — adding demographic confounders directly into a feature is acceptable but must be noted; deferred to sensitivity analysis |
| Compute LDL via Friedewald equation | Friedewald equation is inaccurate at very high triglycerides (>400 mg/dL); NHANES does not directly measure LDL; deferred |

**Advantages**
- Adds clinically validated composite indices that carry more information than raw components
- HOMA-IR directly quantifies the insulin resistance axis — a key biological dimension expected to vary across metabolic phenotypes
- Ratio features are more robust to scale differences than raw values

**Disadvantages**
- Ratios introduce non-linearity and potential heteroscedasticity
- If a denominator is near zero, the ratio can become extreme; HDL < 10 mg/dL would produce unstable TC/HDL and TG/HDL — such outliers must be inspected before clustering

**Effect on Sample Size**
No rows removed. Participants with NaN in any component will have NaN in the derived feature.

**Effect on Variables**
Three new columns added: `HOMA_IR`, `TC_HDL_ratio`, `TG_HDL_ratio`.

**Reproducibility Notes**
- Compute order: HOMA_IR first, then ratios
- Formulas:
  - `HOMA_IR = (LBXGLU * LBXIN) / 405`
  - `TC_HDL_ratio = LBXTC / LBDHDD`
  - `TG_HDL_ratio = LBXSTR / LBDHDD`
- Must be applied AFTER Decisions 003–005 to avoid propagating invalid values

**Pipeline Stage**
Preprocess (after Decision 005)

**Status**
🔵 Proposed — not yet implemented

---

*Last updated: 2026-06-25 · Research 2: The Bioinformatics Bridge · Naya Velvyn · Liana Labs*
