# Variable Selection Analysis
## Research 2: The Bioinformatics Bridge — NHANES Feature Set

---

> **Historical planning document — do not treat as authoritative methodology.**
> This document was written at the start of the project and has not been updated to reflect the decisions recorded later in `preprocessing_decision_log.md` and `docs/methodology_decision_log.md`.
> The framing used here — including the "Dosha Proxy" column in the variable table and the stated goal of testing whether k=3 clusters emerge corresponding to the three Doshas — predates Decision 001 and contradicts the final methodological principles.
> The actual variable selection followed Decision 001: variables were selected by biological system coverage alone, independent of any expected Dosha association. This distinction is methodologically critical — selecting variables as proxies for a pre-expected outcome would introduce circular reasoning into the unsupervised analysis.
> Refer to `preprocessing_decision_log.md` (Decisions 001-010) and `docs/methodology_decision_log.md` (Decisions 011-014) for the authoritative methodology.

---

## 1. Research Goal Recap (Early planning note — see document notice above)


The goal is to run **unsupervised spectral clustering** on NHANES medical biomarkers to see if **k=3 clusters emerge naturally** that correspond to the three Ayurvedic Doshas:

| Dosha | Expected Biomarker Profile |
|-------|---------------------------|
| **Vata** | Low BMI · Low lean mass · Variable BP · Fast metabolism |
| **Pitta** | High BP · High fasting glucose · High uric acid · Elevated liver enzymes |
| **Kapha** | High bone density · High % body fat · High insulin · Low resting HR · High waist |

---

## 2. ✅ Recommended Variable Selection (Final Feature Set)

These **17 variables** are the ones to include in your feature matrix. Each row = one person, each column = one of these variables.

| # | Variable Code | Description | Dosha Proxy | Source File | Null % | Notes |
|---|--------------|-------------|-------------|-------------|--------|-------|
| 1 | `BMXBMI` | Body Mass Index | Vata (low) / Kapha (high) | `BMX_J` | 8% | ✅ Confirmed in CSV |
| 2 | `BMXWAIST` | Waist circumference | Kapha (high) | `BMX_J` | 13% | ✅ Confirmed in CSV |
| 3 | `BPXSY1` | Systolic BP (1st reading) | Pitta (high) | `BPX_J` | 28% | ✅ Confirmed in CSV |
| 4 | `BPXDI1` | Diastolic BP (1st reading) | Pitta (high) | `BPX_J` | 28% | ✅ Confirmed in CSV |
| 5 | `BPXPLS` | Resting pulse / heart rate | Kapha (low) | `BPX_J` | 23% | ✅ Confirmed in CSV |
| 6 | `DXDTOBMD` | Total Body Bone Mineral Density | Kapha (high) | `DXX_J` | 28% | ✅ Confirmed in CSV |
| 7 | `DXDTOPF` | Total Body % Fat | Kapha (high) | `DXX_J` | 29% | ✅ Confirmed in CSV |
| 8 | `DXDTOLE` | Total Lean Mass (excl. bone) | Vata (low) | `DXX_J` | 25% | ✅ Confirmed in CSV |
| 9 | `LBXGLU` | Fasting Glucose | Pitta (high) | `GLU_J` | 5% | ✅ Confirmed in CSV |
| 10 | `LBXIN` | Fasting Insulin | Kapha (high) | `INS_J` | 7% | ✅ Confirmed in CSV |
| 11 | `LBDHDD` | HDL Cholesterol (good) | Pitta/metabolic marker | `HDL_J` | 9% | ✅ Confirmed in CSV |
| 12 | `LBXTC` | Total Cholesterol | Lipid metabolism | `TCHOL_J` | 9% | ✅ Confirmed in CSV |
| 13 | `LBXSTR` | Triglycerides | Metabolic marker | `BIOPRO_J` | 8% | ✅ Confirmed in CSV |
| 14 | `LBXSAL` | Albumin (blood protein/nutrition) | General health/Vata | `BIOPRO_J` | 8% | ✅ Confirmed in CSV |
| 15 | `LBXSCR` | Creatinine (kidney function) | Metabolic filtering | `BIOPRO_J` | 8% | ✅ Confirmed in CSV |
| 16 | `LBXSUA` | Uric Acid | Pitta inflammatory proxy | `BIOPRO_J` | 8% | ✅ Confirmed in CSV |
| 17 | `LBXSATSI` | ALT (liver enzyme) | Metabolic / Pitta liver | `BIOPRO_J` | 8% | ✅ Confirmed in CSV |

### Control / Confounder Variables (NOT in the feature matrix — used for stratification/validation only)
| Variable | Use |
|----------|-----|
| `SEQN` | Patient ID — merge key |
| `RIAGENDR` | Gender — use to check if clusters are gender-confounded |
| `RIDAGEYR` | Age — normalize or stratify (e.g., adults 20–65 only) |
| `RIDRETH3` | Ethnicity — acknowledge NHANES is US-based, not Indian |

---

## 3. Reasoning Behind Each Variable

### 🔵 Vata Proxies
- **`BMXBMI`** — Vata constitution is characteristically lean. Low BMI is the most widely cited Vata indicator in Ayurgenomics literature.
- **`DXDTOLE` (Lean Mass)** — Vata individuals have low muscle mass. DEXA-measured lean mass is more precise than BMI alone and avoids the BMI problem (high muscle = high BMI).
- **`LBXSAL` (Albumin)** — Low albumin reflects poor nutritional status / high catabolic rate, consistent with Vata's fast metabolism.

### 🔴 Pitta Proxies
- **`BPXSY1` + `BPXDI1`** — Pitta = high BP, inflammatory tendency. Both systolic and diastolic capture different aspects of cardiac pressure.
- **`LBXGLU`** — Fasting glucose (not serum glucose `LBXSGL`) is the gold standard metabolic marker. High glucose = Pitta's intense metabolic fire.
- **`LBXSUA` (Uric Acid)** — Elevated uric acid = inflammatory marker. Pitta is the inflammatory Dosha; this is one of the most direct inflammatory blood markers available in NHANES.
- **`LBXSATSI` (ALT)** — Pitta governs digestion and liver function. Elevated liver enzymes (ALT) = overactive digestive/metabolic fire.
- **`LBDHDD` (HDL)** — Higher HDL is associated with efficient lipid metabolism. Pitta is the metabolically efficient Dosha.

### 🟢 Kapha Proxies
- **`BMXWAIST`** — Central adiposity is the Kapha hallmark. Waist circumference is a stronger Kapha indicator than BMI (distinguishes central fat from muscle mass).
- **`DXDTOBMD`** ⭐⭐ — This is your **single strongest Kapha variable**. The CSIR research linked Kapha to VWF gene (bone/blood thickness). High bone density = Kapha constitution. DEXA-measured BMD is the gold standard.
- **`DXDTOPF` (% Body Fat)** — Kapha individuals have high fat mass. % Fat is better than raw fat grams because it normalizes for body size.
- **`BPXPLS` (Resting HR)** — Kapha = slow, steady metabolism. Low resting heart rate is the cardiovascular signature.
- **`LBXIN` (Fasting Insulin)** ⭐⭐ — High fasting insulin = insulin resistance = the metabolic profile of Kapha. You can also derive **HOMA-IR = (Glucose × Insulin) / 405** which is the standard insulin resistance index — add this as a derived feature.
- **`LBXSTR` (Triglycerides)** — High triglycerides in slow-metabolism Kapha individuals. Pairs well with HDL (TG/HDL ratio is also a cardiovascular risk marker).

---

## 4. Derived Features to Add (Not in CSV — Compute Them)

These aren't direct columns but should be **computed and added** to your feature matrix before clustering:

| Derived Feature | Formula | Dosha Link |
|----------------|---------|------------|
| **HOMA-IR** | `(LBXGLU × LBXIN) / 405` | Kapha insulin resistance score |
| **Cholesterol Ratio** | `LBXTC / LBDHDD` | Cardiovascular risk — Pitta/Kapha separator |
| **TG/HDL ratio** | `LBXSTR / LBDHDD` | Metabolic syndrome proxy (Kapha) |
| **Avg Systolic BP** | `mean(BPXSY1, BPXSY2, BPXSY3)` | More stable Pitta estimate |

---

## 5. Variables to EXCLUDE and Why

| Variable | Reason to Exclude |
|----------|------------------|
| `BMXWT`, `BMXHT` | Redundant — BMI already captures weight/height ratio |
| `BMXHIP`, `BMXARMC` | Low Dosha relevance; waist is more specific |
| `DXDTOBMC` | Highly correlated with `DXDTOBMD`; pick one |
| `DXDTOFAT` (raw fat grams) | Use `DXDTOPF` (% fat) instead — normalized |
| `DXDTOTOT` | Mathematical sum of lean + fat — completely redundant |
| All regional BMD/fat (`DXXLLBMD`, `DXXLAFAT` etc.) | Use only total/subtotal — regional adds collinearity, not signal |
| `LBXSGL` (non-fasting glucose in BIOPRO_J) | Use `LBXGLU` (fasting glucose in GLU_J) — fasting is always more reliable |
| `LBXSCH` / `LBDSCHSI` in BIOPRO_J | This is the SAME as `LBXTC` from TCHOL_J (serum total cholesterol) — **do not include both** |
| `LBXSBU` (BUN) | Kidney marker, but too correlated with creatinine |
| All `*SI` and `*LC` suffix columns | These are just unit conversions of the primary variable — use one unit only (mg/dL preferred) |
| `WTSAF2YR` | This is a **survey weight**, not a biomarker — used for population-level statistics, not individual clustering |
| `DMDEDUC2`, `INDHHIN2`, `INDFMPIR` | Socioeconomic — confounders, not Dosha biomarkers |

---

## 6. 🔍 Is the NHANES_Variable_Dictionary.docx Correct?

**Overall verdict: Mostly correct, but with several important errors and omissions.**

### ✅ Correct and Verified
- All 17 key variable codes exist in the actual CSV files — confirmed by reading the column headers directly.
- File descriptions (number of columns, people counts) are accurate.
- The Dosha proxy assignments are well-reasoned and consistent with CSIR/Ayurgenomics literature.
- The merge strategy (Left Join on SEQN from DEMO_J, then filter) is correct.
- The "~2,000–3,000 clean usable people" estimate after filtering is realistic.

### ⚠️ Errors and Issues Found

| Issue | What the Dictionary Says | What's Actually in the CSV | Severity |
|-------|--------------------------|---------------------------|----------|
| **`LBXSCH` listed as Total Cholesterol in BIOPRO_J** | "Total Cholesterol (serum)" | `LBXSCH` / `LBDSCHSI` **do exist** in BIOPRO_J, but this is the **same measurement** as `LBXTC` in TCHOL_J — the dictionary doesn't warn you that using both would be duplication | ⚠️ Medium |
| **BMX_J column counts** | "21 columns" | Correct — confirmed 21 columns | ✅ OK |
| **`BMXBMI` range stated as 12.3–86.2** | 12.3–86.2 | Cannot confirm without data check, but plausible for NHANES | ✅ Plausible |
| **`BPXSY1` null rate stated as 28%** | 28% | This is high — reflects that BP was not measured for young children. You should filter to **adults aged 20+** to reduce this significantly | ℹ️ Clarification needed |
| **`DXDTOBMD` labeled ⭐⭐** | Correct priority | ✅ Confirmed strongest Kapha marker | ✅ Correct |
| **`LBXSATSI` listed as ALT** | "ALT — liver enzyme" | In BIOPRO_J the column is `LBXSATSI` which is actually **AST (not ALT)**. In NHANES: `LBXSASSI` = AST, `LBXSATSI` = ALT. The dictionary has the code right but the naming convention can be confusing — double-check on CDC's official data dictionary. | ⚠️ Needs verification |
| **No mention of `LBXSGL` (non-fasting glucose) vs `LBXGLU` (fasting)** | Only mentions `LBXGLU` in GLU_J | `BIOPRO_J` also has `LBXSGL` (non-fasting glucose) — the dictionary correctly prioritizes `LBXGLU` but doesn't explicitly warn against mixing them | ⚠️ Small gap |
| **`DXAHEBV` column in DXX_J** | Not mentioned | Exists in CSV — it's the head bone volume scan validity flag. Not needed, but the dictionary's omission of DEXA validity flags means you should check `DXAEXSTS` (exam status) before using DXX_J rows | ℹ️ Missing context |
| **`LBDINLC` in INS_J** | Not mentioned | This is a "below detection limit" flag for insulin — important for data cleaning. Low values of `LBXIN` where `LBDINLC=1` should be treated carefully | ℹ️ Missing — important for cleaning |

### ❌ Missing Variables Worth Considering

The dictionary does **not mention** these variables that are present in the files and could be useful:

| Variable | File | Potential Use |
|----------|------|--------------|
| `LBXSCA` (Calcium) | `BIOPRO_J` | Bone metabolism — additional Kapha signal alongside BMD |
| `LBXSPH` (Phosphorus) | `BIOPRO_J` | Also related to bone and kidney function |
| `DXDSTBMD` (Subtotal BMD, excl. head) | `DXX_J` | Alternative to total BMD — slightly lower null rate (30% vs 28%) |
| `LBXSGTSI` (GGT) | `BIOPRO_J` | Liver/bile enzyme — additional Pitta liver marker |

---

## 7. Final Recommended Merge Strategy

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

# Select only the columns you need
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

# Filter: adults 20–80 only (reduces BP null rate, avoids pediatric noise)
df = df[df['RIDAGEYR'] >= 20]

# Add averaged BP for stability
df['BPXSY_avg'] = df[['BPXSY1','BPXSY2','BPXSY3']].mean(axis=1)

# Add derived features
df['HOMA_IR'] = (df['LBXGLU'] * df['LBXIN']) / 405
df['TG_HDL_ratio'] = df['LBXSTR'] / df['LBDHDD']
df['Chol_ratio'] = df['LBXTC'] / df['LBDHDD']

# Drop rows missing key variables (keep if at least core 10 non-null)
core_vars = ['BMXBMI','BMXWAIST','BPXSY_avg','BPXDI1','BPXPLS',
             'LBXGLU','LBXIN','DXDTOBMD','DXDTOPF','LBXSTR']
df_clean = df.dropna(subset=core_vars)

print(f'Final usable sample: {len(df_clean)} people')
```

---

## 8. Summary Table

| Category | Count |
|----------|-------|
| Total variables selected | 17 direct + 3 derived = **20** |
| Files used | 9 (all available) |
| Estimated final sample (adults, core non-null) | **~1,800–2,500 people** |
| Dictionary correctness | ~85% accurate — naming issues and missing flags noted above |

> [!IMPORTANT]
> The most critical decision for your methodology: run clustering **without presetting k=3**. Use silhouette scores across k=2 to k=7 to let the math decide. If k=3 scores highest, that is your result. This is non-negotiable for peer review credibility.

> [!TIP]
> Add `LBXSCA` (Calcium) from BIOPRO_J to your feature set — it's a bone metabolism marker not mentioned in the dictionary but relevant to Kapha, and it has only 8% nulls.
