# Bioinfo Bridge
### Mapping Ayurvedic Doshas to Human Biology Using Unsupervised Machine Learning
*Research 2 · 2026*

---

## Overview

This repository contains the full data pipeline for an exploratory bioinformatics study that applies **unsupervised spectral clustering** to NHANES 2017–2018 health survey data.

The research question:

> *Do human biomarkers measured across major biological systems naturally cluster into groups that align with the three Ayurvedic body types — Vata, Pitta, and Kapha — without being told to look for them?*

The pipeline is intentionally built so that **variable selection is independent of the expected outcome**. Features are chosen because they represent major biological systems (body composition, cardiovascular, metabolic, lipid, hepatic, renal, electrolyte) — not because they resemble Dosha descriptions. The clustering algorithm then finds whatever natural groupings exist. Only after clusters emerge do we ask whether they resemble Dosha profiles.

---

## Repository Structure

```
bioinfo-bridge/
│
├── pipeline/                        # Python package — all core logic
│   ├── __init__.py
│   ├── config.py                    # Single source of truth: variables, paths, merge order
│   ├── merge.py                     # Merge orchestration (Phase 1)
│   ├── utils.py                     # Shared helper functions
│   └── preprocess/                  # Preprocessing sub-package (Phase 2)
│       ├── __init__.py
│       ├── bp_averaging.py          # Decision 002: Blood pressure averaging
│       ├── filters.py               # Decisions 003–005: Age, DEXA validity, insulin LOD
│       └── derived_features.py      # Decision 006: HOMA-IR, TC/HDL, TG/HDL ratios
│
├── docs/                            # Research documentation
│   ├── preprocessing_decision_log.md  # Living log of every preprocessing decision
│   └── variable_selection_rationale.md
│
├── run_pipeline.py                  # Entry point: merge all NHANES CSVs → merged_raw.csv
├── run_preprocess.py                # Entry point: apply all preprocessing → preprocessed.csv
├── test.py                          # Quick data inspection script
├── requirements.txt                 # Python dependencies
└── .gitignore
```

> **Data files are not committed to this repository.**
> NHANES data is freely available from the CDC — see [Data Setup](#data-setup) below.

---

## Biological Systems Covered

Variables were selected to give comprehensive coverage of **7 major biological systems**:

| System | Variables |
|--------|-----------|
| Body Composition | BMI, Waist circumference, Bone mineral density, % Body fat, Lean mass |
| Cardiovascular | Systolic BP (avg), Diastolic BP (avg), Resting heart rate |
| Glucose Metabolism | Fasting glucose, Fasting insulin, HOMA-IR |
| Lipid Metabolism | Total cholesterol, HDL, Triglycerides, TC/HDL ratio, TG/HDL ratio |
| Hepatic Function | ALT, Albumin, Total protein, Bilirubin |
| Renal Function | Creatinine, Uric acid, BUN |
| Electrolyte/Mineral | Calcium, Phosphorus, Sodium, Potassium |

---

## Pipeline Stages

### Stage 1 — Merge (`run_pipeline.py`)

Reads 9 NHANES CSV files, selects only the variables listed in `config.py`, validates SEQN uniqueness, and merges everything on SEQN using a left join anchored to the demographics file.

```
Input  : DATASET/*.csv  (9 files)
Output : output/merged_raw.csv  (9,254 rows × 34 columns)
```

### Stage 2 — Preprocess (`run_preprocess.py`)

Applies 5 preprocessing decisions in sequence on a working copy of `merged_raw.csv`. The original is never modified. An intermediate snapshot CSV is saved after each decision.

```
Input  : output/merged_raw.csv
Output : output/preprocessed.csv  (5,569 rows × 39 columns)

Decision 002 — Blood pressure averaging         → +2 columns (Avg_Systolic_BP, Avg_Diastolic_BP)
Decision 003 — Age filter (20–80 years)         → −3,685 rows
Decision 004 — DEXA scan validity filter        → invalid DEXA values → NaN (no rows dropped)
Decision 005 — Insulin LOD handling             → 7 below-LOD values → NaN
Decision 006 — Derived features                 → +3 columns (HOMA_IR, TC_HDL_ratio, TG_HDL_ratio)
```

---

## Data Setup

This project uses **NHANES 2017–2018** data, converted from SAS XPT format to CSV.

### Files needed (place in `DATASET/` folder)

| File | NHANES Component | Download URL |
|------|-----------------|-------------|
| `DEMO_J.csv` | Demographics | [DEMO_J.XPT](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT) |
| `BMX_J.csv` | Body Measures | [BMX_J.XPT](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT) |
| `BPX_J.csv` | Blood Pressure | [BPX_J.XPT](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPX_J.XPT) |
| `DXX_J.csv` | DEXA Body Scan | [DXX_J.XPT](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DXX_J.XPT) |
| `GLU_J.csv` | Fasting Glucose | [GLU_J.XPT](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/GLU_J.XPT) |
| `INS_J.csv` | Fasting Insulin | [INS_J.XPT](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/INS_J.XPT) |
| `HDL_J.csv` | HDL Cholesterol | [HDL_J.XPT](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/HDL_J.XPT) |
| `TCHOL_J.csv` | Total Cholesterol | [TCHOL_J.XPT](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/TCHOL_J.XPT) |
| `BIOPRO_J.csv` | Biochemistry Panel | [BIOPRO_J.XPT](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BIOPRO_J.XPT) |

### Converting XPT to CSV

```python
import pandas as pd

xpt_file = "DEMO_J.XPT"
df = pd.read_sas(xpt_file, format="xport", encoding="utf-8")
df.to_csv(xpt_file.replace(".XPT", ".csv"), index=False)
```

---

## Installation & Usage

### Requirements

```bash
pip install -r requirements.txt
```

### Run the merge pipeline

```bash
python run_pipeline.py
```

Optional — explore all available columns in every CSV before editing config:

```bash
python run_pipeline.py --explore
```

### Run preprocessing

```bash
python run_preprocess.py
```

### Inspect data

```bash
python test.py
```

---

## Modifying Variables

All variable selections live in one file: **`pipeline/config.py`**

To add a variable to an existing dataset:
```python
# In SELECTED_VARIABLES, find the right key and add your variable:
"biochemistry": [
    "LBXSTR",
    "LBXSATSI",
    "LBXSAL",
    "LBXNEWVAR",   # ← add here
    ...
],
```

To add a completely new NHANES dataset, add entries to `FILE_MAP`, `SELECTED_VARIABLES`, and `MERGE_ORDER` — all in `config.py`. The runtime assertion will catch any mismatch.

---

## Documentation

| Document | Location |
|----------|----------|
| Preprocessing Decision Log | `docs/preprocessing_decision_log.md` |
| Variable Selection Rationale | `docs/variable_selection_rationale.md` |

The **Preprocessing Decision Log** is a living document that records every preprocessing decision: biological reasoning, statistical reasoning, alternatives considered, and measured effects on the dataset. It follows strict rules: entries are never deleted; changes create new entries referencing the original.

---

## Research Context

This project is **Phase 1** of a two-phase study:

| Phase | Focus | Data | Status |
|-------|-------|------|--------|
| Phase 1 | Phenotypic clustering on biomarkers | NHANES 2017–2018 | 🔄 In progress |
| Phase 2 | Genomic validation against CSIR-IGIB TRISUTRA SNP lists | 1000 Genomes Project | Future work |

The CSIR-IGIB TRISUTRA project (Dr. Mitali Mukerji, New Delhi) previously identified genetic variants (EGLN1, CYP2C19, HLA-B, VWF) linked to Dosha phenotypes using supervised classification. This project provides an independent unsupervised validation of those findings using a Western biomarker dataset.

---

## License

MIT License — see `LICENSE` for details.
