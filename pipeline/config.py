"""
pipeline/config.py
==================
Central configuration for the NHANES merging pipeline.

To add or remove variables:
  - Edit the SELECTED_VARIABLES dictionary below.
  - Do NOT modify any other file.

To add a new dataset:
  - Add a new key to SELECTED_VARIABLES.
  - Add the corresponding filename to FILE_MAP.
  - Do NOT modify merge.py or utils.py.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

# Root of the project (one level up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder containing all raw NHANES CSV files
DATASET_DIR = PROJECT_ROOT / "DATASET"

# Output folder — created automatically if it does not exist
OUTPUT_DIR = PROJECT_ROOT / "output"

# Final merged file name
MERGED_OUTPUT_FILE = OUTPUT_DIR / "merged_raw.csv"

# ---------------------------------------------------------------------------
# Dataset file map
# Short logical name  →  actual CSV filename (inside DATASET_DIR)
# ---------------------------------------------------------------------------

FILE_MAP: dict[str, str] = {
    "demographics":   "DEMO_J.csv",
    "body_measures":  "BMX_J.csv",
    "blood_pressure": "BPX_J.csv",
    "dexa_scan":      "DXX_J.csv",
    "glucose":        "GLU_J.csv",
    "insulin":        "INS_J.csv",
    "hdl":            "HDL_J.csv",
    "cholesterol":    "TCHOL_J.csv",
    "biochemistry":   "BIOPRO_J.csv",
}

# ---------------------------------------------------------------------------
# Selected variables per dataset
#
# Each key matches a key in FILE_MAP.
# SEQN is added automatically by the pipeline — do NOT list it here.
#
# Biological system covered (for documentation only):
#   demographics   → control / confounder variables
#   body_measures  → System 1: Body Composition
#   blood_pressure → System 2: Cardiovascular
#   dexa_scan      → System 1: Body Composition (DEXA)
#   glucose        → System 3: Glucose Metabolism
#   insulin        → System 3: Glucose Metabolism
#   hdl            → System 4: Lipid Metabolism
#   cholesterol    → System 4: Lipid Metabolism
#   biochemistry   → Systems 4–7: Lipid / Hepatic / Renal / Electrolyte
# ---------------------------------------------------------------------------

SELECTED_VARIABLES: dict[str, list[str]] = {

    # Control / confounder variables (NOT features — used for validation only)
    "demographics": [
        "RIAGENDR",   # Gender (1=Male, 2=Female)
        "RIDAGEYR",   # Age at screening (years)
        "RIDRETH3",   # Race / ethnicity (code 1–7, includes NH Asian)
    ],

    # System 1 — Body Composition: anthropometric measures
    "body_measures": [
        "BMXBMI",     # Body Mass Index (kg/m²)
        "BMXWAIST",   # Waist circumference (cm)
    ],

    # System 2 — Cardiovascular: blood pressure and heart rate
    "blood_pressure": [
        "BPXSY1",     # Systolic BP — 1st reading (mm Hg)
        "BPXSY2",     # Systolic BP — 2nd reading (mm Hg)
        "BPXSY3",     # Systolic BP — 3rd reading (mm Hg)
        "BPXDI1",     # Diastolic BP — 1st reading (mm Hg)
        "BPXDI2",     # Diastolic BP — 2nd reading (mm Hg)
        "BPXDI3",     # Diastolic BP — 3rd reading (mm Hg)
        "BPXPLS",     # Resting pulse / heart rate (beats/min)
    ],

    # System 1 — Body Composition: DEXA full-body scan
    # DXAEXSTS is a validity flag (1=valid) — kept for cleaning, not clustering
    "dexa_scan": [
        "DXAEXSTS",   # Exam status flag (1=valid scan) — for cleaning only
        "DXDTOBMD",   # Total Body Bone Mineral Density (g/cm²)
        "DXDTOPF",    # Total Body Percent Fat (%)
        "DXDTOLE",    # Total Lean Mass excl. bone (g)
    ],

    # System 3 — Glucose Metabolism: fasting blood glucose
    "glucose": [
        "LBXGLU",     # Fasting Plasma Glucose (mg/dL)
    ],

    # System 3 — Glucose Metabolism: fasting insulin
    # LBDINLC is a below-detection-limit flag — kept for cleaning, not clustering
    "insulin": [
        "LBXIN",      # Fasting Insulin (uU/mL)
        "LBDINLC",    # Below-detection-limit flag — for cleaning only
    ],

    # System 4 — Lipid Metabolism: HDL cholesterol
    "hdl": [
        "LBDHDD",     # HDL Cholesterol — good cholesterol (mg/dL)
    ],

    # System 4 — Lipid Metabolism: total cholesterol
    "cholesterol": [
        "LBXTC",      # Total Cholesterol (mg/dL)
    ],

    # Systems 4–7 — Lipid / Hepatic / Renal / Electrolyte
    "biochemistry": [
        # System 4 — Lipid
        "LBXSTR",     # Triglycerides (mg/dL)

        # System 5 — Hepatic (Liver) Function
        "LBXSATSI",   # ALT — alanine aminotransferase (U/L)
        "LBXSAL",     # Albumin — blood protein (g/dL)
        "LBXSTP",     # Total Protein (g/dL)
        "LBXSTB",     # Total Bilirubin (mg/dL)

        # System 6 — Renal (Kidney) Function
        "LBXSCR",     # Creatinine (mg/dL)
        "LBXSUA",     # Uric Acid (mg/dL)
        "LBXSBU",     # Blood Urea Nitrogen — BUN (mg/dL)

        # System 7 — Electrolyte / Mineral Balance
        "LBXSCA",     # Total Calcium (mg/dL)
        "LBXSPH",     # Phosphorus (mg/dL)
        "LBXSNASI",   # Sodium (mmol/L)
        "LBXSKSI",    # Potassium (mmol/L)
    ],
}

# ---------------------------------------------------------------------------
# Merge order
# The first entry is the MASTER dataset (left side of every join).
# All others are joined in sequence.
# ---------------------------------------------------------------------------

MERGE_ORDER: list[str] = [
    "demographics",   # master — 9,254 participants
    "body_measures",
    "blood_pressure",
    "dexa_scan",
    "glucose",
    "insulin",
    "hdl",
    "cholesterol",
    "biochemistry",
]

# Sanity check: every key in MERGE_ORDER must exist in both FILE_MAP and SELECTED_VARIABLES
assert set(MERGE_ORDER) == set(FILE_MAP.keys()) == set(SELECTED_VARIABLES.keys()), (
    "Mismatch between MERGE_ORDER, FILE_MAP, and SELECTED_VARIABLES. "
    "All three must have the same keys."
)
