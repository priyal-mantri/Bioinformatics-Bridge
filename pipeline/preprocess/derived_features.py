"""
pipeline/preprocess/derived_features.py
=========================================
Derived feature engineering for the NHANES preprocessing pipeline.

Implements Decision 006 from the Preprocessing Decision Log:

    Decision 006 -- Derived Feature Engineering
        Compute three clinically validated composite indices from
        existing variables:

        HOMA_IR     = (LBXGLU * LBXIN) / 405
        TC_HDL_ratio = LBXTC / LBDHDD
        TG_HDL_ratio = LBXSTR / LBDHDD

Rules:
    - Input DataFrame is never mutated.  Returns a new copy.
    - If any component of a formula is NaN, the derived value is NaN.
      This is standard pandas arithmetic propagation -- no special
      handling needed.
    - Must be called AFTER Decisions 003-005 so that invalid values
      (invalid DEXA scans, below-LOD insulin) do not propagate into
      derived features.
    - The formula constants are defined at the top of this file.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Formula definitions -- edit here to change any derived variable
# ---------------------------------------------------------------------------

# Each entry: (output_column_name, formula_description, required_input_cols)
DERIVED_FEATURE_DEFINITIONS: list[tuple[str, str, list[str]]] = [
    (
        "HOMA_IR",
        "(LBXGLU * LBXIN) / 405",
        ["LBXGLU", "LBXIN"],
    ),
    (
        "TC_HDL_ratio",
        "LBXTC / LBDHDD",
        ["LBXTC", "LBDHDD"],
    ),
    (
        "TG_HDL_ratio",
        "LBXSTR / LBDHDD",
        ["LBXSTR", "LBDHDD"],
    ),
]

# HOMA-IR divisor constant (established by Matthews et al. 1985)
HOMA_IR_DIVISOR: float = 405.0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, cols: list[str], step: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"[{step}] Required column(s) not found: {missing}"
        )


# ---------------------------------------------------------------------------
# Decision 006 -- Derived Feature Engineering
# ---------------------------------------------------------------------------

def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision 006: Compute clinically validated composite biomarker indices.

    New columns added:
    ─────────────────────────────────────────────────────────────────
    HOMA_IR
        Homeostatic Model Assessment of Insulin Resistance.
        Formula  : (fasting glucose [mg/dL] * fasting insulin [uU/mL]) / 405
        Reference: Matthews DR et al., Diabetologia (1985) 28:412-419.
        Rationale: Neither glucose nor insulin alone captures insulin
                   resistance.  Their product (normalised by 405) does.
                   A value > 2.5-3.0 is generally used as the clinical
                   threshold for insulin resistance.

    TC_HDL_ratio
        Total Cholesterol to HDL Cholesterol ratio.
        Formula  : LBXTC / LBDHDD
        Reference: American Heart Association; Framingham risk score.
        Rationale: Total cholesterol alone is a poor predictor of
                   cardiovascular risk.  Dividing by HDL ('good'
                   cholesterol) captures the balance between atherogenic
                   and cardioprotective lipid fractions.

    TG_HDL_ratio
        Triglyceride to HDL Cholesterol ratio.
        Formula  : LBXSTR / LBDHDD
        Rationale: A surrogate marker for LDL particle size and metabolic
                   syndrome.  TG/HDL > 3.0 is used as a clinical screen
                   for insulin resistance and dyslipidaemia.

    Missing value behaviour:
        If any input column is NaN for a participant, the derived
        feature for that participant is NaN.  Pandas arithmetic
        propagates NaN by default -- no override is applied.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.  Must contain LBXGLU, LBXIN, LBXTC,
        LBDHDD, LBXSTR.

    Returns
    -------
    pd.DataFrame
        New dataframe with three additional columns appended.
    """
    # Verify all required source columns are present
    all_required = ["LBXGLU", "LBXIN", "LBXTC", "LBDHDD", "LBXSTR"]
    _require_columns(df, all_required, "Decision 006 - Derived Features")

    result = df.copy()

    # Record nulls before computation for the report
    nulls_before = {col: result[col].isna().sum() for col in all_required}

    # ── HOMA-IR ─────────────────────────────────────────────────────────────
    result["HOMA_IR"] = (result["LBXGLU"] * result["LBXIN"]) / HOMA_IR_DIVISOR

    # ── Total Cholesterol / HDL ratio ────────────────────────────────────────
    result["TC_HDL_ratio"] = result["LBXTC"] / result["LBDHDD"]

    # ── Triglyceride / HDL ratio ─────────────────────────────────────────────
    result["TG_HDL_ratio"] = result["LBXSTR"] / result["LBDHDD"]

    # ── Report ───────────────────────────────────────────────────────────────
    sep = "-" * 65
    new_cols = ["HOMA_IR", "TC_HDL_ratio", "TG_HDL_ratio"]

    print(f"\n{sep}")
    print(f"  Decision 006 -- Derived Feature Engineering")
    print(sep)
    print(f"  Columns added: {new_cols}")
    print()
    print(f"  {'Feature':<15} {'Formula':<30} {'Valid':>8} {'Missing':>8} {'% Missing':>10}")
    print(f"  {'-'*15} {'-'*30} {'-'*8} {'-'*8} {'-'*10}")

    for name, formula, _ in DERIVED_FEATURE_DEFINITIONS:
        n_valid   = result[name].notna().sum()
        n_missing = result[name].isna().sum()
        pct       = (n_missing / len(result)) * 100
        print(f"  {name:<15} {formula:<30} {n_valid:>8,} {n_missing:>8,} {pct:>9.1f}%")

    print(f"\n  Descriptive statistics for derived features:")
    print(result[new_cols].describe(percentiles=[0.25, 0.5, 0.75]).round(3).to_string())
    print(sep)

    return result
