"""
pipeline/preprocess/cohorts.py
===============================
Decisions 008 & 009: Feature Set Designation & Analysis Cohorts

Decision 008 -- Raw vs. Derived Features:
    Designates the 24 measured raw variables as the PRIMARY clustering feature set.
    Excludes the 3 derived features (HOMA_IR, TC_HDL_ratio, TG_HDL_ratio) from the
    primary clustering matrix while retaining them in preprocessed.csv for interpretation.

Decision 009 -- Missing Data & Analysis Cohorts:
    Defines two reproducible, non-imputed analysis cohorts:
      - ANALYSIS A (Broad Cohort): 19 broadly available features (excludes DEXA & Fasting panel).
        Maximises sample size (N = 4,482 complete cases, 80.5%).
      - ANALYSIS B (Fasting & DEXA Cohort): All 24 raw features (includes fasting & DEXA).
        Maximises feature coverage for metabolic/DEXA sub-analysis (N = 967 complete cases, 17.4%).
"""

import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Feature Set Definitions
# ---------------------------------------------------------------------------

PRIMARY_RAW_FEATURES: list[str] = [
    # Body Composition
    "BMXBMI", "BMXWAIST", "DXDTOBMD", "DXDTOPF", "DXDTOLE",
    # Cardiovascular
    "Avg_Systolic_BP", "Avg_Diastolic_BP", "BPXPLS",
    # Glucose Metabolism
    "LBXGLU", "LBXIN",
    # Lipid Metabolism
    "LBXTC", "LBDHDD", "LBXSTR",
    # Hepatic Function
    "LBXSATSI", "LBXSAL", "LBXSTP", "LBXSTB",
    # Renal Function
    "LBXSCR", "LBXSUA", "LBXSBU",
    # Electrolyte / Mineral
    "LBXSCA", "LBXSPH", "LBXSNASI", "LBXSKSI",
]

DERIVED_FEATURES: list[str] = [
    "HOMA_IR", "TC_HDL_ratio", "TG_HDL_ratio"
]

BROAD_COHORT_FEATURES: list[str] = [
    # 19 features with high population-wide coverage (excluding DEXA & Fasting panel)
    "BMXBMI", "BMXWAIST",
    "Avg_Systolic_BP", "Avg_Diastolic_BP", "BPXPLS",
    "LBXTC", "LBDHDD", "LBXSTR",
    "LBXSATSI", "LBXSAL", "LBXSTP", "LBXSTB",
    "LBXSCR", "LBXSUA", "LBXSBU",
    "LBXSCA", "LBXSPH", "LBXSNASI", "LBXSKSI",
]


def apply_feature_and_cohort_designations(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Apply Decision 008 (Raw vs Derived) and Decision 009 (Analysis Cohorts A & B).

    Parameters
    ----------
    df : pd.DataFrame
        Input preprocessed dataframe.
    output_dir : Path
        Directory to export cohort files.

    Returns
    -------
    pd.DataFrame
        Updated dataframe with boolean flag columns `in_analysis_a` and `in_analysis_b`.
    """
    result = df.copy()

    # ── Decision 008: Feature Designation ──────────────────────────────────
    print("\n" + "-" * 65)
    print("  Decision 008 -- Primary Raw vs. Derived Feature Designation")
    print("-" * 65)
    print(f"  Primary Clustering Raw Features ({len(PRIMARY_RAW_FEATURES)}):")
    print(f"    {PRIMARY_RAW_FEATURES}")
    print(f"\n  Secondary Derived Features ({len(DERIVED_FEATURES)}) [Retained for interpretation]:")
    print(f"    {DERIVED_FEATURES}")

    # ── Decision 009: Analysis Cohorts ──────────────────────────────────────
    print("\n" + "-" * 65)
    print("  Decision 009 -- Missing Data & Analysis Cohorts")
    print("-" * 65)

    # Analysis A: Broad Cohort
    mask_a = result[BROAD_COHORT_FEATURES].notna().all(axis=1)
    result["in_analysis_a"] = mask_a

    # Analysis B: Full Fasting & DEXA Cohort
    mask_b = result[PRIMARY_RAW_FEATURES].notna().all(axis=1)
    result["in_analysis_b"] = mask_b

    n_total = len(result)
    n_a = mask_a.sum()
    n_b = mask_b.sum()

    print(f"  Total Adult Participants (Age 20-80) : {n_total:,}")
    print(f"\n  ANALYSIS A -- Broad Cohort (No DEXA/Fasting panel):")
    print(f"    Features included : {len(BROAD_COHORT_FEATURES)} variables")
    print(f"    Complete cases    : {n_a:,} / {n_total:,} ({n_a/n_total*100:.1f}%)")

    print(f"\n  ANALYSIS B -- Fasting & DEXA Cohort (All 24 Raw Variables):")
    print(f"    Features included : {len(PRIMARY_RAW_FEATURES)} variables")
    print(f"    Complete cases    : {n_b:,} / {n_total:,} ({n_b/n_total*100:.1f}%)")

    # Export distinct cohort datasets for downstream reproducibility
    cohort_a_path = output_dir / "analysis_cohort_a_broad.csv"
    cohort_b_path = output_dir / "analysis_cohort_b_fasting.csv"

    result.loc[mask_a].to_csv(cohort_a_path, index=False)
    result.loc[mask_b].to_csv(cohort_b_path, index=False)

    print(f"\n  [Exported Cohort A] -> {cohort_a_path.name} ({n_a:,} rows x {result.shape[1]} cols)")
    print(f"  [Exported Cohort B] -> {cohort_b_path.name} ({n_b:,} rows x {result.shape[1]} cols)")
    print("-" * 65)

    return result
