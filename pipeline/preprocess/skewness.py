"""
pipeline/preprocess/skewness.py
================================
Decision 010: Skewness Transformation Evaluation

Evaluates the impact of log1p transformation on strongly right-skewed variables
identified during EDA (|skew| > 2.0).

Rules:
    - Input DataFrame is never mutated. Returns a new copy.
    - Generates parallel log-transformed representations (suffix `_log1p`) for evaluation.
    - Does NOT permanently discard raw representations or force log transformation for clustering.
    - Logs a detailed skewness comparison report (before vs after).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Strongly skewed variables identified in Phase 3 EDA (|skew| > 2.0)
# ---------------------------------------------------------------------------
SKEWED_VARIABLES_TO_EVALUATE: list[str] = [
    "LBXGLU",       # Fasting Glucose (skew = 3.72)
    "LBXIN",        # Fasting Insulin (skew = 9.75)
    "HOMA_IR",      # HOMA-IR (skew = 10.53)
    "LBXSTR",       # Triglycerides (skew = 6.45)
    "TC_HDL_ratio", # TC/HDL ratio (skew = 3.17)
    "TG_HDL_ratio", # TG/HDL ratio (skew = 8.04)
    "LBXSATSI",     # ALT (skew = 3.87)
    "LBXSTB",       # Total Bilirubin (skew = 2.14)
    "LBXSCR",       # Serum Creatinine (skew = 12.04)
    "LBXSBU",       # BUN (skew = 2.42)
]


def evaluate_skewness_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision 010: Skewness Transformation Evaluation.

    Applies log1p transformation to strongly skewed features and creates
    evaluated representation columns with `_log1p` suffix.

    Parameters
    ----------
    df : pd.DataFrame
        Input preprocessed dataframe.

    Returns
    -------
    pd.DataFrame
        New dataframe containing both original columns and `_log1p` evaluated columns.
    """
    result = df.copy()

    sep = "-" * 65
    print(f"\n{sep}")
    print(f"  Decision 010 -- Skewness Transformation Evaluation (|skew| > 2.0)")
    print(sep)
    print(f"  {'Variable':<18} {'Original Skew':>15} {'Log1p Skew':>15} {'Delta Skew':>15}")
    print(f"  {'-'*18} {'-'*15} {'-'*15} {'-'*15}")

    for col in SKEWED_VARIABLES_TO_EVALUATE:
        if col not in result.columns:
            continue
        
        orig_series = result[col]
        orig_skew = orig_series.skew()

        # Log1p transformation (log(1 + x)) — safely handles non-negative biological concentrations
        log_col_name = f"{col}_log1p"
        result[log_col_name] = np.log1p(orig_series)
        log_skew = result[log_col_name].skew()

        delta = log_skew - orig_skew
        print(f"  {col:<18} {orig_skew:>15.3f} {log_skew:>15.3f} {delta:>15.3f}")

    print(f"\n  Note: Parallel _log1p columns created for evaluation.")
    print(f"        Raw representations are preserved; winner not automatically selected.")
    print(sep)

    return result
