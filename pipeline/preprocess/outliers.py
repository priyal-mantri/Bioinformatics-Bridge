"""
pipeline/preprocess/outliers.py
================================
Decision 007: Outlier & Invalid Value Handling

Rules:
    - Input DataFrame is never mutated. Returns a new copy.
    - Biologically extreme values (e.g. Diastolic BP up to 135 mmHg, Systolic BP up to 238 mmHg,
      BMI up to 86.2) are RETAINED without modification.
    - Diastolic BP = 0.0 mmHg is explicitly permitted by official NHANES documentation
      (BPX_J.pdf page 2: "Diastolic BP can be zero") and is RETAINED as a valid observation.
    - No automatic deletion of participants or invalidation of < 20 mmHg values is performed.
      Low/zero values are flagged for post-processing EDA inspection rather than deleted.
"""

import numpy as np
import pandas as pd


DIASTOLIC_BP_COL: str = "Avg_Diastolic_BP"


def apply_outlier_invalid_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision 007: Outlier & Invalid Value Handling.

    Validates that extreme observations are retained in accordance with NHANES guidelines.
    Specifically:
      - Removes custom < 20 mmHg invalidation rule.
      - Retains 0.0 mmHg diastolic BP values as valid observations per BPX_J documentation.
      - Retains plausible extreme measurements (hypertension, high BMI).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        New dataframe with extreme observations preserved.
    """
    result = df.copy()

    n_zero_dia = (result[DIASTOLIC_BP_COL] == 0.0).sum() if DIASTOLIC_BP_COL in result.columns else 0
    n_null_dia = result[DIASTOLIC_BP_COL].isna().sum() if DIASTOLIC_BP_COL in result.columns else 0

    sep = "-" * 65
    print(f"\n{sep}")
    print(f"  Decision 007 -- Outlier & Invalid Value Handling")
    print(sep)
    print(f"  Rule applied                      : Retain all NHANES-valid observations.")
    print(f"  Custom <20 mmHg invalidation rule : REMOVED (0.0 mmHg diastolic BP retained)")
    print(f"  Diastolic BP == 0.0 mmHg count    : {n_zero_dia:,} participants (retained)")
    print(f"  Diastolic BP missing count        : {n_null_dia:,} participants")
    print(f"  Rows removed                      : 0 (participants retained)")
    print(sep)

    return result
