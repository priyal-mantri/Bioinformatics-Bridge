"""
pipeline/preprocess/outliers.py
================================
Decision 007: Outlier & Invalid Value Handling

Converts demonstrably invalid or physiologically impossible measurement values
to NaN while retaining the participant's valid measurements for other variables.

Rules:
    - Input DataFrame is never mutated. Returns a new copy.
    - Extreme but biologically plausible values (e.g. Diastolic BP up to 135 mmHg,
      Systolic BP up to 238 mmHg, BMI up to 86.2) are RETAINED without alteration.
    - Only physiologically impossible values (e.g. Diastolic BP < 20 mmHg representing
      0 mmHg recording artifacts) are converted to NaN.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIASTOLIC_BP_COL: str = "Avg_Diastolic_BP"
MIN_PLAUSIBLE_DIASTOLIC_BP: float = 20.0  # mmHg


def apply_outlier_invalid_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision 007: Outlier & Invalid Value Handling.

    Identifies and converts physiologically impossible values to NaN.
    Specifically:
      - Avg_Diastolic_BP values < 20 mmHg (e.g. 0 mmHg examination artifacts)
        are set to NaN.

    Biologically plausible extremes (e.g. severe hypertension or high BMI) are preserved.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing Avg_Diastolic_BP.

    Returns
    -------
    pd.DataFrame
        New dataframe with invalid values replaced by NaN.
    """
    if DIASTOLIC_BP_COL not in df.columns:
        raise KeyError(f"[Decision 007] Column '{DIASTOLIC_BP_COL}' not found in dataframe.")

    result = df.copy()

    # Identify invalid diastolic BP (< 20 mmHg)
    invalid_bp_mask = (result[DIASTOLIC_BP_COL].notna()) & (result[DIASTOLIC_BP_COL] < MIN_PLAUSIBLE_DIASTOLIC_BP)
    n_invalid_bp = invalid_bp_mask.sum()

    nulls_before = result[DIASTOLIC_BP_COL].isna().sum()
    result.loc[invalid_bp_mask, DIASTOLIC_BP_COL] = np.nan
    nulls_after = result[DIASTOLIC_BP_COL].isna().sum()

    # Report
    sep = "-" * 65
    print(f"\n{sep}")
    print(f"  Decision 007 -- Outlier & Invalid Value Handling")
    print(sep)
    print(f"  Target variable                   : {DIASTOLIC_BP_COL}")
    print(f"  Plausible physiological range     : >= {MIN_PLAUSIBLE_DIASTOLIC_BP} mm Hg")
    print(f"  Physiologically impossible (<20)   : {n_invalid_bp:,} values --> set to NaN")
    print(f"  Rows removed                      : 0 (participants retained)")
    print(f"  {DIASTOLIC_BP_COL} nulls before     : {nulls_before:,}")
    print(f"  {DIASTOLIC_BP_COL} nulls after      : {nulls_after:,} (+{nulls_after - nulls_before:,})")
    print(f"  Plausible extreme values retained : Diastolic BP up to {result[DIASTOLIC_BP_COL].max():.1f} mm Hg")
    print(sep)

    return result
