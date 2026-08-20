"""
pipeline/preprocess/bp_averaging.py
====================================
Blood pressure averaging module for NHANES 2017-2018 based on official CDC protocols.

OFFICIAL CDC NHANES BP COMBINING CONVENTION
-------------------------------------------
Source: CDC NHANES Physician Examination Procedures Manual & AHA/ACC Analytic Guidelines.

NHANES takes up to three consecutive blood pressure readings during a single exam visit.
The CDC NHANES standard protocol for combining blood pressure readings:

1. If 3 valid readings are available (BPXSY1, BPXSY2, BPXSY3):
   -> DROP Reading 1 (to eliminate initial white-coat stress reactivity).
   -> Compute average of Readings 2 and 3: mean(BPXSY2, BPXSY3) and mean(BPXDI2, BPXDI3).

2. If 2 valid readings are available:
   -> If Readings 2 and 3 are present -> average Readings 2 and 3.
   -> If Readings 1 and 2 are present -> use Reading 2.
   -> If Readings 1 and 3 are present -> use Reading 3.

3. If 1 valid reading is available:
   -> Use that single available reading.

4. If 0 valid readings are available:
   -> Result is NaN.

SAS XPT ZERO CONVERSION
-----------------------
In raw SAS XPT export format, 0 mmHg diastolic BP (which NHANES page 2 explicitly permits)
is stored in IEEE floating point format and imports as 5.3976e-79. This module converts
floating point representations < 1e-10 to clean float 0.0.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTOLIC_COLS: list[str] = ["BPXSY1", "BPXSY2", "BPXSY3"]
DIASTOLIC_COLS: list[str] = ["BPXDI1", "BPXDI2", "BPXDI3"]

AVG_SYSTOLIC_COL: str  = "Avg_Systolic_BP"
AVG_DIASTOLIC_COL: str = "Avg_Diastolic_BP"


def _combine_readings_cdc(r1: float, r2: float, r3: float) -> float:
    """
    Combine three NHANES blood pressure readings following CDC analytic guidelines.
    """
    valid = []
    vals = [r1, r2, r3]
    for idx, v in enumerate(vals):
        if pd.notna(v):
            valid.append((idx + 1, v))

    n_valid = len(valid)

    if n_valid == 3:
        # Drop reading 1; average readings 2 and 3
        return (r2 + r3) / 2.0
    elif n_valid == 2:
        # If readings 2 & 3 present, average them
        if pd.notna(r2) and pd.notna(r3):
            return (r2 + r3) / 2.0
        elif pd.notna(r2):
            return r2
        elif pd.notna(r3):
            return r3
        else:
            return r1
    elif n_valid == 1:
        return valid[0][1]
    else:
        return np.nan


def compute_bp_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Avg_Systolic_BP and Avg_Diastolic_BP using official CDC NHANES protocol.

    Parameters
    ----------
    df : pd.DataFrame
        Input merged dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame with Avg_Systolic_BP and Avg_Diastolic_BP appended.
    """
    result = df.copy()

    # Convert SAS XPT IEEE zero exports (< 1e-10) to clean float 0.0
    for col in DIASTOLIC_COLS:
        if col in result.columns:
            result[col] = result[col].apply(lambda v: 0.0 if (pd.notna(v) and v < 1e-10) else v)

    sys_avgs = []
    dia_avgs = []

    for _, row in result.iterrows():
        s1, s2, s3 = row["BPXSY1"], row["BPXSY2"], row["BPXSY3"]
        d1, d2, d3 = row["BPXDI1"], row["BPXDI2"], row["BPXDI3"]

        sys_avgs.append(_combine_readings_cdc(s1, s2, s3))
        dia_avgs.append(_combine_readings_cdc(d1, d2, d3))

    result[AVG_SYSTOLIC_COL] = sys_avgs
    result[AVG_DIASTOLIC_COL] = dia_avgs

    return result


def run_bp_averaging(df: pd.DataFrame, output_path: Path | None = None) -> pd.DataFrame:
    """
    Full blood pressure averaging pipeline following CDC NHANES convention.
    """
    sep = "-" * 65
    print(f"\n{sep}")
    print("  DECISION 002 -- CDC NHANES Blood Pressure Averaging Protocol")
    print(sep)

    df_out = compute_bp_averages(df)

    n_sys_valid = df_out[AVG_SYSTOLIC_COL].notna().sum()
    n_dia_valid = df_out[AVG_DIASTOLIC_COL].notna().sum()

    print(f"  Protocol: Drop Reading 1 when >= 2 readings exist; average Readings 2 & 3.")
    print(f"  Valid Avg_Systolic_BP  : {n_sys_valid:,} participants")
    print(f"  Valid Avg_Diastolic_BP : {n_dia_valid:,} participants")
    print(f"  Mean Systolic BP       : {df_out[AVG_SYSTOLIC_COL].mean():.2f} mm Hg")
    print(f"  Mean Diastolic BP      : {df_out[AVG_DIASTOLIC_COL].mean():.2f} mm Hg")
    print(f"  Zero Diastolic values  : {(df_out[AVG_DIASTOLIC_COL] == 0.0).sum()} participants (retained as valid)")
    print(sep)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output_path, index=False)

    return df_out
