"""
pipeline/preprocess/filters.py
================================
Row and value filters for the NHANES preprocessing pipeline.

Implements three decisions from the Preprocessing Decision Log:

    Decision 003 -- Participant Age Filter
        Keep only adults aged 20 to 80 years (RIDAGEYR).

    Decision 004 -- DEXA Scan Validity Filter
        Null out DEXA body-composition values for participants whose
        scan status flag (DXAEXSTS) is not 1 (valid complete scan).
        The row is kept; only the DEXA values are set to NaN.

    Decision 005 -- Insulin Below-Detection-Limit Handling
        Null out LBXIN for participants whose below-detection-limit
        flag (LBDINLC) equals 1. The row is kept; only LBXIN is
        set to NaN.

Rules enforced in every function:
    - Input DataFrame is never mutated.  All functions return a new copy.
    - No rows are deleted in Decisions 004 and 005.
    - Diagnostic reports are printed before and after each filter.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants -- column names centralised for easy modification
# ---------------------------------------------------------------------------

# Decision 003
AGE_COLUMN: str = "RIDAGEYR"
AGE_MIN: int    = 20
AGE_MAX: int    = 80

# Decision 004
DEXA_STATUS_COL: str      = "DXAEXSTS"
DEXA_VALID_CODE: int      = 1
DEXA_VALUE_COLS: list[str] = ["DXDTOBMD", "DXDTOPF", "DXDTOLE"]

# Decision 005
INSULIN_COL: str        = "LBXIN"
INSULIN_FLAG_COL: str   = "LBDINLC"
INSULIN_FLAG_VALUE: int = 1


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, cols: list[str], step: str) -> None:
    """Raise KeyError with a clear message if any required column is absent."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"[{step}] Required column(s) not found in dataframe: {missing}\n"
            f"Check that the merge pipeline has been run and config.py "
            f"includes these variables."
        )


# ---------------------------------------------------------------------------
# Decision 003 -- Age Filter
# ---------------------------------------------------------------------------

def apply_age_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision 003: Retain only participants aged 20–80 years inclusive.

    Why:
        NHANES covers all ages.  Children and adolescents have
        developmental body-composition and blood-pressure profiles
        that are incomparable to adult metabolic phenotypes.
        Participants coded as 80+ cannot be distinguished by exact age.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.  Must contain RIDAGEYR.

    Returns
    -------
    pd.DataFrame
        New dataframe with only adult rows retained.
    """
    _require_columns(df, [AGE_COLUMN], "Decision 003 - Age Filter")

    rows_before = len(df)
    age_dist_before = df[AGE_COLUMN].describe().round(1)

    # Apply filter -- return a copy so the input is never mutated
    mask = (df[AGE_COLUMN] >= AGE_MIN) & (df[AGE_COLUMN] <= AGE_MAX)
    filtered = df.loc[mask].copy()

    rows_after  = len(filtered)
    rows_removed = rows_before - rows_after

    # Report
    sep = "-" * 65
    print(f"\n{sep}")
    print(f"  Decision 003 -- Age Filter  ({AGE_MIN} <= RIDAGEYR <= {AGE_MAX})")
    print(sep)
    print(f"  Rows before : {rows_before:,}")
    print(f"  Rows removed: {rows_removed:,}  "
          f"(aged <{AGE_MIN} or >{AGE_MAX})")
    print(f"  Rows after  : {rows_after:,}")
    print(f"\n  Age distribution BEFORE filter:")
    print(f"    min={age_dist_before['min']:.0f}  "
          f"mean={age_dist_before['mean']:.1f}  "
          f"max={age_dist_before['max']:.0f}")
    age_dist_after = filtered[AGE_COLUMN].describe().round(1)
    print(f"  Age distribution AFTER filter:")
    print(f"    min={age_dist_after['min']:.0f}  "
          f"mean={age_dist_after['mean']:.1f}  "
          f"max={age_dist_after['max']:.0f}")
    print(sep)

    return filtered


# ---------------------------------------------------------------------------
# Decision 004 -- DEXA Scan Validity Filter
# ---------------------------------------------------------------------------

def apply_dexa_validity_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision 004: Null out DEXA values for invalid or incomplete scans.

    Participants with DXAEXSTS != 1 had scans that were partial,
    failed, or produced unreliable measurements.  Their DEXA body-
    composition values (bone density, % fat, lean mass) are NOT
    representative of whole-body measurements and must not be used.

    Rows are NOT deleted: the participant remains in the dataset
    with valid values for all non-DEXA measurements.

    DXAEXSTS codes:
        1 = Whole body scan completed (valid)
        2 = Completed but data invalid
        3 = Not completed, fat tissue invalid
        4 = Not completed, lean tissue invalid
        5 = Not completed, bone tissue invalid
        6 = Not completed, all tissue invalid

    Parameters
    ----------
    df : pd.DataFrame
        Must contain DXAEXSTS and DEXA value columns.

    Returns
    -------
    pd.DataFrame
        New dataframe with invalid DEXA values replaced by NaN.
        Row count is unchanged.
    """
    required = [DEXA_STATUS_COL] + DEXA_VALUE_COLS
    _require_columns(df, required, "Decision 004 - DEXA Validity")

    result = df.copy()

    # Count participants before nulling
    invalid_mask = result[DEXA_STATUS_COL] != DEXA_VALID_CODE
    n_invalid = invalid_mask.sum()
    n_valid   = (~invalid_mask).sum()

    # Count existing nulls in DEXA columns before this step
    nulls_before = {col: result[col].isna().sum() for col in DEXA_VALUE_COLS}

    # Status code distribution
    status_counts = result[DEXA_STATUS_COL].value_counts(dropna=False).sort_index()

    # Apply: set DEXA measurement values to NaN for invalid scans
    # Rows are preserved -- only the DEXA value columns are affected
    result.loc[invalid_mask, DEXA_VALUE_COLS] = np.nan

    # Count nulls after
    nulls_after = {col: result[col].isna().sum() for col in DEXA_VALUE_COLS}

    # Report
    sep = "-" * 65
    print(f"\n{sep}")
    print(f"  Decision 004 -- DEXA Scan Validity Filter")
    print(sep)
    print(f"  DXAEXSTS value distribution:")
    for code, count in status_counts.items():
        label = "(valid)" if code == DEXA_VALID_CODE else "(invalid/missing)"
        print(f"    Code {code}: {count:,} participants  {label}")
    print(f"\n  Participants with valid scans  : {n_valid:,}")
    print(f"  Participants with invalid scans: {n_invalid:,}  --> DEXA values set to NaN")
    print(f"  Rows removed                   : 0  (rows preserved)")
    print(f"\n  Null counts in DEXA columns:")
    print(f"  {'Column':<15} {'Before':>10} {'After':>10} {'Added':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    for col in DEXA_VALUE_COLS:
        added = nulls_after[col] - nulls_before[col]
        print(f"  {col:<15} {nulls_before[col]:>10,} {nulls_after[col]:>10,} {added:>10,}")
    print(sep)

    return result


# ---------------------------------------------------------------------------
# Decision 005 -- Insulin Below-Detection-Limit Handling
# ---------------------------------------------------------------------------

def apply_insulin_lod_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision 005: Set LBXIN to NaN where the value is below detection limit.

    When LBDINLC == 1, the insulin value reported by the NHANES
    laboratory is not a true measurement.  It is a fill value
    assigned by CDC convention (typically: detection_limit / sqrt(2)).
    Using this as a genuine measurement distorts the lower tail of
    the insulin distribution and would corrupt HOMA-IR computation.

    The row is NOT deleted.  Only LBXIN is set to NaN for flagged rows.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain LBXIN and LBDINLC.

    Returns
    -------
    pd.DataFrame
        New dataframe with below-LOD insulin values replaced by NaN.
        Row count is unchanged.
    """
    _require_columns(df, [INSULIN_COL, INSULIN_FLAG_COL],
                     "Decision 005 - Insulin LOD")

    result = df.copy()

    # Identify flagged rows
    flag_mask = result[INSULIN_FLAG_COL] == INSULIN_FLAG_VALUE
    n_flagged = flag_mask.sum()

    nulls_before = result[INSULIN_COL].isna().sum()

    # Apply: set LBXIN to NaN where flagged
    result.loc[flag_mask, INSULIN_COL] = np.nan

    nulls_after = result[INSULIN_COL].isna().sum()

    # Report
    sep = "-" * 65
    print(f"\n{sep}")
    print(f"  Decision 005 -- Insulin Below-Detection-Limit Handling")
    print(sep)
    flag_dist = result[INSULIN_FLAG_COL].value_counts(dropna=False).sort_index()
    print(f"  LBDINLC flag distribution:")
    for code, count in flag_dist.items():
        label = ("(below detection limit -- LBXIN will be NaN)"
                 if code == INSULIN_FLAG_VALUE else "(valid measurement)")
        print(f"    Flag {code}: {count:,}  {label}")
    print(f"\n  Rows flagged (LBDINLC == 1) : {n_flagged:,}  --> LBXIN set to NaN")
    print(f"  Rows removed               : 0  (rows preserved)")
    print(f"  LBXIN nulls before         : {nulls_before:,}")
    print(f"  LBXIN nulls after          : {nulls_after:,}  "
          f"(+{nulls_after - nulls_before:,})")
    print(sep)

    return result
