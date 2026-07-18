"""
pipeline/preprocess/bp_averaging.py
====================================
Blood pressure averaging module for NHANES 2017-2018.

NHANES protocol:
    Up to three blood pressure readings are taken during a single examination
    visit. The first reading is a practice reading and all readings are
    recorded. Standard epidemiological practice is to average the available
    readings to produce a single representative value per participant.

WHY AVERAGING IS STATISTICALLY PREFERABLE TO USING ONLY THE FIRST READING
--------------------------------------------------------------------------
1.  Measurement error reduction
    Each individual blood pressure reading contains both true signal and
    random measurement error. Averaging multiple readings reduces the
    standard error of the estimate by a factor of 1/sqrt(n), making the
    average a more precise estimate of the participant's true blood pressure.

2.  White-coat effect and reactivity
    The first reading is typically the highest because participants are
    often anxious at the start of an examination (white-coat effect).
    Subsequent readings are taken after the participant has had time to
    relax. Averaging dampens this transient spike and reflects habitual
    blood pressure more accurately.

3.  Intra-session variability
    Blood pressure fluctuates naturally within minutes. A single reading
    captures one point in this fluctuation. The average across multiple
    readings captures the underlying level of that fluctuation distribution.

4.  Concordance with clinical guidelines
    The American Heart Association (AHA) and the Seventh Report of the
    Joint National Committee (JNC-7) both recommend averaging two or more
    readings taken at the same visit. NHANES itself notes in its analytic
    guidelines that the average of available readings is preferred.

5.  Preservation of sample size
    Using only the first reading means that any participant where BPXSY1
    is missing loses their blood pressure data entirely -- even if they
    have valid readings 2 and 3. Averaging with skipna=True retains those
    participants.

WHAT THIS MODULE DOES
---------------------
- Accepts a DataFrame (already merged) as input.
- Computes Avg_Systolic_BP  = nanmean(BPXSY1, BPXSY2, BPXSY3)
- Computes Avg_Diastolic_BP = nanmean(BPXDI1, BPXDI2, BPXDI3)
- If ALL readings for a participant are NaN, the average is NaN (not 0).
- Preserves all original raw columns for full traceability.
- Prints a before/after summary report.
- Does NOT delete rows, impute values, or modify any other variable.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants: column names (centralised so they are easy to change)
# ---------------------------------------------------------------------------

# Raw systolic reading column names (in NHANES examination order)
SYSTOLIC_COLS: list[str] = ["BPXSY1", "BPXSY2", "BPXSY3"]

# Raw diastolic reading column names (in NHANES examination order)
DIASTOLIC_COLS: list[str] = ["BPXDI1", "BPXDI2", "BPXDI3"]

# Output column names for the computed averages
AVG_SYSTOLIC_COL: str  = "Avg_Systolic_BP"
AVG_DIASTOLIC_COL: str = "Avg_Diastolic_BP"


# ---------------------------------------------------------------------------
# Helper: verify required columns exist in the DataFrame
# ---------------------------------------------------------------------------

def _verify_columns_present(df: pd.DataFrame, required: list[str]) -> None:
    """
    Raise a clear KeyError if any required column is missing from df.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to inspect.
    required : list[str]
        Column names that must be present.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f"[ERROR] The following blood pressure columns are missing from "
            f"the dataframe:\n"
            f"        Missing : {missing}\n"
            f"        Available columns: {sorted(df.columns.tolist())}\n\n"
            f"        Ensure that the merge pipeline has been run with these "
            f"variables included in config.py before running bp_averaging."
        )


# ---------------------------------------------------------------------------
# Core function: summarise raw BP readings
# ---------------------------------------------------------------------------

def print_bp_summary_before(df: pd.DataFrame) -> None:
    """
    Print descriptive statistics for raw systolic and diastolic readings
    BEFORE computing averages.

    Parameters
    ----------
    df : pd.DataFrame
        The merged dataframe containing raw BP columns.
    """
    separator = "-" * 65

    print(f"\n{separator}")
    print("  BLOOD PRESSURE SUMMARY -- BEFORE AVERAGING")
    print(separator)

    all_raw_cols = SYSTOLIC_COLS + DIASTOLIC_COLS
    available_raw = [c for c in all_raw_cols if c in df.columns]

    stats = df[available_raw].describe(percentiles=[0.25, 0.50, 0.75]).T

    # Add a missing-value count column for visibility
    stats.insert(0, "n_missing", df[available_raw].isna().sum())
    stats.insert(1, "pct_missing", (df[available_raw].isna().mean() * 100).round(1))

    print(stats.to_string())

    # Count how many participants have at least one valid reading per group
    sys_available  = df[SYSTOLIC_COLS].notna().any(axis=1).sum()
    dias_available = df[DIASTOLIC_COLS].notna().any(axis=1).sum()
    print(f"\n  Participants with >= 1 valid systolic reading  : {sys_available:,}")
    print(f"  Participants with >= 1 valid diastolic reading : {dias_available:,}")

    # Count how many have all three readings
    sys_complete  = df[SYSTOLIC_COLS].notna().all(axis=1).sum()
    dias_complete = df[DIASTOLIC_COLS].notna().all(axis=1).sum()
    print(f"  Participants with all 3 systolic readings      : {sys_complete:,}")
    print(f"  Participants with all 3 diastolic readings     : {dias_complete:,}")
    print(separator)


# ---------------------------------------------------------------------------
# Core function: compute the averages
# ---------------------------------------------------------------------------

def compute_bp_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Avg_Systolic_BP and Avg_Diastolic_BP for each participant.

    Uses pandas rowwise mean with skipna=True so that:
    - A participant with 2 of 3 readings available gets the mean of 2.
    - A participant with all readings missing gets NaN (not 0).

    The original raw columns are preserved unchanged.
    Two new columns are appended to the right of the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The merged dataframe. Must contain BPXSY1-3 and BPXDI1-3.

    Returns
    -------
    pd.DataFrame
        Same dataframe with two new columns:
        Avg_Systolic_BP and Avg_Diastolic_BP.
    """
    # Verify all required source columns exist before computing
    _verify_columns_present(df, SYSTOLIC_COLS + DIASTOLIC_COLS)

    # Work on a copy -- do not mutate the input DataFrame
    result = df.copy()

    # ── Systolic average ────────────────────────────────────────────────────
    # axis=1 → row-wise operation (one participant per row)
    # skipna=True → ignore NaN in the mean; if all NaN, result is NaN
    # min_count=1 → if all values in a row are NaN, return NaN not 0
    result[AVG_SYSTOLIC_COL] = (
        result[SYSTOLIC_COLS]
        .mean(axis=1, skipna=True)
    )

    # Enforce min_count=1 rule: if all three readings are NaN, set to NaN
    # (pandas .mean() already handles this correctly; this line is explicit
    # documentation that the behaviour is intentional, not accidental)
    all_systolic_missing = result[SYSTOLIC_COLS].isna().all(axis=1)
    result.loc[all_systolic_missing, AVG_SYSTOLIC_COL] = np.nan

    # ── Diastolic average ────────────────────────────────────────────────────
    result[AVG_DIASTOLIC_COL] = (
        result[DIASTOLIC_COLS]
        .mean(axis=1, skipna=True)
    )

    all_diastolic_missing = result[DIASTOLIC_COLS].isna().all(axis=1)
    result.loc[all_diastolic_missing, AVG_DIASTOLIC_COL] = np.nan

    return result


# ---------------------------------------------------------------------------
# Core function: summarise averaged BP columns
# ---------------------------------------------------------------------------

def print_bp_summary_after(df: pd.DataFrame) -> None:
    """
    Print descriptive statistics for the computed averages
    AFTER bp_averaging has been applied.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe after compute_bp_averages() has been called.
    """
    separator = "-" * 65

    print(f"\n{separator}")
    print("  BLOOD PRESSURE SUMMARY -- AFTER AVERAGING")
    print(separator)

    avg_cols = [AVG_SYSTOLIC_COL, AVG_DIASTOLIC_COL]
    _verify_columns_present(df, avg_cols)

    stats = df[avg_cols].describe(percentiles=[0.25, 0.50, 0.75]).T
    stats.insert(0, "n_missing", df[avg_cols].isna().sum())
    stats.insert(1, "pct_missing", (df[avg_cols].isna().mean() * 100).round(1))

    print(stats.to_string())

    # Compare coverage gain against using only the first reading
    sys_first_valid  = df["BPXSY1"].notna().sum()
    sys_avg_valid    = df[AVG_SYSTOLIC_COL].notna().sum()
    dias_first_valid = df["BPXDI1"].notna().sum()
    dias_avg_valid   = df[AVG_DIASTOLIC_COL].notna().sum()

    print(f"\n  Coverage comparison (participants with a valid value):")
    print(f"  {'Measure':<30} {'1st reading only':>18} {'Average (this step)':>20}")
    print(f"  {'-'*30} {'-'*18} {'-'*20}")
    print(f"  {'Systolic BP':<30} {sys_first_valid:>18,} {sys_avg_valid:>20,}")
    print(f"  {'Diastolic BP':<30} {dias_first_valid:>18,} {dias_avg_valid:>20,}")

    # How many readings went into each average -- distribution
    sys_n_available = df[SYSTOLIC_COLS].notna().sum(axis=1)
    dias_n_available = df[DIASTOLIC_COLS].notna().sum(axis=1)

    print(f"\n  Readings used per participant (systolic):")
    sys_counts = sys_n_available.value_counts().sort_index()
    for n_readings, count in sys_counts.items():
        label = "all missing" if n_readings == 0 else f"{n_readings} reading(s)"
        print(f"    {label:<20}: {count:,} participants")

    print(f"\n  Readings used per participant (diastolic):")
    dias_counts = dias_n_available.value_counts().sort_index()
    for n_readings, count in dias_counts.items():
        label = "all missing" if n_readings == 0 else f"{n_readings} reading(s)"
        print(f"    {label:<20}: {count:,} participants")

    print(separator)


# ---------------------------------------------------------------------------
# Main orchestration function
# ---------------------------------------------------------------------------

def run_bp_averaging(
    df: pd.DataFrame,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Full blood pressure averaging pipeline.

    Steps:
    1. Print before-summary of raw readings.
    2. Compute Avg_Systolic_BP and Avg_Diastolic_BP.
    3. Print after-summary of computed averages.
    4. Optionally save the updated dataframe to a CSV.

    Parameters
    ----------
    df : pd.DataFrame
        The merged NHANES dataframe (output of the merge pipeline).
    output_path : Path | None
        If provided, saves the dataframe with averages to this path.

    Returns
    -------
    pd.DataFrame
        The dataframe with two new columns: Avg_Systolic_BP, Avg_Diastolic_BP.
        All original raw columns are preserved.
    """
    print("\n" + "=" * 65)
    print("  BLOOD PRESSURE AVERAGING -- NHANES Best Practice")
    print("=" * 65)

    # Step 1: Before summary
    print_bp_summary_before(df)

    # Step 2: Compute averages (returns a new DataFrame; input is unchanged)
    df_with_avg = compute_bp_averages(df)

    # Step 3: After summary
    print_bp_summary_after(df_with_avg)

    # Step 4: Optionally save
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_with_avg.to_csv(output_path, index=False)
        print(f"\n  [SAVED] {output_path}")

    print("\n  BP averaging completed successfully.")
    print(f"  New columns added: '{AVG_SYSTOLIC_COL}', '{AVG_DIASTOLIC_COL}'")
    print("=" * 65)

    return df_with_avg
