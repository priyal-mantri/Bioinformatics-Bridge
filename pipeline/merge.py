"""
pipeline/merge.py
=================
Core merging logic for the NHANES pipeline.

Orchestrates the full merge sequence defined in config.py.
Uses only functions from utils.py — no logic is duplicated here.

This file should rarely need editing.
To change what gets merged, edit config.py only.
"""

import pandas as pd
from pathlib import Path

from pipeline.config import (
    DATASET_DIR,
    OUTPUT_DIR,
    MERGED_OUTPUT_FILE,
    FILE_MAP,
    SELECTED_VARIABLES,
    MERGE_ORDER,
)
from pipeline.utils import (
    load_csv,
    validate_dataset,
    report_after_merge,
    print_final_summary,
)


# ---------------------------------------------------------------------------
# Step 1: Load all datasets
# ---------------------------------------------------------------------------

def load_all_datasets() -> dict[str, pd.DataFrame]:
    """
    Load and validate every dataset defined in MERGE_ORDER.

    For each dataset:
    - Reads the CSV, keeping only SEQN + selected columns
    - Validates SEQN presence and uniqueness
    - Prints a per-dataset report

    Returns
    -------
    dict[str, pd.DataFrame]
        Maps logical dataset name → loaded and validated dataframe.
    """
    print("\n" + "=" * 65)
    print("  PHASE 1: LOADING AND VALIDATING DATASETS")
    print("=" * 65)

    loaded: dict[str, pd.DataFrame] = {}

    for dataset_name in MERGE_ORDER:
        filename = FILE_MAP[dataset_name]
        selected_cols = SELECTED_VARIABLES[dataset_name]
        file_path = DATASET_DIR / filename

        # Load: keep SEQN + selected variables only
        df = load_csv(file_path, selected_cols)

        # Validate: check SEQN uniqueness and print row/column info
        validate_dataset(df, dataset_name)

        loaded[dataset_name] = df

    return loaded


# ---------------------------------------------------------------------------
# Step 2: Merge all datasets sequentially
# ---------------------------------------------------------------------------

def merge_all_datasets(loaded_datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all loaded datasets using sequential LEFT JOINs on SEQN.

    The first dataset in MERGE_ORDER is the master (left side).
    Every subsequent dataset is joined to it in order.

    After each join, a diagnostic report is printed showing:
    - New shape
    - Newly introduced missing values
    - Number of matched participants

    Parameters
    ----------
    loaded_datasets : dict[str, pd.DataFrame]
        Output from load_all_datasets().

    Returns
    -------
    pd.DataFrame
        The fully merged dataframe.
    """
    print("\n" + "=" * 65)
    print("  PHASE 2: MERGING DATASETS")
    print("=" * 65)

    # Start with the master dataset (first in MERGE_ORDER)
    master_name = MERGE_ORDER[0]
    merged = loaded_datasets[master_name].copy()

    print(f"\n  Master dataset  : '{master_name}'")
    print(f"  Starting shape  : {merged.shape}")

    # Join each subsequent dataset in sequence
    for dataset_name in MERGE_ORDER[1:]:
        right_df = loaded_datasets[dataset_name]

        # Count nulls BEFORE the merge to calculate the delta afterward
        nulls_before = int(merged.isna().sum().sum())

        # LEFT JOIN: all master rows are preserved; unmatched right rows → NaN
        merged = merged.merge(right_df, on="SEQN", how="left")

        # Report what changed after this join
        report_after_merge(merged, dataset_name, nulls_before)

    return merged


# ---------------------------------------------------------------------------
# Step 3: Save output
# ---------------------------------------------------------------------------

def save_merged_dataset(df: pd.DataFrame) -> None:
    """
    Save the merged dataframe to CSV.

    Creates the output directory if it does not exist.
    Does NOT modify the dataframe in any way before saving.

    Parameters
    ----------
    df : pd.DataFrame
        The fully merged dataframe.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MERGED_OUTPUT_FILE, index=False)
    print(f"\n  [SAVED] {MERGED_OUTPUT_FILE}")


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_merge_pipeline() -> pd.DataFrame:
    """
    Execute the full merge pipeline end-to-end.

    Steps:
    1. Load and validate all datasets
    2. Merge sequentially via left join on SEQN
    3. Print final summary report
    4. Save merged_raw.csv

    Returns
    -------
    pd.DataFrame
        The final merged dataframe (also saved to disk).
    """
    print("\n" + "=" * 65)
    print("  NHANES MERGE PIPELINE — Research 2: The Bioinformatics Bridge")
    print("=" * 65)
    print(f"  Dataset directory : {DATASET_DIR}")
    print(f"  Output directory  : {OUTPUT_DIR}")
    print(f"  Merge order       : {' -> '.join(MERGE_ORDER)}")

    # Phase 1: Load
    loaded_datasets = load_all_datasets()

    # Phase 2: Merge
    merged_df = merge_all_datasets(loaded_datasets)

    # Phase 3: Final summary report
    print_final_summary(merged_df, MERGED_OUTPUT_FILE)

    # Phase 4: Save
    save_merged_dataset(merged_df)

    return merged_df
