"""
pipeline/utils.py
=================
Reusable helper functions for the NHANES merging pipeline.

All functions are pure (no side effects beyond printing).
None of these functions modify data — they only read, validate, and report.
"""

import sys
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

def load_csv(file_path: Path, selected_columns: list[str]) -> pd.DataFrame:
    """
    Load a CSV file, keeping only SEQN and the specified selected columns.

    Parameters
    ----------
    file_path : Path
        Absolute path to the CSV file.
    selected_columns : list[str]
        Variable names to retain (SEQN is always added automatically).

    Returns
    -------
    pd.DataFrame
        Dataframe with columns [SEQN] + selected_columns (in that order).

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    KeyError
        If SEQN or any selected column is absent from the file.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"[ERROR] File not found: {file_path}\n"
            f"        Check that DATASET_DIR in config.py points to the correct folder."
        )

    # Read the full file first to inspect available columns
    df_full = pd.read_csv(file_path, low_memory=False)
    available_columns = set(df_full.columns)

    # Verify SEQN exists
    if "SEQN" not in available_columns:
        raise KeyError(
            f"[ERROR] SEQN not found in {file_path.name}. "
            f"Available columns: {sorted(available_columns)}"
        )

    # Verify all selected columns exist
    missing_cols = [c for c in selected_columns if c not in available_columns]
    if missing_cols:
        raise KeyError(
            f"[ERROR] The following columns were requested but NOT found in {file_path.name}:\n"
            f"        Missing : {missing_cols}\n"
            f"        Available: {sorted(available_columns)}"
        )

    # Keep only SEQN + selected columns
    columns_to_keep = ["SEQN"] + selected_columns
    return df_full[columns_to_keep].copy()


# ---------------------------------------------------------------------------
# 2. Validation
# ---------------------------------------------------------------------------

def validate_dataset(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Run pre-merge quality checks on a single dataset and print a report.

    Checks performed:
    - SEQN exists (guaranteed by load_csv, but re-verified defensively)
    - No duplicated SEQN values within the file
    - Row count

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.
    dataset_name : str
        Human-readable name used in console output (e.g. "demographics").

    Raises
    ------
    SystemExit
        If duplicate SEQNs are found (fatal — stops the pipeline).
    """
    separator = "-" * 60
    print(f"\n{separator}")
    print(f"  Validating: {dataset_name}")
    print(separator)

    # Row count
    print(f"  Rows        : {len(df):,}")
    print(f"  Columns     : {list(df.columns)}")

    # Duplicate SEQN check
    n_duplicates = df["SEQN"].duplicated().sum()
    print(f"  Duplicate SEQNs: {n_duplicates}")

    if n_duplicates > 0:
        duplicate_seqns = df[df["SEQN"].duplicated(keep=False)]["SEQN"].unique()
        print(f"  [FATAL] Duplicate SEQNs detected in '{dataset_name}'.")
        print(f"          Affected SEQNs: {duplicate_seqns[:10]} ...")
        print(f"          Cannot merge until duplicates are resolved. Exiting.")
        sys.exit(1)
    else:
        print(f"  [OK] No duplicate SEQNs.")


# ---------------------------------------------------------------------------
# 3. Merge reporting
# ---------------------------------------------------------------------------

def report_after_merge(
    merged_df: pd.DataFrame,
    new_dataset_name: str,
    pre_merge_nulls: int,
) -> None:
    """
    Print a diagnostic report immediately after a merge step.

    Parameters
    ----------
    merged_df : pd.DataFrame
        The dataframe resulting from the latest merge.
    new_dataset_name : str
        Name of the dataset that was just joined in.
    pre_merge_nulls : int
        Total missing value count BEFORE this merge (used to compute delta).
    """
    post_merge_nulls = int(merged_df.isna().sum().sum())
    new_nulls_introduced = post_merge_nulls - pre_merge_nulls
    n_matched = merged_df.dropna(subset=[merged_df.columns[-1]]).shape[0]

    print(f"\n  -- After joining '{new_dataset_name}' --")
    print(f"  Shape                    : {merged_df.shape}")
    print(f"  New missing values added : {new_nulls_introduced:,}")
    print(f"  Matched participants     : {n_matched:,}  "
          f"(had data in '{new_dataset_name}')")


# ---------------------------------------------------------------------------
# 4. Final summary
# ---------------------------------------------------------------------------

def print_final_summary(df: pd.DataFrame, output_path: Path) -> None:
    """
    Print a comprehensive summary of the final merged dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The fully merged dataframe.
    output_path : Path
        Path where merged_raw.csv was saved.
    """
    separator = "=" * 65

    print(f"\n{separator}")
    print("  FINAL MERGED DATASET SUMMARY")
    print(separator)
    print(f"  Total rows              : {len(df):,}")
    print(f"  Total columns           : {len(df.columns)}")
    print(f"  Duplicate SEQN count    : {df['SEQN'].duplicated().sum()}")
    print(f"  Saved to                : {output_path}")

    # Per-variable missingness report
    print(f"  {'Variable':<20} {'Missing':>10} {'% Missing':>12}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 12}")

    total_rows = len(df)
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        pct_missing = (n_missing / total_rows) * 100
        flag = "  <- HIGH" if pct_missing > 40 else ""
        print(f"  {col:<20} {n_missing:>10,} {pct_missing:>11.1f}%{flag}")

    total_missing = int(df.isna().sum().sum())
    total_cells = total_rows * len(df.columns)
    print(f"\n  Total missing cells : {total_missing:,} / {total_cells:,} "
          f"({(total_missing / total_cells) * 100:.1f}%)")
    print(separator)


# ---------------------------------------------------------------------------
# 5. Column presence checker (utility — used during development)
# ---------------------------------------------------------------------------

def list_all_columns(dataset_dir: Path, file_map: dict[str, str]) -> None:
    """
    Print every column available in every CSV file.
    Useful for exploring variables before editing config.py.

    Parameters
    ----------
    dataset_dir : Path
        Folder containing the CSV files.
    file_map : dict[str, str]
        Maps logical dataset name → CSV filename.
    """
    print("\n" + "=" * 65)
    print("  ALL AVAILABLE COLUMNS (exploration helper)")
    print("=" * 65)
    for logical_name, filename in file_map.items():
        path = dataset_dir / filename
        if path.exists():
            df_head = pd.read_csv(path, nrows=0)
            print(f"\n  {logical_name} ({filename})")
            print(f"  {list(df_head.columns)}")
        else:
            print(f"\n  [MISSING] {logical_name} ({filename}) — file not found")
