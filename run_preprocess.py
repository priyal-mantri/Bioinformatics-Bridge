"""
run_preprocess.py
=================
Entry point for the NHANES preprocessing pipeline.

Reads output/merged_raw.csv, makes a working copy, then applies all
preprocessing decisions from the Decision Log in sequence.

Preprocessing chain (in order):
    Decision 002 -- Blood Pressure Averaging     (bp_averaging.py)
    Decision 003 -- Participant Age Filter        (filters.py)
    Decision 004 -- DEXA Scan Validity Filter     (filters.py)
    Decision 005 -- Insulin LOD Handling          (filters.py)
    Decision 006 -- Derived Feature Engineering   (derived_features.py)

The original merged_raw.csv is NEVER modified.
All changes are applied to a working copy and saved as preprocessed.csv.

Usage:
    python run_preprocess.py
"""

import shutil
import pandas as pd
from pathlib import Path

from pipeline.config import OUTPUT_DIR
from pipeline.preprocess.bp_averaging    import run_bp_averaging
from pipeline.preprocess.filters         import (
    apply_age_filter,
    apply_dexa_validity_filter,
    apply_insulin_lod_filter,
)
from pipeline.preprocess.derived_features import compute_derived_features


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MERGED_RAW_FILE     = OUTPUT_DIR / "merged_raw.csv"          # original -- never touched
WORKING_COPY_FILE   = OUTPUT_DIR / "merged_raw_working.csv"  # duplicate to work on
PREPROCESSED_FILE   = OUTPUT_DIR / "preprocessed.csv"        # final output


# ---------------------------------------------------------------------------
# Snapshot helper -- saves intermediate state after each decision
# ---------------------------------------------------------------------------

def _snapshot(df: pd.DataFrame, decision_num: str, label: str) -> None:
    """
    Save an intermediate snapshot of the dataframe after each decision.

    Files are saved as:
        output/snapshot_decision_XXX_<label>.csv

    This allows step-by-step inspection of what each filter changed.
    """
    path = OUTPUT_DIR / f"snapshot_decision_{decision_num}_{label}.csv"
    df.to_csv(path, index=False)
    print(f"  [snapshot] Saved -> {path.name}  "
          f"({len(df):,} rows x {df.shape[1]} cols)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:

    # ── Guard: merged_raw.csv must exist ────────────────────────────────────
    if not MERGED_RAW_FILE.exists():
        raise FileNotFoundError(
            f"[ERROR] {MERGED_RAW_FILE} not found.\n"
            f"        Run 'python run_pipeline.py' first."
        )

    print("\n" + "=" * 65)
    print("  NHANES PREPROCESSING PIPELINE")
    print("  Research 2: The Bioinformatics Bridge")
    print("=" * 65)

    # ── Step 0: Make a working copy of merged_raw.csv ───────────────────────
    # The original is never modified.  All decisions operate on this copy.
    shutil.copy2(MERGED_RAW_FILE, WORKING_COPY_FILE)
    print(f"\n  [COPY] {MERGED_RAW_FILE.name}")
    print(f"      -> {WORKING_COPY_FILE.name}  (working copy -- original preserved)")

    df = pd.read_csv(WORKING_COPY_FILE, low_memory=False)
    print(f"\n  Loaded working copy: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # ── Decision 002: Blood Pressure Averaging ───────────────────────────────
    print("\n" + "=" * 65)
    print("  DECISION 002 -- Blood Pressure Averaging")
    print("=" * 65)
    df = run_bp_averaging(df)          # adds Avg_Systolic_BP, Avg_Diastolic_BP
    _snapshot(df, "002", "bp_averaged")

    # ── Decision 003: Age Filter ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  DECISION 003 -- Participant Age Filter  (20 <= age <= 80)")
    print("=" * 65)
    df = apply_age_filter(df)
    _snapshot(df, "003", "age_filtered")

    # ── Decision 004: DEXA Validity Filter ──────────────────────────────────
    print("\n" + "=" * 65)
    print("  DECISION 004 -- DEXA Scan Validity Filter  (DXAEXSTS == 1)")
    print("=" * 65)
    df = apply_dexa_validity_filter(df)
    _snapshot(df, "004", "dexa_filtered")

    # ── Decision 005: Insulin Below-Detection-Limit ──────────────────────────
    print("\n" + "=" * 65)
    print("  DECISION 005 -- Insulin Below-Detection-Limit  (LBDINLC == 1)")
    print("=" * 65)
    df = apply_insulin_lod_filter(df)
    _snapshot(df, "005", "insulin_lod")

    # ── Decision 006: Derived Features ──────────────────────────────────────
    print("\n" + "=" * 65)
    print("  DECISION 006 -- Derived Feature Engineering")
    print("=" * 65)
    df = compute_derived_features(df)   # adds HOMA_IR, TC_HDL_ratio, TG_HDL_ratio
    _snapshot(df, "006", "derived_features")

    # ── Final save ───────────────────────────────────────────────────────────
    df.to_csv(PREPROCESSED_FILE, index=False)

    print("\n" + "=" * 65)
    print("  PREPROCESSING COMPLETE")
    print("=" * 65)
    print(f"  Original (untouched) : {MERGED_RAW_FILE.name}")
    print(f"  Working copy         : {WORKING_COPY_FILE.name}")
    print(f"  Final output         : {PREPROCESSED_FILE.name}")
    print(f"  Final shape          : {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Quick missingness overview of the final dataset
    print(f"\n  Missing values summary (final preprocessed.csv):")
    print(f"  {'Variable':<20} {'Missing':>10} {'% Missing':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*12}")
    for col in df.columns:
        n_miss = df[col].isna().sum()
        pct    = (n_miss / len(df)) * 100
        flag   = "  <- HIGH" if pct > 40 else ""
        print(f"  {col:<20} {n_miss:>10,} {pct:>11.1f}%{flag}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
