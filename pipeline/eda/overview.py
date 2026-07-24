"""
pipeline/eda/overview.py
========================
Phase 1 -- Dataset Overview

Analyses the structural properties of the preprocessed dataset:
shape, data types, memory usage, and variable classification.

Saves:
    overview/overview_summary.txt
"""

import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Variable classifications (keep in sync with pipeline/config.py intent)
# ---------------------------------------------------------------------------

# These are flag/coding columns -- not continuous biomarkers
FLAG_COLS = {"DXAEXSTS", "LBDINLC"}

# These are demographic/control variables (not in the feature matrix)
DEMOGRAPHIC_COLS = {"SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3"}

# Raw repeated readings (retained for traceability; averaged versions used)
RAW_REPEATED_BP = {"BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3"}


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_overview(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Phase 1: Dataset Overview.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed dataset.
    output_dir : Path
        Root EDA output directory. Saves into output_dir/overview/.

    Returns
    -------
    dict
        Key findings for EDA_SUMMARY.md.
    """
    save_dir = output_dir / "overview"
    save_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 65
    lines = []

    def log(text=""):
        lines.append(text)
        print(text)

    log(sep)
    log("  PHASE 1 -- DATASET OVERVIEW")
    log(sep)

    # ── Shape ────────────────────────────────────────────────────────────────
    n_rows, n_cols = df.shape
    log(f"\n  Shape             : {n_rows:,} rows x {n_cols} columns")
    log(f"  Total cells       : {n_rows * n_cols:,}")

    # ── Memory ───────────────────────────────────────────────────────────────
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    log(f"  Memory usage      : {mem_mb:.2f} MB")

    # ── Data types ───────────────────────────────────────────────────────────
    dtype_counts = df.dtypes.value_counts()
    log(f"\n  Data types:")
    for dtype, count in dtype_counts.items():
        log(f"    {str(dtype):<12} : {count} columns")

    # ── Variable classification ───────────────────────────────────────────────
    all_cols = set(df.columns)
    numerical_cols    = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                         and c not in FLAG_COLS]
    flag_cols_present = [c for c in df.columns if c in FLAG_COLS]
    demo_cols_present = [c for c in df.columns if c in DEMOGRAPHIC_COLS]
    raw_bp_present    = [c for c in df.columns if c in RAW_REPEATED_BP]
    feature_cols      = [c for c in numerical_cols
                         if c not in DEMOGRAPHIC_COLS
                         and c not in RAW_REPEATED_BP]

    log(f"\n  Variable classification:")
    log(f"    Demographic / control : {len(demo_cols_present)}  "
        f"{demo_cols_present}")
    log(f"    Raw repeated BP cols  : {len(raw_bp_present)}  "
        f"(retained for traceability; averaged versions are used)")
    log(f"    Flag / coding cols    : {len(flag_cols_present)}  "
        f"{flag_cols_present}")
    log(f"    Feature columns       : {len(feature_cols)}  (for clustering)")

    log(f"\n  Feature columns ({len(feature_cols)} total):")
    for col in feature_cols:
        log(f"    {col}")

    # ── Basic statistics snapshot ────────────────────────────────────────────
    log(f"\n  Descriptive statistics (feature columns):")
    desc = df[feature_cols].describe().T
    desc_str = desc[["count", "mean", "std", "min", "max"]].round(3).to_string()
    log(desc_str)

    # ── Save report ──────────────────────────────────────────────────────────
    report_path = save_dir / "overview_summary.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [SAVED] {report_path}")
    print(sep)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "mem_mb": round(mem_mb, 2),
        "n_feature_cols": len(feature_cols),
        "feature_cols": feature_cols,
        "n_demo_cols": len(demo_cols_present),
        "n_flag_cols": len(flag_cols_present),
    }
