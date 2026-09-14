"""
clustering_sandbox/hdbscan_diagnostics/validate_hdbscan_calibration_diagnostic.py
===================================================================================
Validation Script for HDBSCAN Parameter Calibration Diagnostic (Pre-Experiment 5)

Verifies:
  1. File existence and non-emptiness of all expected outputs.
  2. Metrics CSV row counts (12 configurations per matrix x 4 matrices = 48 rows).
  3. Completeness of metadata JSON.
  4. Valid noise proportions (0.0 to 1.0) and non-negative cluster counts.
"""

import sys
import json
from pathlib import Path
import pandas as pd


def validate_hdbscan_diagnostics(base_dir: Path) -> bool:
    diag_dir = base_dir / "clustering_sandbox" / "hdbscan_diagnostics"
    metrics_dir = diag_dir / "metrics"
    plots_dir = diag_dir / "plots"
    metadata_dir = diag_dir / "metadata"

    print("=" * 70)
    print("  VALIDATING HDBSCAN PARAMETER-CALIBRATION DIAGNOSTIC")
    print("=" * 70)

    errors = []

    expected_files = [
        metrics_dir / "diag1_hdbscan_calibration_summary.csv",
        metrics_dir / "A_RAW_hdbscan_calibration.csv",
        metrics_dir / "A_LOG_hdbscan_calibration.csv",
        metrics_dir / "B_RAW_hdbscan_calibration.csv",
        metrics_dir / "B_LOG_hdbscan_calibration.csv",
        plots_dir / "hdbscan_diag1_n_clusters_vs_params.png",
        plots_dir / "hdbscan_diag2_noise_proportion_vs_params.png",
        plots_dir / "hdbscan_diag4_pca_projections_rep.png",
        metadata_dir / "hdbscan_calibration_metadata.json",
    ]

    for filepath in expected_files:
        if not filepath.exists():
            errors.append(f"Missing output file: {filepath}")
        elif filepath.stat().st_size == 0:
            errors.append(f"Output file is empty: {filepath}")

    if errors:
        for err in errors:
            print(f"  ❌ ERROR: {err}")
        return False
    print("  ✅ All expected output files exist and are non-empty.")

    # Check Summary Metrics CSV
    df_sum = pd.read_csv(metrics_dir / "diag1_hdbscan_calibration_summary.csv")
    print(f"  Summary rows: {len(df_sum)} (Expected: 48 = 4 matrices × 12 configs)")
    if len(df_sum) != 48:
        errors.append(f"Expected 48 summary rows, got {len(df_sum)}")

    for idx, row in df_sum.iterrows():
        if not (0.0 <= row["noise_proportion"] <= 1.0):
            errors.append(f"Row {idx} invalid noise_proportion: {row['noise_proportion']}")
        if row["n_clusters_found"] < 0:
            errors.append(f"Row {idx} negative n_clusters_found: {row['n_clusters_found']}")

    # Check Metadata JSON
    with open(metadata_dir / "hdbscan_calibration_metadata.json", "r") as f:
        meta = json.load(f)

    if meta["total_configurations_tested"] != 48:
        errors.append(f"Metadata total configurations mismatch: {meta['total_configurations_tested']}")

    if errors:
        print("\n  ❌ VALIDATION FAILED with errors:")
        for err in errors:
            print(f"    - {err}")
        return False

    print("\n  ==================================================")
    print("  ✅ HDBSCAN DIAGNOSTIC VALIDATION PASSED 100% CLEANLY!")
    print("  ==================================================\n")
    return True


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    success = validate_hdbscan_diagnostics(base_dir)
    sys.exit(0 if success else 1)
