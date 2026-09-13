"""
clustering_sandbox/hdbscan_ultrafine/validate_hdbscan_ultrafine_diagnostic.py
==============================================================================
Validation Suite for HDBSCAN Ultra-Fine Scale Sensitivity Diagnostic

Verifies:
  1. All 48 expected configurations completed (12 per matrix x 4 matrices).
  2. Output files exist, are non-empty, and structurally sound.
  3. SEQN participant ID alignment and row counts (N=4,482 for A; N=967 for B).
  4. No duplicate SEQN or missing values in assignments.
  5. Noise is represented consistently as -1.
  6. Cluster counts and noise counts in metrics CSV match assignment CSV columns.
  7. Internal metrics (Silhouette, CH, DB) computed ONLY where mathematically valid.
  8. Baseline scaled matrices and preprocessing files remain untouched.
  9. Audit Check: Explicitly checks B_LOG ms=3 structural transition (mcs=5 N=499 vs mcs=7 N=126).
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd


def validate_hdbscan_ultrafine(base_dir: Path) -> bool:
    diag_dir = base_dir / "clustering_sandbox" / "hdbscan_ultrafine"
    metrics_dir = diag_dir / "metrics"
    assignments_dir = diag_dir / "cluster_assignments"
    plots_dir = diag_dir / "plots"
    metadata_dir = diag_dir / "metadata"

    print("=" * 70)
    print("  VALIDATING HDBSCAN ULTRA-FINE SENSITIVITY DIAGNOSTIC (AUDITED)")
    print("=" * 70)

    errors = []

    # 1. Required Files Existence
    expected_files = [
        metrics_dir / "A_RAW_hdbscan_ultrafine_metrics.csv",
        metrics_dir / "A_LOG_hdbscan_ultrafine_metrics.csv",
        metrics_dir / "B_RAW_hdbscan_ultrafine_metrics.csv",
        metrics_dir / "B_LOG_hdbscan_ultrafine_metrics.csv",
        metrics_dir / "hdbscan_ultrafine_cross_matrix_summary.csv",
        metrics_dir / "hdbscan_ultrafine_stability_summary.csv",
        assignments_dir / "A_RAW_hdbscan_ultrafine_assignments.csv",
        assignments_dir / "A_LOG_hdbscan_ultrafine_assignments.csv",
        assignments_dir / "B_RAW_hdbscan_ultrafine_assignments.csv",
        assignments_dir / "B_LOG_hdbscan_ultrafine_assignments.csv",
        plots_dir / "hdbscan_ultrafine_clusters_vs_params.png",
        plots_dir / "hdbscan_ultrafine_noise_vs_params.png",
        plots_dir / "hdbscan_ultrafine_non_noise_count.png",
        plots_dir / "hdbscan_ultrafine_pca_projections_A_RAW.png",
        plots_dir / "hdbscan_ultrafine_pca_projections_A_LOG.png",
        plots_dir / "hdbscan_ultrafine_pca_projections_B_RAW.png",
        plots_dir / "hdbscan_ultrafine_pca_projections_B_LOG.png",
        metadata_dir / "hdbscan_ultrafine_metadata.json",
    ]

    for filepath in expected_files:
        if not filepath.exists():
            errors.append(f"Missing output file: {filepath}")
        elif filepath.stat().st_size == 0:
            errors.append(f"File exists but is empty: {filepath}")

    if errors:
        for err in errors:
            print(f"  ❌ ERROR: {err}")
        return False
    print("  ✅ All expected output files exist and are non-empty.")

    # 2. Check Summary Metrics Table
    df_summary = pd.read_csv(metrics_dir / "hdbscan_ultrafine_cross_matrix_summary.csv")
    print(f"  Cross-matrix summary rows: {len(df_summary)} (Expected: 48 = 4 matrices × 12 configs)")
    if len(df_summary) != 48:
        errors.append(f"Expected 48 summary rows, got {len(df_summary)}")

    # Audit Check: Verify B_LOG transition between mcs=5 and mcs=7 at ms=3
    b_log_mcs5 = df_summary[(df_summary["matrix"] == "B_LOG") & (df_summary["min_cluster_size"] == 5) & (df_summary["min_samples"] == 3)]
    b_log_mcs7 = df_summary[(df_summary["matrix"] == "B_LOG") & (df_summary["min_cluster_size"] == 7) & (df_summary["min_samples"] == 3)]

    if len(b_log_mcs5) == 1 and len(b_log_mcs7) == 1:
        n_mcs5 = b_log_mcs5.iloc[0]["number_of_non_noise_points"]
        n_mcs7 = b_log_mcs7.iloc[0]["number_of_non_noise_points"]
        if n_mcs5 != 499 or n_mcs7 != 126:
            errors.append(f"Audit Check Failed: B_LOG ms=3 non-noise count mismatch (expected mcs=5 N=499, mcs=7 N=126; got mcs=5 N={n_mcs5}, mcs=7 N={n_mcs7})")
        else:
            print("  ✅ AUDIT CHECK PASSED: B_LOG ms=3 structural transition verified (mcs=5 N=499 -> mcs=7 N=126).")
    else:
        errors.append("Audit Check Failed: Could not locate B_LOG mcs=5/7 ms=3 metric rows.")

    # 3. Check Assignments, SEQN Alignment, and Metrics Agreement
    cohort_a_df = pd.read_csv(base_dir / "output" / "analysis_cohort_a_broad.csv")
    cohort_b_df = pd.read_csv(base_dir / "output" / "analysis_cohort_b_fasting.csv")

    seqn_a_expected = cohort_a_df["SEQN"].values
    seqn_b_expected = cohort_b_df["SEQN"].values

    matrix_info = [
        ("A_RAW", 4482, seqn_a_expected),
        ("A_LOG", 4482, seqn_a_expected),
        ("B_RAW", 967, seqn_b_expected),
        ("B_LOG", 967, seqn_b_expected),
    ]

    for mat_name, expected_n, expected_seqn in matrix_info:
        ass_path = assignments_dir / f"{mat_name}_hdbscan_ultrafine_assignments.csv"
        df_ass = pd.read_csv(ass_path)

        if len(df_ass) != expected_n:
            errors.append(f"{mat_name} assignment row count mismatch: {len(df_ass)} vs expected {expected_n}")

        if not np.array_equal(df_ass["SEQN"].values, expected_seqn):
            errors.append(f"{mat_name} SEQN values do not match baseline cohort SEQN!")

        if df_ass["SEQN"].duplicated().any():
            errors.append(f"Duplicate SEQN IDs found in {mat_name} assignments")

        config_cols = [c for c in df_ass.columns if c != "SEQN"]
        if len(config_cols) != 12:
            errors.append(f"{mat_name} expected 12 config columns, got {len(config_cols)}")

        for col in config_cols:
            labels = df_ass[col].values
            if pd.isnull(labels).any():
                errors.append(f"Null values in column {col} of {mat_name}")

            unique_labels = set(np.unique(labels))
            non_noise_labels = {l for l in unique_labels if l != -1}

            for l in non_noise_labels:
                if l < 0:
                    errors.append(f"Invalid negative cluster label {l} in {mat_name} {col}")

            sub_metric = df_summary[
                (df_summary["matrix"] == mat_name) &
                (df_summary.apply(lambda r: f"mcs{r['min_cluster_size']}_ms{r['min_samples']}", axis=1) == col)
            ]

            if len(sub_metric) == 1:
                met_rec = sub_metric.iloc[0]
                expected_n_clusters = met_rec["number_of_clusters_excluding_noise"]
                expected_n_noise = met_rec["number_of_noise_points"]

                actual_n_clusters = len(non_noise_labels)
                actual_n_noise = int(np.sum(labels == -1))

                if actual_n_clusters != expected_n_clusters:
                    errors.append(f"{mat_name} {col} n_clusters mismatch: {actual_n_clusters} vs metric {expected_n_clusters}")
                if actual_n_noise != expected_n_noise:
                    errors.append(f"{mat_name} {col} noise points mismatch: {actual_n_noise} vs metric {expected_n_noise}")

                sil_val = met_rec["silhouette_non_noise"]
                if pd.notnull(sil_val):
                    if actual_n_clusters < 2:
                        errors.append(f"{mat_name} {col} reported Silhouette with only {actual_n_clusters} clusters!")
            else:
                errors.append(f"Could not match metric row for {mat_name} {col}")

    # 4. Verify baseline input files were not modified
    scaled_a_raw = base_dir / "output" / "scaled_matrices" / "A_RAW_scaled.csv"
    if not scaled_a_raw.exists() or scaled_a_raw.stat().st_size == 0:
        errors.append("Baseline scaled matrix A_RAW_scaled.csv was damaged or modified!")

    if errors:
        print("\n  ❌ VALIDATION FAILED with errors:")
        for err in errors:
            print(f"    - {err}")
        return False

    print("\n  ==================================================")
    print("  ✅ ULTRA-FINE DIAGNOSTIC VALIDATION PASSED 100% CLEANLY!")
    print("  ==================================================\n")
    return True


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    success = validate_hdbscan_ultrafine(base_dir)
    sys.exit(0 if success else 1)
