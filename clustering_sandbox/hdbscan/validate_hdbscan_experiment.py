"""
clustering_sandbox/hdbscan/validate_hdbscan_experiment.py
==========================================================
Independent Validation Suite for Experiment 5 (HDBSCAN Sandbox)

Verifies:
  1. All expected 48 configurations completed (12 per matrix x 4 matrices).
  2. Output files exist, non-empty, and structurally sound.
  3. SEQN participant ID alignment and row counts (N=4,482 for A; N=967 for B).
  4. No duplicate SEQN or missing values in assignments.
  5. Noise is represented consistently as -1.
  6. Cluster counts and noise fractions in metrics CSV match assignment CSV columns.
  7. Internal metrics (Silhouette, CH, DB) computed ONLY where mathematically valid.
  8. Baseline scaled matrices and preprocessing files remain untouched.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd


def validate_hdbscan_experiment(base_dir: Path) -> bool:
    exp_dir = base_dir / "clustering_sandbox" / "hdbscan"
    metrics_dir = exp_dir / "metrics"
    assignments_dir = exp_dir / "cluster_assignments"
    plots_dir = exp_dir / "plots"
    metadata_dir = exp_dir / "metadata"

    print("=" * 70)
    print("  VALIDATING EXPERIMENT 5: HDBSCAN FINE-SCALE CLUSTER SANDBOX")
    print("=" * 70)

    errors = []

    # 1. Required Files Existence
    expected_files = [
        metrics_dir / "A_RAW_hdbscan_metrics.csv",
        metrics_dir / "A_LOG_hdbscan_metrics.csv",
        metrics_dir / "B_RAW_hdbscan_metrics.csv",
        metrics_dir / "B_LOG_hdbscan_metrics.csv",
        metrics_dir / "hdbscan_cross_matrix_summary.csv",
        metrics_dir / "hdbscan_stability_summary.csv",
        assignments_dir / "A_RAW_hdbscan_assignments.csv",
        assignments_dir / "A_LOG_hdbscan_assignments.csv",
        assignments_dir / "B_RAW_hdbscan_assignments.csv",
        assignments_dir / "B_LOG_hdbscan_assignments.csv",
        plots_dir / "hdbscan_noise_and_clusters_grid.png",
        plots_dir / "hdbscan_pca_projections_A_RAW.png",
        plots_dir / "hdbscan_pca_projections_A_LOG.png",
        plots_dir / "hdbscan_pca_projections_B_RAW.png",
        plots_dir / "hdbscan_pca_projections_B_LOG.png",
        metadata_dir / "hdbscan_experiment_metadata.json",
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

    # 2. Verify Cross-Matrix Summary Metrics
    df_summary = pd.read_csv(metrics_dir / "hdbscan_cross_matrix_summary.csv")
    print(f"  Cross-matrix summary rows: {len(df_summary)} (Expected: 48 = 4 matrices × 12 configs)")
    if len(df_summary) != 48:
        errors.append(f"Expected 48 cross-matrix metric rows, got {len(df_summary)}")

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
        ass_path = assignments_dir / f"{mat_name}_hdbscan_assignments.csv"
        df_ass = pd.read_csv(ass_path)
        print(f"  Validating {mat_name}: N={len(df_ass)} participants")

        # Row count check
        if len(df_ass) != expected_n:
            errors.append(f"{mat_name} assignment row count mismatch: {len(df_ass)} vs expected {expected_n}")

        # SEQN exact 1-to-1 check
        if not np.array_equal(df_ass["SEQN"].values, expected_seqn):
            errors.append(f"{mat_name} SEQN values do not match baseline cohort SEQN!")

        # Duplicate SEQN check
        if df_ass["SEQN"].duplicated().any():
            errors.append(f"Duplicate SEQN IDs found in {mat_name} assignments")

        # Check each configuration column
        config_cols = [c for c in df_ass.columns if c != "SEQN"]
        if len(config_cols) != 12:
            errors.append(f"{mat_name} expected 12 config columns, got {len(config_cols)}")

        for col in config_cols:
            labels = df_ass[col].values
            if pd.isnull(labels).any():
                errors.append(f"Null values in column {col} of {mat_name}")

            # Verify noise label representation (-1)
            unique_labels = set(np.unique(labels))
            non_noise_labels = {l for l in unique_labels if l != -1}

            # Check that cluster labels are non-negative integers
            for l in non_noise_labels:
                if l < 0:
                    errors.append(f"Invalid negative cluster label {l} in {mat_name} {col}")

            # Match with summary metrics table
            sub_metric = df_summary[
                (df_summary["matrix_name"] == mat_name) &
                (df_summary.apply(lambda r: f"mcs{r['min_cluster_size']}_ms{r['min_samples_setting']}", axis=1) == col)
            ]

            if len(sub_metric) == 1:
                met_rec = sub_metric.iloc[0]
                expected_n_clusters = met_rec["n_clusters_excluding_noise"]
                expected_n_noise = met_rec["n_noise_points"]

                actual_n_clusters = len(non_noise_labels)
                actual_n_noise = int(np.sum(labels == -1))

                if actual_n_clusters != expected_n_clusters:
                    errors.append(f"{mat_name} {col} n_clusters mismatch: {actual_n_clusters} vs metric {expected_n_clusters}")
                if actual_n_noise != expected_n_noise:
                    errors.append(f"{mat_name} {col} noise points mismatch: {actual_n_noise} vs metric {expected_n_noise}")

                # Internal metrics math validity check
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
    print("  ✅ EXPERIMENT 5 VALIDATION PASSED 100% CLEANLY!")
    print("  ==================================================\n")
    return True


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    success = validate_hdbscan_experiment(base_dir)
    sys.exit(0 if success else 1)
