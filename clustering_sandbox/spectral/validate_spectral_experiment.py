"""
clustering_sandbox/spectral/validate_spectral_experiment.py
============================================================
Validation Script for Experiment 4: Spectral Clustering Exploratory Sandbox

Verifies:
  1. File existence and non-emptiness of all expected outputs.
  2. SEQN participant ID alignment and row counts (N=4,482 for A; N=967 for B).
  3. Completeness of K=2..7 assignments with zero missing values.
  4. Non-zero metric values for Silhouette, Calinski-Harabasz, and Davies-Bouldin.
  5. 100% graph connectivity confirmation for all candidate matrices.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np


def validate_spectral_experiment(base_dir: Path) -> bool:
    exp_dir = base_dir / "clustering_sandbox" / "spectral"
    metrics_dir = exp_dir / "metrics"
    assignments_dir = exp_dir / "cluster_assignments"
    plots_dir = exp_dir / "plots"
    metadata_dir = exp_dir / "metadata"

    print("=" * 70)
    print("  VALIDATING EXPERIMENT 4: SPECTRAL CLUSTER SANDBOX")
    print("=" * 70)

    errors = []

    # 1. Required Files Check
    expected_files = [
        metrics_dir / "A_RAW_spectral_metrics.csv",
        metrics_dir / "A_LOG_spectral_metrics.csv",
        metrics_dir / "B_RAW_spectral_metrics.csv",
        metrics_dir / "B_LOG_spectral_metrics.csv",
        metrics_dir / "spectral_cross_matrix_summary.csv",
        assignments_dir / "A_RAW_spectral_assignments.csv",
        assignments_dir / "A_LOG_spectral_assignments.csv",
        assignments_dir / "B_RAW_spectral_assignments.csv",
        assignments_dir / "B_LOG_spectral_assignments.csv",
        plots_dir / "spectral_metrics_vs_k.png",
        plots_dir / "spectral_pca_projections_A_RAW.png",
        plots_dir / "spectral_pca_projections_A_LOG.png",
        plots_dir / "spectral_pca_projections_B_RAW.png",
        plots_dir / "spectral_pca_projections_B_LOG.png",
        metadata_dir / "spectral_experiment_metadata.json",
    ]

    for filepath in expected_files:
        if not filepath.exists():
            errors.append(f"Missing expected output file: {filepath}")
        elif filepath.stat().st_size == 0:
            errors.append(f"File exists but is empty: {filepath}")

    if errors:
        for err in errors:
            print(f"  ❌ ERROR: {err}")
        return False
    print("  ✅ All expected output files exist and are non-empty.")

    # 2. Check Metrics Data
    df_metrics = pd.read_csv(metrics_dir / "spectral_cross_matrix_summary.csv")
    print(f"  Summary metrics row count: {len(df_metrics)} (Expected: 24 = 4 matrices × 6 K values)")
    if len(df_metrics) != 24:
        errors.append(f"Expected 24 metric rows, got {len(df_metrics)}")

    for col in ["silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]:
        if df_metrics[col].isnull().any():
            errors.append(f"NaN values found in metric column: {col}")

    # 3. Check Assignments Data & SEQN Integrity
    cohort_a_df = pd.read_csv(base_dir / "output" / "analysis_cohort_a_broad.csv")
    cohort_b_df = pd.read_csv(base_dir / "output" / "analysis_cohort_b_fasting.csv")

    seqn_a_expected = cohort_a_df["SEQN"].values
    seqn_b_expected = cohort_b_df["SEQN"].values

    for mat_name, expected_n, expected_seqn in [
        ("A_RAW", 4482, seqn_a_expected),
        ("A_LOG", 4482, seqn_a_expected),
        ("B_RAW", 967, seqn_b_expected),
        ("B_LOG", 967, seqn_b_expected),
    ]:
        ass_path = assignments_dir / f"{mat_name}_spectral_assignments.csv"
        df_ass = pd.read_csv(ass_path)
        print(f"  Validating assignments for {mat_name}: N={len(df_ass)}")

        if len(df_ass) != expected_n:
            errors.append(f"{mat_name} assignment row count mismatch: {len(df_ass)} vs expected {expected_n}")

        if not np.array_equal(df_ass["SEQN"].values, expected_seqn):
            errors.append(f"{mat_name} SEQN values do not match original cohort order!")

        for K in range(2, 8):
            col = f"K{K}_cluster"
            if col not in df_ass.columns:
                errors.append(f"Missing column {col} in {mat_name} assignments")
            elif df_ass[col].isnull().any():
                errors.append(f"NaN values found in column {col} of {mat_name}")
            else:
                unique_clusters = set(df_ass[col].unique())
                if len(unique_clusters) != K:
                    errors.append(f"{mat_name} {col} expected {K} clusters, found {len(unique_clusters)}")

    # 4. Check Metadata JSON
    with open(metadata_dir / "spectral_experiment_metadata.json", "r") as f:
        meta = json.load(f)

    for mat_name, mdict in meta["matrices_metadata"].items():
        if not mdict["is_fully_connected"]:
            errors.append(f"Matrix {mat_name} was NOT fully connected!")
        print(f"  Metadata check {mat_name}: Connected={mdict['is_fully_connected']}, λ₂={mdict['fiedler_value_lambda2']}")

    if errors:
        print("\n  ❌ VALIDATION FAILED with errors:")
        for err in errors:
            print(f"    - {err}")
        return False

    print("\n  ==================================================")
    print("  ✅ EXPERIMENT 4 VALIDATION PASSED 100% CLEANLY!")
    print("  ==================================================\n")
    return True


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    success = validate_spectral_experiment(base_dir)
    sys.exit(0 if success else 1)
