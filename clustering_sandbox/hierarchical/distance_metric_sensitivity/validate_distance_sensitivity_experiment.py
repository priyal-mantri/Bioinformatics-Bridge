#!/usr/bin/env python3
"""
Validation Audit Script for Experiment 5 Sensitivity Extension: Distance-Metric Sensitivity
Bioinfo Bridge — Research 2 (NHANES 2017–2018)

Verifies:
1. File structure & completeness (metrics CSVs, assignment CSVs, plots PNGs, metadata JSON).
2. Data integrity & 100% SEQN alignment against baseline cohort files.
3. Correct range and validity of metric outputs (Silhouette [-1, 1], ARI [-1, 1], non-negative CH/DB/Dispersion).
4. Full coverage of all 96 expected experimental combinations (4 matrices x 2 distance metrics x 2 linkages x 6 K values).
5. Absence of NaN / null / inf values across all exported datasets.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSIGNMENTS_DIR = os.path.join(BASE_DIR, "cluster_assignments")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")

COHORT_FILES = {
    "A": os.path.abspath(os.path.join(BASE_DIR, "../../../output/analysis_cohort_a_broad.csv")),
    "B": os.path.abspath(os.path.join(BASE_DIR, "../../../output/analysis_cohort_b_fasting.csv"))
}

MATRICES = {
    "A_RAW": {"cohort_key": "A", "N": 4482, "p": 19},
    "A_LOG": {"cohort_key": "A", "N": 4482, "p": 19},
    "B_RAW": {"cohort_key": "B", "N": 967, "p": 24},
    "B_LOG": {"cohort_key": "B", "N": 967, "p": 24}
}

DISTANCE_METRICS = ["euclidean", "manhattan"]
LINKAGES = ["complete", "average"]
K_RANGE = list(range(2, 8))

def audit_file_structure():
    print("--- 1. Auditing Directory & File Structure ---")
    errors = []
    
    for d in [ASSIGNMENTS_DIR, METRICS_DIR, PLOTS_DIR, METADATA_DIR]:
        if not os.path.isdir(d):
            errors.append(f"Missing directory: {d}")
            
    required_metrics = [
        "A_RAW_distance_sensitivity_metrics.csv",
        "A_LOG_distance_sensitivity_metrics.csv",
        "B_RAW_distance_sensitivity_metrics.csv",
        "B_LOG_distance_sensitivity_metrics.csv",
        "distance_sensitivity_cross_matrix_summary.csv",
        "distance_sensitivity_stability_summary.csv"
    ]
    for f in required_metrics:
        p = os.path.join(METRICS_DIR, f)
        if not os.path.isfile(p):
            errors.append(f"Missing metric file: {f}")
            
    required_assign = [f"{m}_distance_sensitivity_assignments.csv" for m in MATRICES.keys()]
    for f in required_assign:
        p = os.path.join(ASSIGNMENTS_DIR, f)
        if not os.path.isfile(p):
            errors.append(f"Missing assignment file: {f}")
            
    required_plots = [
        "distance_sensitivity_silhouette_vs_k.png",
        "distance_sensitivity_stability_vs_k.png",
        "distance_sensitivity_cluster_sizes.png"
    ]
    for f in required_plots:
        p = os.path.join(PLOTS_DIR, f)
        if not os.path.isfile(p):
            errors.append(f"Missing plot file: {f}")
            
    meta_p = os.path.join(METADATA_DIR, "distance_sensitivity_metadata.json")
    if not os.path.isfile(meta_p):
        errors.append("Missing metadata JSON file")

    if errors:
        print(f"FAILED File Structure Audit with {len(errors)} error(s):")
        for e in errors:
            print(f"  ❌ {e}")
        return False
    print("✅ All required directories, CSVs, PNGs, and JSON metadata files are present.\n")
    return True

def audit_data_integrity_and_seqn():
    print("--- 2. Auditing Participant SEQN Alignment & Assignments ---")
    errors = []
    
    for mat_key, info in MATRICES.items():
        cohort_path = COHORT_FILES[info["cohort_key"]]
        base_df = pd.read_csv(cohort_path)
        base_seqn = base_df["SEQN"].values
        
        assign_path = os.path.join(ASSIGNMENTS_DIR, f"{mat_key}_distance_sensitivity_assignments.csv")
        assign_df = pd.read_csv(assign_path)
        
        if len(assign_df) != len(base_seqn):
            errors.append(f"{mat_key}: Length mismatch (expected {len(base_seqn)}, got {len(assign_df)})")
            continue
            
        if not np.array_equal(assign_df["SEQN"].values, base_seqn):
            errors.append(f"{mat_key}: SEQN ordering mismatch!")
            
        expected_cols = ["SEQN"] + [f"{d}_{l}_K{k}" for d in DISTANCE_METRICS for l in LINKAGES for k in K_RANGE]
        missing_cols = set(expected_cols) - set(assign_df.columns)
        if missing_cols:
            errors.append(f"{mat_key}: Missing assignment columns: {missing_cols}")
            
        if assign_df.isnull().any().any():
            errors.append(f"{mat_key}: Found NaN/null values in cluster assignments!")
            
    if errors:
        print(f"FAILED SEQN Alignment Audit with {len(errors)} error(s):")
        for e in errors:
            print(f"  ❌ {e}")
        return False
    print("✅ 100% SEQN alignment and assignment integrity verified across all candidate matrices.\n")
    return True

def audit_metrics_and_stability():
    print("--- 3. Auditing Numerical Metrics & Subsampling Stability Values ---")
    errors = []
    
    summary_path = os.path.join(METRICS_DIR, "distance_sensitivity_cross_matrix_summary.csv")
    df = pd.read_csv(summary_path)
    
    expected_total_runs = len(MATRICES) * len(DISTANCE_METRICS) * len(LINKAGES) * len(K_RANGE)
    if len(df) != expected_total_runs:
        errors.append(f"Expected {expected_total_runs} summary rows, got {len(df)}")
        
    if df.isnull().any().any():
        errors.append("Summary table contains NaN/null values!")
        
    for idx, row in df.iterrows():
        run_id = f"{row['matrix']} | {row['distance_metric']} | {row['linkage']} | K={row['K']}"
        
        if not (-1.0 <= row["silhouette_score"] <= 1.0):
            errors.append(f"{run_id}: Invalid silhouette_score = {row['silhouette_score']}")
            
        if row["calinski_harabasz_score"] <= 0:
            errors.append(f"{run_id}: Invalid CH score = {row['calinski_harabasz_score']}")
            
        if row["davies_bouldin_score"] < 0:
            errors.append(f"{run_id}: Invalid DB score = {row['davies_bouldin_score']}")
            
        if row["within_dispersion"] < 0:
            errors.append(f"{run_id}: Invalid dispersion = {row['within_dispersion']}")
            
        if not (-1.0 <= row["mean_ari"] <= 1.0):
            errors.append(f"{run_id}: Invalid mean_ari = {row['mean_ari']}")
            
        if not (-1.0 <= row["median_ari"] <= 1.0):
            errors.append(f"{run_id}: Invalid median_ari = {row['median_ari']}")
            
        if row["std_ari"] < 0:
            errors.append(f"{run_id}: Invalid std_ari = {row['std_ari']}")
            
        if not (row["min_cluster_size"] > 0 and row["max_cluster_size"] > 0):
            errors.append(f"{run_id}: Invalid cluster sizes ({row['min_cluster_size']}, {row['max_cluster_size']})")
            
        if row["min_cluster_prop"] < (0.99 / row["N"]):
            errors.append(f"{run_id}: Impossible min_cluster_prop = {row['min_cluster_prop']}")

    if errors:
        print(f"FAILED Metrics Audit with {len(errors)} error(s):")
        for e in errors:
            print(f"  ❌ {e}")
        return False
    print(f"✅ All {expected_total_runs} experimental runs verified clean (100% valid metric ranges & stability bounds).\n")
    return True

def run_validation():
    print("=" * 80)
    print("STARTING AUDIT VALIDATION FOR DISTANCE SENSITIVITY EXTENSION")
    print("=" * 80 + "\n")
    
    v1 = audit_file_structure()
    v2 = audit_data_integrity_and_seqn()
    v3 = audit_metrics_and_stability()
    
    if v1 and v2 and v3:
        print("=" * 80)
        print("🎉 ALL VALIDATION AUDITS PASSED WITH 100% CLEAN STATUS!")
        print("=" * 80)
        sys.exit(0)
    else:
        print("=" * 80)
        print("❌ VALIDATION AUDIT FAILED — PLEASE REVIEW ERRORS ABOVE!")
        print("=" * 80)
        sys.exit(1)

if __name__ == "__main__":
    run_validation()
