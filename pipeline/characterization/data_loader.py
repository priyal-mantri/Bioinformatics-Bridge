"""
Data loader module for Stage 8 phenotype characterization.
Loads cohort datasets from master output/ and operational cluster assignments
from origin/experiment/clustering-sandbox via git show.
"""

import io
import os
import subprocess
import pandas as pd
from typing import Dict, List, Tuple, Any

SANDBOX_BRANCH = "origin/experiment/clustering-sandbox"

COHORT_A_PRIMARY_FEATURES = [
    "BMXBMI", "BMXWAIST", "Avg_Systolic_BP", "Avg_Diastolic_BP", "BPXPLS",
    "LBXTC", "LBDHDD", "LBXSTR", "LBXSATSI", "LBXSAL", "LBXSTP", "LBXSTB",
    "LBXSCR", "LBXSUA", "LBXSBU", "LBXSCA", "LBXSPH", "LBXSNASI", "LBXSKSI"
]

COHORT_B_PRIMARY_FEATURES = COHORT_A_PRIMARY_FEATURES + [
    "DXDTOBMD", "DXDTOPF", "DXDTOLE", "LBXGLU", "LBXIN"
]

DERIVED_FEATURES = ["HOMA_IR", "TC_HDL_ratio", "TG_HDL_ratio"]
CONTEXTUAL_FEATURES = ["RIDAGEYR", "RIAGENDR", "RIDRETH3"]


def load_git_assignment(git_relative_path: str) -> pd.DataFrame:
    """Reads a cluster assignment CSV directly from the sandbox git branch."""
    cmd = ["git", "show", f"{SANDBOX_BRANCH}:{git_relative_path}"]
    try:
        raw_bytes = subprocess.check_output(cmd, stderr=subprocess.PIPE)
        output_str = raw_bytes.decode("utf-8")
        df = pd.read_csv(io.StringIO(output_str))
        if "SEQN" in df.columns:
            df["SEQN"] = df["SEQN"].astype(float)
        return df
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to load git assignment file '{git_relative_path}' "
            f"from branch '{SANDBOX_BRANCH}'. Error: {e.stderr.decode('utf-8')}"
        )


def load_master_cohort(cohort_key: str) -> pd.DataFrame:
    """Loads master analysis dataset for Cohort A or Cohort B."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if cohort_key.upper() == "A":
        filepath = os.path.join(repo_root, "output", "analysis_cohort_a_broad.csv")
    elif cohort_key.upper() == "B":
        filepath = os.path.join(repo_root, "output", "analysis_cohort_b_fasting.csv")
    else:
        raise ValueError(f"Unknown cohort key '{cohort_key}'. Must be 'A' or 'B'.")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Master cohort file not found: {filepath}")

    df = pd.read_csv(filepath)
    df["SEQN"] = df["SEQN"].astype(float)
    return df


def load_characterization_dataset(
    cohort_key: str, representation: str = "RAW"
) -> Tuple[pd.DataFrame, List[str], List[str], List[str], Dict[str, Any]]:
    """
    Loads master cohort dataset and joins operational K-Means cluster assignments
    retained in Decision 014 (K=2 for Cohort A, K=3 for Cohort B).

    Returns:
        merged_df: Combined dataframe with cluster assignment column ('cluster')
        primary_features: List of primary clustering features
        derived_features: List of post-hoc derived features
        contextual_features: List of post-hoc demographic features
        validation_info: Audit dictionary of sample size and join verification
    """
    cohort_upper = cohort_key.upper()
    rep_upper = representation.upper()

    master_df = load_master_cohort(cohort_upper)
    n_master = len(master_df)

    if cohort_upper == "A":
        assign_path = f"clustering_sandbox/kmeans/cluster_assignments/A_{rep_upper}_kmeans_assignments.csv"
        target_col = "K2_cluster"
        retained_k = 2
        primary_features = COHORT_A_PRIMARY_FEATURES
    else:
        assign_path = f"clustering_sandbox/kmeans/cluster_assignments/B_{rep_upper}_kmeans_assignments.csv"
        target_col = "K3_cluster"
        retained_k = 3
        primary_features = COHORT_B_PRIMARY_FEATURES

    assign_df = load_git_assignment(assign_path)
    n_assign = len(assign_df)

    if target_col not in assign_df.columns:
        raise KeyError(f"Expected column '{target_col}' not found in '{assign_path}'.")

    # Merge on SEQN
    merged_df = master_df.merge(
        assign_df[["SEQN", target_col]], on="SEQN", how="inner"
    )
    merged_df = merged_df.rename(columns={target_col: "cluster"})
    merged_df["cluster"] = merged_df["cluster"].astype(int)

    n_merged = len(merged_df)
    if n_merged != n_master or n_merged != n_assign:
        raise ValueError(
            f"Join mismatch for Cohort {cohort_upper}! Master rows: {n_master}, "
            f"Assign rows: {n_assign}, Merged rows: {n_merged}."
        )

    validation_info = {
        "cohort": cohort_upper,
        "representation": rep_upper,
        "retained_k": retained_k,
        "source_file": assign_path,
        "master_rows": n_master,
        "assign_rows": n_assign,
        "merged_rows": n_merged,
        "join_status": "SUCCESS_100_PERCENT_MATCH",
    }

    return merged_df, primary_features, DERIVED_FEATURES, CONTEXTUAL_FEATURES, validation_info


def load_all_assignments(cohort_key: str) -> Dict[str, pd.DataFrame]:
    """
    Loads all available model family cluster assignments from the sandbox branch
    for sensitivity and robustness analysis.
    """
    cohort_upper = cohort_key.upper()
    models = {
        "kmeans_RAW": f"clustering_sandbox/kmeans/cluster_assignments/{cohort_upper}_RAW_kmeans_assignments.csv",
        "kmeans_LOG": f"clustering_sandbox/kmeans/cluster_assignments/{cohort_upper}_LOG_kmeans_assignments.csv",
        "gmm_RAW": f"clustering_sandbox/gmm/assignments/{cohort_upper}_RAW_gmm_assignments.csv",
        "gmm_LOG": f"clustering_sandbox/gmm/assignments/{cohort_upper}_LOG_gmm_assignments.csv",
        "ward_RAW": f"clustering_sandbox/hierarchical/cluster_assignments/{cohort_upper}_RAW_hierarchical_assignments.csv",
        "ward_LOG": f"clustering_sandbox/hierarchical/cluster_assignments/{cohort_upper}_LOG_hierarchical_assignments.csv",
        "spectral_RAW": f"clustering_sandbox/spectral/cluster_assignments/{cohort_upper}_RAW_spectral_assignments.csv",
        "spectral_LOG": f"clustering_sandbox/spectral/cluster_assignments/{cohort_upper}_LOG_spectral_assignments.csv",
    }

    loaded_assignments = {}
    for key, path in models.items():
        try:
            df = load_git_assignment(path)
            loaded_assignments[key] = df
        except Exception as e:
            print(f"Warning: Could not load assignment '{key}' ({path}): {e}")

    return loaded_assignments
