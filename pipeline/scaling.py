"""
pipeline/scaling.py
===================
Z-Score Standardization & Validation Module for Bioinformatics Bridge.

Standardizes the numerical scale of biological features across the four candidate
feature matrices independently:
  1. A_RAW  -> Z-score -> A_RAW_scaled  (4,482 x 19)
  2. A_LOG  -> Z-score -> A_LOG_scaled  (4,482 x 19)
  3. B_RAW  -> Z-score -> B_RAW_scaled  (967 x 24)
  4. B_LOG  -> Z-score -> B_LOG_scaled  (967 x 24)

FORMULA:
  z = (x - mean) / standard_deviation

STRICT RULES:
  - Fits each matrix independently (means and stds calculated solely from that matrix).
  - Preserves 100% of biological observations (no outlier deletion or winsorizing).
  - Preserves exact row counts, feature names, and participant order.
  - Source feature matrices and preprocessed datasets remain 100% unchanged.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def z_score_scale(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], dict[str, float]]:
    """
    Standardize a feature matrix using Z-score standardization (z = (x - mu) / sigma).

    Parameters
    ----------
    df : pd.DataFrame
        Input feature matrix (numeric values only).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, float], dict[str, float]]
        (scaled_df, means_dict, stds_dict)
    """
    means = df.mean(axis=0)
    stds = df.std(axis=0, ddof=0)  # Population standard deviation for exact std=1.0

    # Avoid division by zero if a feature has zero variance (not expected here)
    for col, std_val in stds.items():
        assert std_val > 0, f"[SCALING ERROR] Feature '{col}' has zero standard deviation!"

    scaled_df = (df - means) / stds

    means_dict = {col: float(means[col]) for col in df.columns}
    stds_dict = {col: float(stds[col]) for col in df.columns}

    return scaled_df, means_dict, stds_dict


def validate_scaled_matrix(
    name: str,
    scaled_df: pd.DataFrame,
    source_df: pd.DataFrame,
    expected_rows: int,
    expected_cols: int,
    means_dict: dict[str, float],
    stds_dict: dict[str, float]
) -> dict:
    """
    Perform complete read-only validation of a scaled feature matrix.
    """
    n_rows, n_cols = scaled_df.shape
    assert n_rows == expected_rows, f"[{name}] Expected {expected_rows} rows, got {n_rows}"
    assert n_cols == expected_cols, f"[{name}] Expected {expected_cols} cols, got {n_cols}"

    # 1. Missing and infinite values
    n_nulls = int(scaled_df.isna().sum().sum())
    assert n_nulls == 0, f"[{name}] Found {n_nulls} missing values! Expected 0."

    n_infs = int(np.isinf(scaled_df.values).sum())
    assert n_infs == 0, f"[{name}] Found {n_infs} infinite values!"

    # 2. Duplicate columns
    n_dupe_cols = len(scaled_df.columns) - len(set(scaled_df.columns))
    assert n_dupe_cols == 0, f"[{name}] Duplicate column names found!"

    # 3. Feature-wise mean and std checks
    col_validations = {}
    for col in scaled_df.columns:
        m_val = float(scaled_df[col].mean())
        s_val = float(scaled_df[col].std(ddof=0))
        mu_orig = means_dict[col]
        sigma_orig = stds_dict[col]

        assert abs(m_val) < 1e-6, f"[{name}] Column {col} mean ({m_val}) deviates from 0!"
        assert abs(s_val - 1.0) < 1e-6, f"[{name}] Column {col} std ({s_val}) deviates from 1!"

        # 4. Mathematical correspondence check: z == (x - mu) / sigma
        expected_z = (source_df[col] - mu_orig) / sigma_orig
        max_math_diff = float(np.abs(scaled_df[col] - expected_z).max())
        assert max_math_diff < 1e-6, f"[{name}] Math correspondence mismatch in {col}: max_diff={max_math_diff}"

        # 5. Rank preservation check (Spearman rho == 1.0)
        rho, _ = spearmanr(source_df[col], scaled_df[col])
        assert abs(rho - 1.0) < 1e-6, f"[{name}] Rank order altered in feature {col}! Rho={rho}"

        col_validations[col] = {
            "source_mean": mu_orig,
            "source_std": sigma_orig,
            "scaled_mean": m_val,
            "scaled_std": s_val,
            "max_math_diff": max_math_diff,
            "spearman_rank_rho": float(rho),
            "scaled_min": float(scaled_df[col].min()),
            "scaled_max": float(scaled_df[col].max()),
        }

    return {
        "matrix_name": name,
        "rows": n_rows,
        "cols": n_cols,
        "missing_values": n_nulls,
        "infinite_values": n_infs,
        "duplicate_cols": n_dupe_cols,
        "all_means_approx_zero": True,
        "all_stds_approx_one": True,
        "rank_preservation_confirmed": True,
        "math_correspondence_confirmed": True,
        "column_validations": col_validations,
    }


def run_scaling_pipeline(
    matrix_dir: Path,
    cohort_a_path: Path,
    cohort_b_path: Path,
    output_dir: Path
) -> dict:
    """
    Main orchestration function to standardize the 4 feature matrices independently.
    """
    sep = "=" * 65
    print(f"\n{sep}")
    print("  FEATURE-MATRIX SCALING & STANDARDIZATION (Z-Score)")
    print(sep)

    # 1. Load source candidate feature matrices
    file_a_raw = matrix_dir / "A_RAW.csv"
    file_a_log = matrix_dir / "A_LOG.csv"
    file_b_raw = matrix_dir / "B_RAW.csv"
    file_b_log = matrix_dir / "B_LOG.csv"

    a_raw = pd.read_csv(file_a_raw)
    a_log = pd.read_csv(file_a_log)
    b_raw = pd.read_csv(file_b_raw)
    b_log = pd.read_csv(file_b_log)

    print(f"  [Loaded Source Matrices]")
    print(f"    - A_RAW : {a_raw.shape[0]:,} rows x {a_raw.shape[1]} cols")
    print(f"    - A_LOG : {a_log.shape[0]:,} rows x {a_log.shape[1]} cols")
    print(f"    - B_RAW : {b_raw.shape[0]:,} rows x {b_raw.shape[1]} cols")
    print(f"    - B_LOG : {b_log.shape[0]:,} rows x {b_log.shape[1]} cols")

    # Load source cohorts for SEQN participant order verification
    df_cohort_a = pd.read_csv(cohort_a_path)
    df_cohort_b = pd.read_csv(cohort_b_path)
    assert len(df_cohort_a) == len(a_raw), "Cohort A row mismatch with A_RAW"
    assert len(df_cohort_b) == len(b_raw), "Cohort B row mismatch with B_RAW"

    # 2. Perform independent Z-score scaling
    a_raw_s, m_ar, s_ar = z_score_scale(a_raw)
    a_log_s, m_al, s_al = z_score_scale(a_log)
    b_raw_s, m_br, s_br = z_score_scale(b_raw)
    b_log_s, m_bl, s_bl = z_score_scale(b_log)

    # 3. Export scaled matrices
    output_dir.mkdir(parents=True, exist_ok=True)
    out_a_raw_s = output_dir / "A_RAW_scaled.csv"
    out_a_log_s = output_dir / "A_LOG_scaled.csv"
    out_b_raw_s = output_dir / "B_RAW_scaled.csv"
    out_b_log_s = output_dir / "B_LOG_scaled.csv"

    a_raw_s.to_csv(out_a_raw_s, index=False)
    a_log_s.to_csv(out_a_log_s, index=False)
    b_raw_s.to_csv(out_b_raw_s, index=False)
    b_log_s.to_csv(out_b_log_s, index=False)

    print(f"\n  [EXPORTED SCALED MATRICES]")
    print(f"    - {out_a_raw_s.relative_to(output_dir.parent.parent)}: {a_raw_s.shape[0]:,} rows x {a_raw_s.shape[1]} cols")
    print(f"    - {out_a_log_s.relative_to(output_dir.parent.parent)}: {a_log_s.shape[0]:,} rows x {a_log_s.shape[1]} cols")
    print(f"    - {out_b_raw_s.relative_to(output_dir.parent.parent)}: {b_raw_s.shape[0]:,} rows x {b_raw_s.shape[1]} cols")
    print(f"    - {out_b_log_s.relative_to(output_dir.parent.parent)}: {b_log_s.shape[0]:,} rows x {b_log_s.shape[1]} cols")

    # 4. Perform Validations
    val_a_raw = validate_scaled_matrix("A_RAW_scaled", a_raw_s, a_raw, 4482, 19, m_ar, s_ar)
    val_a_log = validate_scaled_matrix("A_LOG_scaled", a_log_s, a_log, 4482, 19, m_al, s_al)
    val_b_raw = validate_scaled_matrix("B_RAW_scaled", b_raw_s, b_raw, 967, 24, m_br, s_br)
    val_b_log = validate_scaled_matrix("B_LOG_scaled", b_log_s, b_log, 967, 24, m_bl, s_bl)

    metadata = {
        "pipeline_stage": "FEATURE_MATRIX_SCALING",
        "scaling_method": "Z-score Standardization (z = (x - mu) / sigma)",
        "scaling_formula": "z_j = (x_j - mu_j) / sigma_j",
        "scaling_fit_scope": "INDEPENDENT_PER_MATRIX",
        "cohort_a_primary": {
            "A_RAW_scaled": {
                "file": "output/scaled_matrices/A_RAW_scaled.csv",
                "participants": 4482,
                "features": 19,
                "missing_values": 0,
                "means": m_ar,
                "stds": s_ar,
            },
            "A_LOG_scaled": {
                "file": "output/scaled_matrices/A_LOG_scaled.csv",
                "participants": 4482,
                "features": 19,
                "missing_values": 0,
                "means": m_al,
                "stds": s_al,
            }
        },
        "cohort_b_secondary": {
            "B_RAW_scaled": {
                "file": "output/scaled_matrices/B_RAW_scaled.csv",
                "participants": 967,
                "features": 24,
                "missing_values": 0,
                "means": m_br,
                "stds": s_br,
            },
            "B_LOG_scaled": {
                "file": "output/scaled_matrices/B_LOG_scaled.csv",
                "participants": 967,
                "features": 24,
                "missing_values": 0,
                "means": m_bl,
                "stds": s_bl,
            }
        },
        "exclusions_and_boundaries_confirmed": {
            "source_matrices_modified": False,
            "source_cohorts_modified": False,
            "outliers_removed": False,
            "imputation_performed": False,
            "pca_performed": False,
            "clustering_performed": False,
            "tridosha_mapping_performed": False,
        },
        "validations": {
            "A_RAW_scaled": val_a_raw,
            "A_LOG_scaled": val_a_log,
            "B_RAW_scaled": val_b_raw,
            "B_LOG_scaled": val_b_log,
        }
    }

    meta_file = output_dir / "scaled_matrices_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  [SAVED METADATA] -> {meta_file.relative_to(output_dir.parent.parent)}")
    print(sep)

    return metadata
