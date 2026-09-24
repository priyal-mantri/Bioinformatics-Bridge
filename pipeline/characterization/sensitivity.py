"""
Sensitivity and robustness module for Stage 8 phenotype characterization.
Evaluates persistence of phenotype patterns across RAW vs LOG representations
and alternative model families (K-Means, GMM, Ward, Spectral) without ranking models
or selecting a new K.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from typing import Dict, List, Any
from .data_loader import load_all_assignments, load_master_cohort


def compute_pairwise_ari(assignments_dict: Dict[str, pd.DataFrame], target_k: int) -> pd.DataFrame:
    """Computes Adjusted Rand Index (ARI) between compatible fixed-K assignments."""
    records = []
    keys = list(assignments_dict.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            df1, df2 = assignments_dict[k1], assignments_dict[k2]

            # Find common K column
            col1 = f"K{target_k}_cluster" if f"K{target_k}_cluster" in df1.columns else f"ward_K{target_k}"
            col2 = f"K{target_k}_cluster" if f"K{target_k}_cluster" in df2.columns else f"ward_K{target_k}"

            if col1 in df1.columns and col2 in df2.columns:
                merged = df1[["SEQN", col1]].merge(
                    df2[["SEQN", col2]], on="SEQN", suffixes=("_m1", "_m2"), how="inner"
                )
                col1_merged = f"{col1}_m1" if f"{col1}_m1" in merged.columns else col1
                col2_merged = f"{col2}_m2" if f"{col2}_m2" in merged.columns else col2

                ari = adjusted_rand_score(merged[col1_merged], merged[col2_merged])
                records.append({
                    "model_1": k1,
                    "model_2": k2,
                    "target_k": target_k,
                    "common_n": len(merged),
                    "adjusted_rand_index": ari
                })

    return pd.DataFrame(records)


def evaluate_feature_persistence(
    profile_raw: pd.DataFrame,
    profile_log: pd.DataFrame,
    retained_k: int
) -> pd.DataFrame:
    """
    Classifies feature profile persistence across RAW and LOG representations.

    Persistence Categories:
    - Persistent: Directional pattern preserved between RAW and LOG scales.
    - Representation-Dependent: Directional pattern changes between RAW and LOG scales.
    - Partially Persistent: Partial pattern agreement across scales.
    """
    records = []
    features = profile_raw["feature"].unique()

    for feat in features:
        r_row = profile_raw[profile_raw["feature"] == feat]
        l_row = profile_log[profile_log["feature"] == feat]

        if r_row.empty or l_row.empty:
            continue

        raw_zs = [r_row[f"c{c}_zscore"].values[0] for c in range(retained_k)]
        log_zs = [l_row[f"c{c}_zscore"].values[0] for c in range(retained_k)]

        # Determine directional vectors (+1 for Higher > 0.20, -1 for Lower < -0.20, 0 for Near Ref)
        def _get_dir(zs):
            return [1 if z > 0.20 else (-1 if z < -0.20 else 0) for z in zs]

        raw_dirs = _get_dir(raw_zs)
        log_dirs = _get_dir(log_zs)

        if raw_dirs == log_dirs:
            classification = "Persistent"
        else:
            # Check if signs are completely flipped or just shifted across threshold
            has_opposite = any(r * l == -1 for r, l in zip(raw_dirs, log_dirs))
            if has_opposite:
                classification = "Representation-Dependent"
            else:
                classification = "Partially Persistent"

        records.append({
            "feature": feat,
            "retained_k": retained_k,
            "raw_zscores": str([round(z, 3) for z in raw_zs]),
            "log_zscores": str([round(z, 3) for z in log_zs]),
            "raw_directions": str(raw_dirs),
            "log_directions": str(log_dirs),
            "persistence_category": classification
        })

    return pd.DataFrame(records)


def run_sensitivity_analysis(
    cohort_key: str,
    profile_raw: pd.DataFrame,
    profile_log: pd.DataFrame,
    retained_k: int
) -> Dict[str, pd.DataFrame]:
    """Executes sensitivity and robustness analysis for a given cohort."""
    assignments = load_all_assignments(cohort_key)
    ari_df = compute_pairwise_ari(assignments, retained_k)
    persistence_df = evaluate_feature_persistence(profile_raw, profile_log, retained_k)

    return {
        "pairwise_ari": ari_df,
        "feature_persistence": persistence_df
    }
