"""
Profiling module for Stage 8 phenotype characterization.
Computes cluster membership counts, cohort baseline statistics,
per-cluster distribution metrics, and standardized Z-scores relative to the cohort.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any


def compute_cluster_membership(df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    """Computes cluster sizes and cohort percentage representation."""
    total_n = len(df)
    counts = df[cluster_col].value_counts().sort_index()
    percentages = (counts / total_n) * 100.0

    membership_df = pd.DataFrame({
        "cluster": counts.index,
        "n": counts.values,
        "percentage": np.round(percentages.values, 2)
    })
    return membership_df


def compute_feature_profiles(
    df: pd.DataFrame,
    cluster_col: str,
    feature_list: List[str],
    feature_category: str = "primary"
) -> pd.DataFrame:
    """
    Computes cohort baseline statistics (mean, std, median, IQR) and per-cluster
    statistics alongside cohort-standardized Z-scores for each feature.

    Standardized Cluster Z-Score:
        Z = (mean_cluster - mean_cohort) / std_cohort
    """
    records = []
    clusters = sorted(df[cluster_col].unique())

    for feat in feature_list:
        if feat not in df.columns:
            continue

        series_cohort = df[feat].dropna()
        cohort_mean = series_cohort.mean()
        cohort_std = series_cohort.std(ddof=1)
        cohort_median = series_cohort.median()
        cohort_q25 = series_cohort.quantile(0.25)
        cohort_q75 = series_cohort.quantile(0.75)
        cohort_iqr = cohort_q75 - cohort_q25

        row = {
            "feature": feat,
            "category": feature_category,
            "cohort_mean": cohort_mean,
            "cohort_std": cohort_std,
            "cohort_median": cohort_median,
            "cohort_q25": cohort_q25,
            "cohort_q75": cohort_q75,
            "cohort_iqr": cohort_iqr,
        }

        for c in clusters:
            sub = df[df[cluster_col] == c][feat].dropna()
            c_mean = sub.mean()
            c_std = sub.std(ddof=1) if len(sub) > 1 else 0.0
            c_median = sub.median()
            c_q25 = sub.quantile(0.25)
            c_q75 = sub.quantile(0.75)

            # Calculate Cohort Standardized Z-Score
            z_score = (c_mean - cohort_mean) / cohort_std if cohort_std > 0 else 0.0

            row[f"c{c}_n"] = len(sub)
            row[f"c{c}_mean"] = c_mean
            row[f"c{c}_std"] = c_std
            row[f"c{c}_median"] = c_median
            row[f"c{c}_q25"] = c_q25
            row[f"c{c}_q75"] = c_q75
            row[f"c{c}_zscore"] = z_score

        records.append(row)

    profile_df = pd.DataFrame(records)
    return profile_df
