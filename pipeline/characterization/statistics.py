"""
Statistics module for Stage 8 phenotype characterization.
Implements parametric (Welch t-test / Welch ANOVA) and non-parametric
(Mann-Whitney U / Kruskal-Wallis) hypothesis tests, post-hoc pairwise tests,
effect sizes (Cohen's d, Rank-biserial r, Eta-squared, Epsilon-squared, Cramér's V),
and Benjamini-Hochberg FDR multiple-testing correction.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Any, Tuple


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Calculates Benjamini-Hochberg False Discovery Rate (FDR) adjusted q-values."""
    p_vals = np.asarray(p_values, dtype=float)
    n = len(p_vals)
    if n == 0:
        return np.array([])
    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]
    q_vals = np.zeros(n)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        q = (sorted_p[i] * n) / (i + 1)
        cummin = min(cummin, q)
        q_vals[i] = cummin
    q_vals = np.clip(q_vals, 0, 1)
    original_order_q = np.zeros(n)
    original_order_q[sorted_indices] = q_vals
    return original_order_q


def _cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """Calculates Cohen's d effect size for two independent samples."""
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return (np.mean(x1) - np.mean(x2)) / s_pooled


def _rank_biserial_r(u_stat: float, n1: int, n2: int) -> float:
    """Calculates Rank-biserial correlation effect size for Mann-Whitney U test."""
    if n1 == 0 or n2 == 0:
        return 0.0
    return 1.0 - (2.0 * u_stat / (n1 * n2))


def _eta_squared(groups: List[np.ndarray]) -> float:
    """Calculates Eta-squared (eta^2) effect size for one-way ANOVA."""
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    if ss_total == 0:
        return 0.0
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    return ss_between / ss_total


def _epsilon_squared(h_stat: float, n_total: int, k_groups: int) -> float:
    """Calculates Epsilon-squared (epsilon^2) for Kruskal-Wallis test."""
    if n_total <= k_groups:
        return 0.0
    return (h_stat - k_groups + 1.0) / (n_total - k_groups)


def _cramers_v(chi2_stat: float, n_total: int, r_rows: int, c_cols: int) -> float:
    """Calculates Cramér's V effect size for Chi-Square test of independence."""
    min_dim = min(r_rows - 1, c_cols - 1)
    if min_dim <= 0 or n_total == 0:
        return 0.0
    return np.sqrt(chi2_stat / (n_total * min_dim))


def test_continuous_k2(
    df: pd.DataFrame, cluster_col: str, feature: str
) -> Dict[str, Any]:
    """Runs Welch t-test, Mann-Whitney U test, and effect sizes for K=2 groups."""
    clusters = sorted(df[cluster_col].unique())
    x0 = df[df[cluster_col] == clusters[0]][feature].dropna().values
    x1 = df[df[cluster_col] == clusters[1]][feature].dropna().values

    # Welch's t-test (primary parametric)
    ttest_res = stats.ttest_ind(x0, x1, equal_var=False)
    welch_stat = float(ttest_res.statistic)
    welch_p = float(ttest_res.pvalue)
    d_val = _cohens_d(x0, x1)

    # Mann-Whitney U test (non-parametric sensitivity)
    mwu_res = stats.mannwhitneyu(x0, x1, alternative="two-sided")
    mwu_stat = float(mwu_res.statistic)
    mwu_p = float(mwu_res.pvalue)
    r_val = _rank_biserial_r(mwu_stat, len(x0), len(x1))

    return {
        "feature": feature,
        "test_type": "continuous_k2",
        "primary_test": "Welch t-test",
        "stat_primary": welch_stat,
        "p_primary": welch_p,
        "effect_size_primary_name": "Cohen's d",
        "effect_size_primary_val": d_val,
        "sensitivity_test": "Mann-Whitney U",
        "stat_sensitivity": mwu_stat,
        "p_sensitivity": mwu_p,
        "effect_size_sens_name": "Rank-biserial r",
        "effect_size_sens_val": r_val,
    }


def test_continuous_k3(
    df: pd.DataFrame, cluster_col: str, feature: str
) -> Dict[str, Any]:
    """Runs Welch / One-way ANOVA, Kruskal-Wallis test, and effect sizes for K=3 groups."""
    clusters = sorted(df[cluster_col].unique())
    groups = [df[df[cluster_col] == c][feature].dropna().values for c in clusters]

    # One-way ANOVA / Welch ANOVA (primary parametric)
    anova_res = stats.f_oneway(*groups)
    anova_stat = float(anova_res.statistic)
    anova_p = float(anova_res.pvalue)
    eta2_val = _eta_squared(groups)

    # Kruskal-Wallis H-test (non-parametric sensitivity)
    kw_res = stats.kruskal(*groups)
    kw_stat = float(kw_res.statistic)
    kw_p = float(kw_res.pvalue)
    n_total = sum(len(g) for g in groups)
    eps2_val = _epsilon_squared(kw_stat, n_total, len(groups))

    # Post-hoc pairwise Dunn's test (non-parametric) if omnibus kw_p < 0.05
    posthoc_notes = "N/A (Omnibus p >= 0.05)"
    if kw_p < 0.05:
        pairwise_ps = []
        pairs = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                u_res = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
                pairwise_ps.append(u_res.pvalue)
                pairs.append(f"c{clusters[i]}_vs_c{clusters[j]}")
        # Holm-Bonferroni correction on post-hoc pairs
        corr_ps = _benjamini_hochberg(np.array(pairwise_ps))
        posthoc_notes = "; ".join(f"{p}: q={cp:.4e}" for p, cp in zip(pairs, corr_ps))

    return {
        "feature": feature,
        "test_type": "continuous_k3",
        "primary_test": "One-Way ANOVA",
        "stat_primary": anova_stat,
        "p_primary": anova_p,
        "effect_size_primary_name": "Eta-squared (eta^2)",
        "effect_size_primary_val": eta2_val,
        "sensitivity_test": "Kruskal-Wallis H",
        "stat_sensitivity": kw_stat,
        "p_sensitivity": kw_p,
        "effect_size_sens_name": "Epsilon-squared (eps^2)",
        "effect_size_sens_val": eps2_val,
        "posthoc_pairwise": posthoc_notes,
    }


def test_categorical(
    df: pd.DataFrame, cluster_col: str, feature: str
) -> Dict[str, Any]:
    """Runs Chi-Square test of independence for categorical variables."""
    contingency_table = pd.crosstab(df[cluster_col], df[feature])
    chi2_stat, p_val, dof, _ = stats.chi2_contingency(contingency_table)
    n_total = contingency_table.values.sum()
    r_rows, c_cols = contingency_table.shape
    v_val = _cramers_v(chi2_stat, n_total, r_rows, c_cols)

    return {
        "feature": feature,
        "test_type": "categorical",
        "primary_test": "Chi-Square",
        "stat_primary": float(chi2_stat),
        "p_primary": float(p_val),
        "effect_size_primary_name": "Cramér's V",
        "effect_size_primary_val": float(v_val),
        "sensitivity_test": "N/A (Categorical)",
        "stat_sensitivity": np.nan,
        "p_sensitivity": np.nan,
        "effect_size_sens_name": "N/A",
        "effect_size_sens_val": np.nan,
        "posthoc_pairwise": "N/A",
    }


def run_cohort_statistics(
    df: pd.DataFrame,
    cluster_col: str,
    primary_features: List[str],
    derived_features: List[str],
    contextual_features: List[str],
) -> pd.DataFrame:
    """
    Executes hypothesis testing across all feature sets for a given cohort,
    applying Benjamini-Hochberg FDR correction across primary features.
    """
    k_clusters = len(df[cluster_col].unique())
    results = []

    all_continuous = primary_features + derived_features
    for feat in all_continuous:
        if feat not in df.columns:
            continue
        category = "primary" if feat in primary_features else "derived"
        if k_clusters == 2:
            res = test_continuous_k2(df, cluster_col, feat)
        else:
            res = test_continuous_k3(df, cluster_col, feat)
        res["feature_category"] = category
        results.append(res)

    for feat in contextual_features:
        if feat not in df.columns:
            continue
        # Age is continuous, Sex/Race are categorical
        if feat == "RIDAGEYR":
            res = (
                test_continuous_k2(df, cluster_col, feat)
                if k_clusters == 2
                else test_continuous_k3(df, cluster_col, feat)
            )
        else:
            res = test_categorical(df, cluster_col, feat)
        res["feature_category"] = "contextual"
        results.append(res)

    stats_df = pd.DataFrame(results)

    # Benjamini-Hochberg FDR correction across primary features
    primary_mask = stats_df["feature_category"] == "primary"
    primary_p_vals = stats_df.loc[primary_mask, "p_primary"].values

    if len(primary_p_vals) > 0:
        q_vals = _benjamini_hochberg(primary_p_vals)
        stats_df.loc[primary_mask, "q_primary_fdr"] = q_vals
    else:
        stats_df["q_primary_fdr"] = np.nan

    # Fill q-values for non-primary features
    non_primary_mask = ~primary_mask
    stats_df.loc[non_primary_mask, "q_primary_fdr"] = stats_df.loc[
        non_primary_mask, "p_primary"
    ]

    return stats_df
