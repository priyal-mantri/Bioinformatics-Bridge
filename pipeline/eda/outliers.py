"""
pipeline/eda/outliers.py
=========================
Phase 4 -- Outlier Investigation

For every feature variable, computes IQR-based outlier counts and
classifies each outlier as:

    BIOLOGICALLY PLAUSIBLE  -- within known human physiological range
    SUSPICIOUS              -- outside known physiological range; warrants review
    REQUIRES INVESTIGATION  -- extreme, likely erroneous

Saves:
    outliers/individual/<VARNAME>_boxplot.png
    outliers/outliers_summary_grid.png
    outliers/outlier_counts.csv
    outlier_report.txt

Outliers are NOT removed or winsorized.
If removal is scientifically justified, a NEW preprocessing decision is recommended.
"""

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ---------------------------------------------------------------------------
# Biological plausibility ranges (min, max) per variable
# Values inside this range are "biologically plausible" even if statistical
# outliers. Values outside this range are "suspicious" or require investigation.
# References: NHANES analytic guidelines, clinical reference ranges.
# ---------------------------------------------------------------------------
BIO_RANGES: dict[str, tuple[float, float]] = {
    "BMXBMI":           (10.0,  80.0),
    "BMXWAIST":         (30.0, 200.0),
    "Avg_Systolic_BP":  (60.0, 240.0),
    "Avg_Diastolic_BP": (20.0, 140.0),
    "BPXPLS":           (25.0, 200.0),
    "DXDTOBMD":         (0.30,  2.50),
    "DXDTOPF":          (2.0,   70.0),
    "DXDTOLE":          (5000, 120000),
    "LBXGLU":           (40.0, 600.0),
    "LBXIN":            (0.5,  500.0),
    "HOMA_IR":          (0.0,  100.0),
    "LBXTC":            (50.0, 500.0),
    "LBDHDD":           (5.0,  200.0),
    "LBXSTR":           (10.0, 5000.0),
    "TC_HDL_ratio":     (1.0,   30.0),
    "TG_HDL_ratio":     (0.1,   80.0),
    "LBXSATSI":         (1.0,  500.0),
    "LBXSAL":           (1.5,    6.0),
    "LBXSTP":           (3.0,   12.0),
    "LBXSTB":           (0.1,   20.0),
    "LBXSCR":           (0.1,   20.0),
    "LBXSUA":           (0.5,   20.0),
    "LBXSBU":           (1.0,  150.0),
    "LBXSCA":           (5.0,   15.0),
    "LBXSPH":           (0.5,   10.0),
    "LBXSNASI":         (100,   175.0),
    "LBXSKSI":          (2.0,    8.0),
}

# IQR multiplier for outlier detection
IQR_MILD    = 1.5
IQR_EXTREME = 3.0


def _classify_outlier(value: float, varname: str) -> str:
    """Classify a single outlier value using biological plausibility."""
    bio_min, bio_max = BIO_RANGES.get(varname, (-np.inf, np.inf))
    if bio_min <= value <= bio_max:
        return "BIOLOGICALLY PLAUSIBLE"
    elif value < bio_min * 0.5 or value > bio_max * 2.0:
        return "REQUIRES INVESTIGATION"
    else:
        return "SUSPICIOUS"


def _analyze_variable_outliers(series: pd.Series,
                                varname: str) -> dict:
    """
    Compute IQR-based outlier statistics for one variable.
    Returns a summary dict.
    """
    data = series.dropna()
    if len(data) < 10:
        return {}

    Q1  = data.quantile(0.25)
    Q3  = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_mild    = Q1 - IQR_MILD * IQR
    upper_mild    = Q3 + IQR_MILD * IQR
    lower_extreme = Q1 - IQR_EXTREME * IQR
    upper_extreme = Q3 + IQR_EXTREME * IQR

    mild_mask    = (data < lower_mild) | (data > upper_mild)
    extreme_mask = (data < lower_extreme) | (data > upper_extreme)

    mild_outliers    = data[mild_mask]
    extreme_outliers = data[extreme_mask]

    # Classify extremes
    classifications = extreme_outliers.apply(
        lambda v: _classify_outlier(v, varname)
    ).value_counts().to_dict()

    return {
        "n_valid":        len(data),
        "Q1":             round(Q1, 4),
        "Q3":             round(Q3, 4),
        "IQR":            round(IQR, 4),
        "lower_fence":    round(lower_mild, 4),
        "upper_fence":    round(upper_mild, 4),
        "n_mild_outliers":    int(mild_mask.sum()),
        "pct_mild":           round(mild_mask.sum() / len(data) * 100, 2),
        "n_extreme_outliers": int(extreme_mask.sum()),
        "pct_extreme":        round(extreme_mask.sum() / len(data) * 100, 2),
        "max_value":      round(data.max(), 4),
        "min_value":      round(data.min(), 4),
        "n_plausible":    classifications.get("BIOLOGICALLY PLAUSIBLE", 0),
        "n_suspicious":   classifications.get("SUSPICIOUS", 0),
        "n_requires_inv": classifications.get("REQUIRES INVESTIGATION", 0),
    }


def run_outliers(df: pd.DataFrame, output_dir: Path,
                 feature_cols: list[str]) -> dict:
    """
    Phase 4: Outlier Investigation.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset.
    output_dir : Path
        Root EDA output directory.
    feature_cols : list[str]
        Feature columns to analyse.

    Returns
    -------
    dict
        Key findings for EDA_SUMMARY.md.
    """
    save_dir  = output_dir / "outliers"
    indiv_dir = save_dir / "individual"
    indiv_dir.mkdir(parents=True, exist_ok=True)

    sep  = "=" * 65
    sep2 = "-" * 65
    lines = []

    def log(text=""):
        lines.append(text)
        print(text)

    log(sep)
    log("  PHASE 4 -- OUTLIER INVESTIGATION")
    log(sep)

    # ── Analyse each variable ────────────────────────────────────────────────
    outlier_rows = {}
    for col in feature_cols:
        result = _analyze_variable_outliers(df[col], col)
        if result:
            outlier_rows[col] = result

    outlier_df = pd.DataFrame(outlier_rows).T
    outlier_df.index.name = "variable"
    outlier_df.to_csv(save_dir / "outlier_counts.csv")

    # ── Print table ──────────────────────────────────────────────────────────
    log(f"\n  {'Variable':<22} {'N Valid':>8} {'Mild Outliers':>14} "
        f"{'Extreme Outliers':>17} {'Suspicious':>11} {'Req.Invest.':>12}")
    log(f"  {'-'*22} {'-'*8} {'-'*14} {'-'*17} {'-'*11} {'-'*12}")
    for var, row in outlier_df.iterrows():
        log(f"  {var:<22} {int(row['n_valid']):>8,} "
            f"{int(row['n_mild_outliers']):>8,} ({row['pct_mild']:>4.1f}%)  "
            f"{int(row['n_extreme_outliers']):>8,} ({row['pct_extreme']:>4.1f}%)  "
            f"{int(row['n_suspicious']):>8,}     "
            f"{int(row['n_requires_inv']):>8,}")

    # ── Generate individual boxplots ─────────────────────────────────────────
    log(f"\n  Generating individual boxplots ...")
    for col in feature_cols:
        data = df[col].dropna()
        if len(data) < 10:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        stats = outlier_df.loc[col] if col in outlier_df.index else {}

        ax.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#bbdefb", color="#1565c0"),
                   medianprops=dict(color="#e74c3c", linewidth=2.5),
                   whiskerprops=dict(color="#1565c0", linewidth=1.5),
                   capprops=dict(color="#1565c0", linewidth=1.5),
                   flierprops=dict(marker="o", markerfacecolor="#FF5722",
                                   markersize=4, alpha=0.5, markeredgewidth=0))

        n_extreme = int(stats.get("n_extreme_outliers", 0))
        n_suspicious = int(stats.get("n_suspicious", 0))
        subtitle = (f"Extreme: {n_extreme}  |  Suspicious: {n_suspicious}  |  "
                    f"Max: {data.max():.3f}  |  Min: {data.min():.3f}")
        ax.set_title(f"{col}\n{subtitle}", fontsize=10, fontweight="bold")
        ax.set_ylabel(col, fontsize=9)
        ax.set_xticks([])
        sns.despine(ax=ax)

        plt.tight_layout()
        fig.savefig(indiv_dir / f"{col}_boxplot.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    log(f"  [SAVED] {len(feature_cols)} individual boxplots -> outliers/individual/")

    # ── Summary boxplot grid ─────────────────────────────────────────────────
    log("  Generating outlier summary grid ...")
    valid_cols  = [c for c in feature_cols if c in outlier_df.index]
    n_cols_grid = 4
    n_rows_grid = math.ceil(len(valid_cols) / n_cols_grid)

    fig, axes = plt.subplots(n_rows_grid, n_cols_grid,
                              figsize=(n_cols_grid * 4, n_rows_grid * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(valid_cols):
        data = df[col].dropna()
        ax = axes[i]
        ax.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#bbdefb", color="#1565c0"),
                   medianprops=dict(color="#e74c3c", linewidth=2),
                   whiskerprops=dict(color="#1565c0"),
                   capprops=dict(color="#1565c0"),
                   flierprops=dict(marker="o", markerfacecolor="#FF5722",
                                   markersize=2.5, alpha=0.4, markeredgewidth=0))
        n_ext = int(outlier_df.loc[col, "n_extreme_outliers"])
        color = "#e74c3c" if n_ext > 20 else ("#FF9800" if n_ext > 5 else "#4caf50")
        ax.set_title(f"{col}\next={n_ext}", fontsize=7.5, color=color,
                     fontweight="bold" if n_ext > 5 else "normal")
        ax.set_xticks([])
        ax.tick_params(labelsize=6)
        sns.despine(ax=ax)

    for j in range(len(valid_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Outlier Summary Grid\n"
                 "(red=many extremes, orange=some, green=few)",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_dir / "outliers_summary_grid.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    log("  [SAVED] outliers_summary_grid.png")

    # ── Report ────────────────────────────────────────────────────────────────
    log(f"\n{sep2}")
    log("  OUTLIER REPORT")
    log(sep2)

    vars_requiring_inv = outlier_df[outlier_df["n_requires_inv"] > 0].index.tolist()
    vars_suspicious    = outlier_df[outlier_df["n_suspicious"]   > 0].index.tolist()

    log(f"\n  Variables with extreme values requiring investigation:")
    if vars_requiring_inv:
        for col in vars_requiring_inv:
            row = outlier_df.loc[col]
            log(f"    {col:<22}  max={row['max_value']:.3f}  "
                f"requires_investigation={int(row['n_requires_inv'])}")
    else:
        log("    None.")

    log(f"\n  Variables with suspicious extreme values:")
    if vars_suspicious:
        for col in vars_suspicious:
            row = outlier_df.loc[col]
            log(f"    {col:<22}  max={row['max_value']:.3f}  "
                f"suspicious={int(row['n_suspicious'])}")
    else:
        log("    None.")

    log(f"\n  Classification note:")
    log(f"    BIOLOGICALLY PLAUSIBLE -- statistical outlier but within known")
    log(f"      human physiological range. Retain without modification.")
    log(f"    SUSPICIOUS             -- outside expected physiological range.")
    log(f"      Review individual cases before clustering.")
    log(f"    REQUIRES INVESTIGATION -- likely erroneous or extreme pathology.")
    log(f"      Recommend a preprocessing decision to handle.")

    # Build recommendations
    recommendations = []
    if vars_requiring_inv:
        recommendations.append(
            f"Recommend creation of Decision 007: Outlier handling for "
            f"{vars_requiring_inv}. Classify whether to winsorize or remove "
            f"on a per-variable basis after scientific review."
        )
    if not recommendations:
        recommendations.append(
            "No additional preprocessing recommended based on outlier analysis. "
            "All extreme values appear biologically plausible."
        )

    for rec in recommendations:
        log(f"\n  Recommendation:\n    {rec}")

    report_path = save_dir / "outlier_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [SAVED] {report_path}")
    print(sep)

    return {
        "vars_requiring_inv": vars_requiring_inv,
        "vars_suspicious":    vars_suspicious,
        "outlier_df":         outlier_df,
        "recommendations":    recommendations,
    }
