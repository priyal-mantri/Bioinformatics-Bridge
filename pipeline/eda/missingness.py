"""
pipeline/eda/missingness.py
============================
Phase 2 -- Missing Data Analysis

Computes per-variable missingness, identifies patterns of co-missingness,
estimates the effective clustering sample size, and generates:

    missingness/missing_values_table.csv
    missingness/missingness_heatmap.png
    missingness/missingness_correlation.png
    missingness/missingness_report.txt

This phase is DESCRIPTIVE ONLY.
No imputation or row deletion is performed.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Threshold above which missingness is flagged as HIGH
HIGH_MISSING_THRESHOLD = 0.40   # 40 %
# Missingness correlation above which two variables are considered co-missing
CO_MISSING_CORR_THRESHOLD = 0.70


def run_missingness(df: pd.DataFrame, output_dir: Path,
                    feature_cols: list[str]) -> dict:
    """
    Phase 2: Missing Data Analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset.
    output_dir : Path
        Root EDA output directory.
    feature_cols : list[str]
        Columns to treat as features.

    Returns
    -------
    dict
        Key findings for EDA_SUMMARY.md.
    """
    save_dir = output_dir / "missingness"
    save_dir.mkdir(parents=True, exist_ok=True)

    sep  = "=" * 65
    sep2 = "-" * 65
    lines = []

    def log(text=""):
        lines.append(text)
        print(text)

    log(sep)
    log("  PHASE 2 -- MISSING DATA ANALYSIS")
    log(sep)

    n_rows = len(df)

    # ── Per-variable missingness ─────────────────────────────────────────────
    miss_count = df.isna().sum()
    miss_pct   = (miss_count / n_rows * 100).round(2)

    miss_table = pd.DataFrame({
        "missing_count": miss_count,
        "missing_pct":   miss_pct,
        "present_count": n_rows - miss_count,
    }).sort_values("missing_pct", ascending=False)

    miss_table.to_csv(save_dir / "missing_values_table.csv")

    high_miss = miss_table[miss_table["missing_pct"] > HIGH_MISSING_THRESHOLD * 100]
    low_miss  = miss_table[miss_table["missing_pct"] == 0]

    log(f"\n  Total participants  : {n_rows:,}")
    log(f"  Total variables     : {len(df.columns)}")
    log(f"\n  Variables with > {HIGH_MISSING_THRESHOLD*100:.0f}% missing ({len(high_miss)}):")
    for col, row in high_miss.iterrows():
        log(f"    {col:<22} {row['missing_count']:>6,}  ({row['missing_pct']:.1f}%)")

    log(f"\n  Variables with 0% missing ({len(low_miss)}): {list(low_miss.index)}")

    # ── Complete cases analysis ───────────────────────────────────────────────
    log(f"\n{sep2}")
    log("  COMPLETE CASES ANALYSIS")
    log(sep2)

    # All feature columns
    complete_all_features = df[feature_cols].dropna().shape[0]
    log(f"  Complete cases (all {len(feature_cols)} features present) : "
        f"{complete_all_features:,}  ({complete_all_features/n_rows*100:.1f}%)")

    # Without DEXA (large missingness subset)
    dexa_cols    = [c for c in feature_cols if c in
                    {"DXDTOBMD", "DXDTOPF", "DXDTOLE"}]
    fasting_cols = [c for c in feature_cols if c in
                    {"LBXGLU", "LBXIN", "HOMA_IR"}]
    core_cols    = [c for c in feature_cols
                    if c not in dexa_cols and c not in fasting_cols]

    complete_core    = df[core_cols].dropna().shape[0]
    complete_no_dexa = df[[c for c in feature_cols
                           if c not in dexa_cols]].dropna().shape[0]
    complete_no_fast = df[[c for c in feature_cols
                           if c not in fasting_cols]].dropna().shape[0]

    log(f"  Complete cases (core only, no DEXA/fasting) : "
        f"{complete_core:,}  ({complete_core/n_rows*100:.1f}%)")
    log(f"  Complete cases (excluding DEXA columns)     : "
        f"{complete_no_dexa:,}  ({complete_no_dexa/n_rows*100:.1f}%)")
    log(f"  Complete cases (excluding fasting columns)  : "
        f"{complete_no_fast:,}  ({complete_no_fast/n_rows*100:.1f}%)")
    log(f"\n  --> Effective clustering sample (all features present): "
        f"{complete_all_features:,} participants")

    # ── Missingness heatmap ───────────────────────────────────────────────────
    log(f"\n{sep2}")
    log("  Generating missingness heatmap ...")

    # Use all columns for the heatmap
    plot_cols = [c for c in df.columns if c != "SEQN"]
    missing_matrix = df[plot_cols].isna().astype(int)

    fig, ax = plt.subplots(figsize=(18, 7))
    sns.heatmap(
        missing_matrix.T,
        cmap=["#e8f4f8", "#e74c3c"],
        cbar=False,
        yticklabels=True,
        xticklabels=False,
        ax=ax,
        linewidths=0,
    )
    ax.set_title("Missingness Pattern\n(red = missing, blue = present)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(f"Participants (n={n_rows:,})", fontsize=10)
    ax.set_ylabel("Variables", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig.savefig(save_dir / "missingness_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("  [SAVED] missingness_heatmap.png")

    # ── Missingness correlation matrix ───────────────────────────────────────
    log("  Generating missingness correlation matrix ...")

    miss_indicators = df[feature_cols].isna().astype(int)
    # Only keep variables that have some missingness (corr of all-zeros is NaN)
    miss_indicators = miss_indicators.loc[:, miss_indicators.sum() > 0]
    miss_corr = miss_indicators.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(miss_corr, dtype=bool), k=1)
    sns.heatmap(
        miss_corr,
        mask=mask,
        cmap="RdYlBu_r",
        center=0,
        vmin=-1, vmax=1,
        annot=False,
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.8, "label": "Missingness correlation"},
        ax=ax,
    )
    ax.set_title("Missingness Correlation Matrix\n"
                 "(high values = variables tend to be missing together)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(axis="both", labelsize=7)
    plt.tight_layout()
    fig.savefig(save_dir / "missingness_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("  [SAVED] missingness_correlation.png")

    # ── Co-missingness pairs ─────────────────────────────────────────────────
    high_co_miss = []
    corr_vals    = miss_corr.unstack()
    for (c1, c2), val in corr_vals.items():
        if c1 < c2 and abs(val) >= CO_MISSING_CORR_THRESHOLD:
            high_co_miss.append((c1, c2, round(val, 3)))

    # ── Report ────────────────────────────────────────────────────────────────
    log(f"\n{sep2}")
    log("  MISSINGNESS REPORT")
    log(sep2)

    # Expected high-missingness (by study design)
    expected_high = {"DXDTOBMD", "DXDTOPF", "DXDTOLE",
                     "DXAEXSTS", "LBXGLU", "LBXIN", "HOMA_IR", "LBDINLC"}
    unexpected_high = [c for c in high_miss.index if c not in expected_high]

    log(f"\n  Expected high-missingness variables (DEXA/fasting subsamples):")
    for col in high_miss.index:
        if col in expected_high:
            pct = miss_table.loc[col, "missing_pct"]
            log(f"    {col:<22} {pct:.1f}%  [expected by study design]")

    if unexpected_high:
        log(f"\n  Unexpected high-missingness variables:")
        for col in unexpected_high:
            pct = miss_table.loc[col, "missing_pct"]
            log(f"    {col:<22} {pct:.1f}%  [INVESTIGATE]")
    else:
        log(f"\n  No unexpected high-missingness variables detected.")

    log(f"\n  Co-missing variable pairs (corr >= {CO_MISSING_CORR_THRESHOLD}):")
    if high_co_miss:
        for c1, c2, val in sorted(high_co_miss, key=lambda x: -abs(x[2])):
            log(f"    {c1:<22} <--> {c2:<22}  corr={val:.3f}")
    else:
        log("    None above threshold.")

    log(f"\n  Effective clustering sample size: {complete_all_features:,} participants")

    recommendation = (
        "No additional preprocessing recommended based on missingness alone.\n"
        "Missingness follows expected NHANES subsampling patterns.\n"
        "Imputation strategy should be decided before clustering (future decision)."
    )
    log(f"\n  Recommendation:\n    {recommendation}")

    # Save report
    report_path = save_dir / "missingness_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [SAVED] {report_path}")
    print(sep)

    return {
        "high_missing_vars": list(high_miss.index),
        "high_missing_pcts": {c: miss_table.loc[c, "missing_pct"]
                              for c in high_miss.index},
        "n_complete_all":    complete_all_features,
        "n_complete_core":   complete_core,
        "co_missing_pairs":  high_co_miss,
        "unexpected_high":   unexpected_high,
        "recommendation":    "No additional preprocessing recommended based on missingness.",
    }
