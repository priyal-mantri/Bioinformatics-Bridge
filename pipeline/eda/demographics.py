"""
pipeline/eda/demographics.py
=============================
Phase 6 -- Demographic Profiling

Generates descriptive summaries for age, gender, and ethnicity,
and computes biomarker means by demographic groups.

Purpose: provide biological context for interpreting clusters.
Demographic adjustment is NOT performed here.

Saves:
    demographics/age_distribution.png
    demographics/gender_distribution.png
    demographics/ethnicity_distribution.png
    demographics/biomarkers_by_gender.png
    demographics/biomarkers_by_age_group.png
    demographics/biomarkers_by_gender.csv
    demographics/biomarkers_by_age_group.csv
    demographics/demographics_report.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# NHANES coding references
GENDER_MAP   = {1: "Male", 2: "Female"}
ETHNICITY_MAP = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other/Multiracial",
}

# Age groups for stratified biomarker means
AGE_BINS    = [19, 39, 59, 80]
AGE_LABELS  = ["20-39", "40-59", "60-80"]

# Select a representative subset of features for the demographic plots
# (avoid overcrowding the grouped bar charts)
DEMO_PLOT_FEATURES = [
    "BMXBMI", "BMXWAIST",
    "Avg_Systolic_BP", "Avg_Diastolic_BP",
    "LBXTC", "LBDHDD", "LBXSTR",
    "LBXGLU", "HOMA_IR",
    "DXDTOBMD", "DXDTOPF",
    "LBXSCR", "LBXSUA",
]


def run_demographics(df: pd.DataFrame, output_dir: Path,
                     feature_cols: list[str]) -> dict:
    """
    Phase 6: Demographic Profiling.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset. Must contain RIAGENDR, RIDAGEYR, RIDRETH3.
    output_dir : Path
        Root EDA output directory.
    feature_cols : list[str]
        All feature columns (for mean tables).

    Returns
    -------
    dict
        Key findings for EDA_SUMMARY.md.
    """
    save_dir = output_dir / "demographics"
    save_dir.mkdir(parents=True, exist_ok=True)

    sep  = "=" * 65
    sep2 = "-" * 65
    lines = []

    def log(text=""):
        lines.append(text)
        print(text)

    log(sep)
    log("  PHASE 6 -- DEMOGRAPHIC PROFILING")
    log(sep)

    # ── Decode demographic variables ─────────────────────────────────────────
    df_plot = df.copy()
    df_plot["Gender"]    = df_plot["RIAGENDR"].map(GENDER_MAP)
    df_plot["Ethnicity"] = df_plot["RIDRETH3"].map(ETHNICITY_MAP)
    df_plot["AgeGroup"]  = pd.cut(df_plot["RIDAGEYR"],
                                   bins=AGE_BINS, labels=AGE_LABELS, right=True)

    # ── Age distribution ─────────────────────────────────────────────────────
    log(f"\n  Age distribution:")
    age_stats = df["RIDAGEYR"].describe()
    log(f"    min={age_stats['min']:.0f}  mean={age_stats['mean']:.1f}  "
        f"median={df['RIDAGEYR'].median():.0f}  max={age_stats['max']:.0f}")
    for lbl in AGE_LABELS:
        n = (df_plot["AgeGroup"] == lbl).sum()
        pct = n / len(df) * 100
        log(f"    {lbl}: {n:,}  ({pct:.1f}%)")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["RIDAGEYR"], bins=30, kde=True, ax=ax,
                 color="#2196F3", alpha=0.75,
                 line_kws={"linewidth": 2, "color": "#0D47A1"})
    ax.axvline(df["RIDAGEYR"].mean(),   color="#e74c3c", linestyle="--",
               linewidth=2, label=f"Mean={df['RIDAGEYR'].mean():.1f}")
    ax.axvline(df["RIDAGEYR"].median(), color="#2ecc71", linestyle="--",
               linewidth=2, label=f"Median={df['RIDAGEYR'].median():.0f}")
    ax.set_title("Age Distribution (Adults 20-80)", fontsize=12,
                 fontweight="bold")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.legend()
    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(save_dir / "age_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Gender distribution ──────────────────────────────────────────────────
    gender_counts = df_plot["Gender"].value_counts()
    log(f"\n  Gender distribution:")
    for gender, count in gender_counts.items():
        log(f"    {gender}: {count:,}  ({count/len(df)*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#2196F3", "#E91E63"]
    ax.pie(gender_counts.values, labels=gender_counts.index,
           autopct="%1.1f%%", colors=colors, startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 2})
    ax.set_title("Gender Distribution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_dir / "gender_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Ethnicity distribution ───────────────────────────────────────────────
    eth_counts = df_plot["Ethnicity"].value_counts()
    log(f"\n  Ethnicity distribution:")
    for eth, count in eth_counts.items():
        log(f"    {eth}: {count:,}  ({count/len(df)*100:.1f}%)")

    palette = ["#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336","#795548"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(eth_counts.index, eth_counts.values,
                   color=palette[:len(eth_counts)], edgecolor="white")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 10, bar.get_y() + bar.get_height()/2,
                f"{w:,}  ({w/len(df)*100:.1f}%)",
                va="center", fontsize=9)
    ax.set_title("Ethnicity Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Count")
    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(save_dir / "ethnicity_distribution.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Biomarkers by gender ─────────────────────────────────────────────────
    plot_feats = [c for c in DEMO_PLOT_FEATURES if c in df.columns]
    gender_means = df_plot.groupby("Gender")[plot_feats].mean().T.round(3)
    gender_means.to_csv(save_dir / "biomarkers_by_gender.csv")

    log(f"\n  Biomarker means by gender (selected features):")
    log(f"  {'Variable':<22} {'Male':>10} {'Female':>10} {'Diff %':>10}")
    log(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10}")
    for col in plot_feats:
        male_mean   = gender_means.loc[col, "Male"]   if "Male"   in gender_means.columns else np.nan
        female_mean = gender_means.loc[col, "Female"] if "Female" in gender_means.columns else np.nan
        if pd.notna(male_mean) and pd.notna(female_mean) and female_mean != 0:
            diff_pct = ((male_mean - female_mean) / female_mean) * 100
        else:
            diff_pct = np.nan
        log(f"  {col:<22} {male_mean:>10.3f} {female_mean:>10.3f} {diff_pct:>+9.1f}%")

    # Grouped bar chart for gender means (normalized for visual comparison)
    n_feats = len(plot_feats)
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    for i, col in enumerate(plot_feats):
        ax = axes[i]
        data_m = df_plot[df_plot["Gender"] == "Male"][col].dropna()
        data_f = df_plot[df_plot["Gender"] == "Female"][col].dropna()
        ax.boxplot([data_m, data_f], labels=["Male", "Female"],
                   patch_artist=True,
                   boxprops=dict(facecolor="#bbdefb"),
                   medianprops=dict(color="#e74c3c", linewidth=2))
        ax.set_title(col, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7)
        sns.despine(ax=ax)
    for j in range(n_feats, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Biomarker Distributions by Gender",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_dir / "biomarkers_by_gender.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Biomarkers by age group ──────────────────────────────────────────────
    age_means = df_plot.groupby("AgeGroup", observed=True)[plot_feats].mean().T.round(3)
    age_means.to_csv(save_dir / "biomarkers_by_age_group.csv")

    log(f"\n  Biomarker means by age group (selected features):")
    header = f"  {'Variable':<22}"
    for lbl in AGE_LABELS:
        header += f" {lbl:>10}"
    log(header)
    log(f"  {'-'*22}" + "".join([f" {'-'*10}" for _ in AGE_LABELS]))
    for col in plot_feats:
        row_str = f"  {col:<22}"
        for lbl in AGE_LABELS:
            val = age_means.loc[col, lbl] if lbl in age_means.columns else np.nan
            row_str += f" {val:>10.3f}" if pd.notna(val) else f" {'N/A':>10}"
        log(row_str)

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    age_palette = ["#42a5f5", "#66bb6a", "#ffa726"]
    for i, col in enumerate(plot_feats):
        ax = axes[i]
        groups = [df_plot[df_plot["AgeGroup"] == lbl][col].dropna()
                  for lbl in AGE_LABELS]
        ax.boxplot(groups, labels=AGE_LABELS, patch_artist=True,
                   boxprops=dict(facecolor="#bbdefb"),
                   medianprops=dict(color="#e74c3c", linewidth=2))
        ax.set_title(col, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7)
        sns.despine(ax=ax)
    for j in range(n_feats, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Biomarker Distributions by Age Group",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_dir / "biomarkers_by_age_group.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Report ────────────────────────────────────────────────────────────────
    log(f"\n{sep2}")
    log("  DEMOGRAPHICS REPORT")
    log(sep2)
    log(f"\n  Observations:")
    log(f"    - NHANES is a US-based survey; ethnicity distribution reflects")
    log(f"      the US adult population, not Indian subcontinent demographics.")
    log(f"    - Sex differences are expected in body composition (BMD, lean mass,")
    log(f"      body fat %) and lipid profiles (HDL typically higher in females).")
    log(f"    - Age-related trends are expected in blood pressure, BMD, and")
    log(f"      metabolic markers -- these are biological, not confounders per se.")
    log(f"    - The clustering is unsupervised; demographic variables are NOT")
    log(f"      included in the feature matrix. Post-hoc analysis should check")
    log(f"      whether clusters are disproportionately age- or sex-stratified.")
    log(f"\n  Recommendation:")
    log(f"    No demographic adjustment is recommended before clustering.")
    log(f"    Post-clustering: compute demographic composition of each cluster")
    log(f"    to assess whether clusters are confounded by age or sex.")

    report_path = save_dir / "demographics_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [SAVED] {report_path}")
    print(sep)

    # Summary stats for the EDA summary
    gender_split = {g: int(c) for g, c in gender_counts.items()}
    eth_split    = {e: int(c) for e, c in eth_counts.items()}

    return {
        "age_mean":       round(float(df["RIDAGEYR"].mean()), 1),
        "age_median":     float(df["RIDAGEYR"].median()),
        "age_range":      (int(df["RIDAGEYR"].min()), int(df["RIDAGEYR"].max())),
        "gender_split":   gender_split,
        "ethnicity_split": eth_split,
        "recommendation": "No demographic adjustment recommended before clustering.",
    }
