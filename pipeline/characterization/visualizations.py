"""
Visualizations module for Stage 8 phenotype characterization.
Generates publication-quality analytical figures using Matplotlib and Seaborn.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11


def plot_zscore_heatmap(
    profile_df: pd.DataFrame,
    retained_k: int,
    cohort_title: str,
    output_filepath: str
) -> None:
    """Generates standardized Z-score heatmap for primary physiological features."""
    z_cols = [f"c{c}_zscore" for c in range(retained_k)]
    features = profile_df["feature"].values
    data_matrix = profile_df[z_cols].values

    fig, ax = plt.subplots(figsize=(8, max(6, len(features) * 0.35)))
    sns.heatmap(
        data_matrix,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0.0,
        vmin=-1.2,
        vmax=1.2,
        yticklabels=features,
        xticklabels=[f"Cluster {c}" for c in range(retained_k)],
        cbar_kws={"label": "Standardized Cluster Z-Score"},
        ax=ax,
    )
    ax.set_title(f"Physiological Profile Heatmap — {cohort_title}", pad=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300)
    plt.close()


def plot_discriminating_boxplots(
    df: pd.DataFrame,
    cluster_col: str,
    selected_features: List[str],
    cohort_title: str,
    output_filepath: str
) -> None:
    """Generates box plots for selected discriminating physiological variables."""
    n_feats = len(selected_features)
    if n_feats == 0:
        return

    n_cols = min(3, n_feats)
    n_rows = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    palette = sns.color_palette("Set2", len(df[cluster_col].unique()))

    for idx, feat in enumerate(selected_features):
        ax = axes[idx]
        sns.boxplot(
            x=cluster_col,
            y=feat,
            data=df,
            palette=palette,
            ax=ax,
            width=0.4,
            fliersize=2
        )
        ax.set_title(feat, fontweight="bold")
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Value")

    # Hide unused axes
    for idx in range(n_feats, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Selected Discriminating Physiological Variables — {cohort_title}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_derived_ratios(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    output_filepath: str
) -> None:
    """Generates comparative box plots for derived metabolic ratios across Cohorts A and B."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    ratios = ["TC_HDL_ratio", "TG_HDL_ratio", "HOMA_IR"]
    titles = ["TC / HDL Ratio", "TG / HDL Ratio", "HOMA-IR"]

    palette_a = sns.color_palette("Set1", 2)
    palette_b = sns.color_palette("Set2", 3)

    for i, (r, t) in enumerate(zip(ratios, titles)):
        # Cohort A
        ax_a = axes[0, i]
        if r in df_a.columns:
            sns.boxplot(x="cluster", y=r, data=df_a, palette=palette_a, ax=ax_a, width=0.4, fliersize=1)
            ax_a.set_title(f"Cohort A: {t}", fontweight="bold")
            ax_a.set_xlabel("Cluster ID")

        # Cohort B
        ax_b = axes[1, i]
        if r in df_b.columns:
            sns.boxplot(x="cluster", y=r, data=df_b, palette=palette_b, ax=ax_b, width=0.4, fliersize=1)
            ax_b.set_title(f"Cohort B: {t}", fontweight="bold")
            ax_b.set_xlabel("Cluster ID")

    fig.suptitle("Post-Hoc Derived Metabolic Ratios (Excluded from Primary Clustering)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300)
    plt.close()


def plot_literature_matrix(
    conc_a: pd.DataFrame,
    conc_b: pd.DataFrame,
    output_filepath: str
) -> None:
    """Generates structured literature concordance summary matrix plot for Tier A & B features."""
    tier_ab_a = conc_a[conc_a["evidence_tier"].str.startswith(("A", "B"))].copy()
    tier_ab_b = conc_b[conc_b["evidence_tier"].str.startswith(("A", "B"))].copy()

    # Map concordance categories to numeric colors
    cmap_dict = {"Concordant": 1, "Discordant": -1, "Unclear / Insufficient Evidence": 0}
    
    features_a = sorted(tier_ab_a["feature"].unique())
    clusters_a = sorted(tier_ab_a["cluster_id"].unique())
    
    mat_a = np.zeros((len(features_a), len(clusters_a)))
    for i, f in enumerate(features_a):
        for j, c in enumerate(clusters_a):
            sub = tier_ab_a[(tier_ab_a["feature"] == f) & (tier_ab_a["cluster_id"] == c)]
            if not sub.empty:
                mat_a[i, j] = cmap_dict.get(sub["concordance"].values[0], 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), gridspec_kw={"width_ratios": [1, 1.5]})

    sns.heatmap(
        mat_a,
        annot=True,
        cbar=False,
        cmap="coolwarm",
        yticklabels=features_a,
        xticklabels=[f"Cluster {c}" for c in clusters_a],
        ax=ax1,
        center=0,
    )
    ax1.set_title("Cohort A (K=2) Concordance Matrix", fontweight="bold")

    features_b = sorted(tier_ab_b["feature"].unique())
    clusters_b = sorted(tier_ab_b["cluster_id"].unique())
    
    mat_b = np.zeros((len(features_b), len(clusters_b)))
    for i, f in enumerate(features_b):
        for j, c in enumerate(clusters_b):
            sub = tier_ab_b[(tier_ab_b["feature"] == f) & (tier_ab_b["cluster_id"] == c)]
            if not sub.empty:
                mat_b[i, j] = cmap_dict.get(sub["concordance"].values[0], 0)

    sns.heatmap(
        mat_b,
        annot=True,
        cbar=False,
        cmap="coolwarm",
        yticklabels=features_b,
        xticklabels=[f"Cluster {c}" for c in clusters_b],
        ax=ax2,
        center=0,
    )
    ax2.set_title("Cohort B (K=3) Concordance Matrix", fontweight="bold")

    fig.suptitle("Literature Concordance Matrix (1: Concordant, 0: Unclear, -1: Discordant)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_filepath, dpi=300)
    plt.close()


def generate_all_plots(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    profile_a: pd.DataFrame,
    profile_b: pd.DataFrame,
    stats_a: pd.DataFrame,
    stats_b: pd.DataFrame,
    conc_a: pd.DataFrame,
    conc_b: pd.DataFrame,
    plots_dir: str
) -> List[str]:
    """Generates all Stage 8 publication-quality figures and returns list of output file paths."""
    os.makedirs(plots_dir, exist_ok=True)
    generated_plots = []

    # 1. Z-score Heatmaps
    fig1 = os.path.join(plots_dir, "fig1_cohort_a_zscore_heatmap.png")
    plot_zscore_heatmap(profile_a, 2, "Cohort A (Broad, K=2)", fig1)
    generated_plots.append(fig1)

    fig2 = os.path.join(plots_dir, "fig2_cohort_b_zscore_heatmap.png")
    plot_zscore_heatmap(profile_b, 3, "Cohort B (Fasting/DEXA, K=3)", fig2)
    generated_plots.append(fig2)

    # 2. Discriminating Boxplots (Top 6 features by Z-score variance)
    top_a = profile_a.sort_values(by="c0_zscore", key=abs, ascending=False)["feature"].head(6).tolist()
    fig3 = os.path.join(plots_dir, "fig3_cohort_a_discriminating_features_boxplots.png")
    plot_discriminating_boxplots(df_a, "cluster", top_a, "Cohort A (Broad, K=2)", fig3)
    generated_plots.append(fig3)

    top_b = profile_b.sort_values(by="c0_zscore", key=abs, ascending=False)["feature"].head(6).tolist()
    fig4 = os.path.join(plots_dir, "fig4_cohort_b_discriminating_features_boxplots.png")
    plot_discriminating_boxplots(df_b, "cluster", top_b, "Cohort B (Fasting/DEXA, K=3)", fig4)
    generated_plots.append(fig4)

    # 3. Derived Ratios
    fig5 = os.path.join(plots_dir, "fig5_derived_metabolic_ratios.png")
    plot_derived_ratios(df_a, df_b, fig5)
    generated_plots.append(fig5)

    # 4. Literature Matrix
    fig6 = os.path.join(plots_dir, "fig6_literature_concordance_matrix.png")
    plot_literature_matrix(conc_a, conc_b, fig6)
    generated_plots.append(fig6)

    return generated_plots
