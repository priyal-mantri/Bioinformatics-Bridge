"""
pipeline/eda/distributions.py
==============================
Phase 3 -- Univariate Distributions

For every feature variable, generates a histogram + KDE curve and
computes skewness and kurtosis.

Saves:
    distributions/individual/<VARNAME>_distribution.png  (one per variable)
    distributions/distributions_summary_grid.png
    distributions/distributions_summary.csv
    distributions/distribution_report.txt

Variables are NOT transformed.
This phase is DESCRIPTIVE ONLY.
"""

import math
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Skewness thresholds for classification
SKEW_HEAVY   = 2.0    # |skew| > 2 → heavily skewed
SKEW_MODERATE = 1.0   # 1 < |skew| <= 2 → moderately skewed


def _classify_skew(skew: float) -> str:
    """Return a human-readable skewness classification."""
    abs_skew = abs(skew)
    if abs_skew > SKEW_HEAVY:
        direction = "right" if skew > 0 else "left"
        return f"HEAVILY SKEWED ({direction})"
    elif abs_skew > SKEW_MODERATE:
        direction = "right" if skew > 0 else "left"
        return f"Moderately skewed ({direction})"
    else:
        return "Approximately symmetric"


def _plot_single_distribution(series: pd.Series, varname: str,
                               save_path: Path) -> None:
    """
    Plot histogram + KDE for a single variable and save as PNG.
    """
    data = series.dropna()
    if len(data) < 10:
        return   # not enough data to plot

    skew = float(data.skew())
    kurt = float(data.kurtosis())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{varname}  |  n={len(data):,}  |  skew={skew:.2f}  |  "
                 f"kurt={kurt:.2f}", fontsize=11, fontweight="bold")

    # Left: Histogram + KDE
    ax = axes[0]
    sns.histplot(data, kde=True, ax=ax, color="#2196F3", alpha=0.7,
                 line_kws={"linewidth": 2, "color": "#0D47A1"})
    ax.axvline(data.mean(),   color="#e74c3c", linestyle="--",
               linewidth=1.5, label=f"Mean={data.mean():.2f}")
    ax.axvline(data.median(), color="#2ecc71", linestyle="--",
               linewidth=1.5, label=f"Median={data.median():.2f}")
    ax.set_title("Histogram + KDE")
    ax.set_xlabel(varname)
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    sns.despine(ax=ax)

    # Right: Box + strip (shows spread and outliers)
    ax2 = axes[1]
    # Use a sample for the strip if n is large (performance)
    sample = data.sample(min(500, len(data)), random_state=42)
    ax2.boxplot(data, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#bbdefb", color="#1565c0"),
                medianprops=dict(color="#e74c3c", linewidth=2),
                whiskerprops=dict(color="#1565c0"),
                capprops=dict(color="#1565c0"),
                flierprops=dict(marker="o", markerfacecolor="#FF5722",
                                markersize=3, alpha=0.5))
    ax2.set_title("Box Plot")
    ax2.set_ylabel(varname)
    ax2.set_xticks([])
    sns.despine(ax=ax2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def run_distributions(df: pd.DataFrame, output_dir: Path,
                      feature_cols: list[str]) -> dict:
    """
    Phase 3: Univariate Distributions.

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
    save_dir     = output_dir / "distributions"
    indiv_dir    = save_dir / "individual"
    indiv_dir.mkdir(parents=True, exist_ok=True)

    sep  = "=" * 65
    sep2 = "-" * 65
    lines = []

    def log(text=""):
        lines.append(text)
        print(text)

    log(sep)
    log("  PHASE 3 -- UNIVARIATE DISTRIBUTIONS")
    log(sep)

    # ── Compute stats for every feature ─────────────────────────────────────
    stats_rows = []
    for col in feature_cols:
        data = df[col].dropna()
        if len(data) < 10:
            continue
        skew  = float(data.skew())
        kurt  = float(data.kurtosis())
        stats_rows.append({
            "variable":  col,
            "n_valid":   len(data),
            "mean":      round(data.mean(), 4),
            "std":       round(data.std(), 4),
            "min":       round(data.min(), 4),
            "p25":       round(data.quantile(0.25), 4),
            "median":    round(data.median(), 4),
            "p75":       round(data.quantile(0.75), 4),
            "max":       round(data.max(), 4),
            "skewness":  round(skew, 4),
            "kurtosis":  round(kurt, 4),
            "skew_class": _classify_skew(skew),
        })

    stats_df = pd.DataFrame(stats_rows).set_index("variable")
    stats_df.to_csv(save_dir / "distributions_summary.csv")

    # ── Print summary table ──────────────────────────────────────────────────
    log(f"\n  {'Variable':<22} {'N Valid':>8} {'Mean':>10} {'Skewness':>10} "
        f"{'Classification'}")
    log(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*30}")
    for var, row in stats_df.iterrows():
        log(f"  {var:<22} {int(row['n_valid']):>8,} {row['mean']:>10.3f} "
            f"{row['skewness']:>10.3f}  {row['skew_class']}")

    # ── Generate individual distribution plots ───────────────────────────────
    log(f"\n  Generating individual distribution plots ...")
    for col in feature_cols:
        if col not in stats_df.index:
            continue
        save_path = indiv_dir / f"{col}_distribution.png"
        _plot_single_distribution(df[col], col, save_path)
    log(f"  [SAVED] {len(feature_cols)} individual distribution plots -> "
        f"distributions/individual/")

    # ── Summary grid (all variables in one figure) ───────────────────────────
    log("  Generating summary distribution grid ...")
    valid_cols = [c for c in feature_cols if c in stats_df.index]
    n_cols_grid = 4
    n_rows_grid = math.ceil(len(valid_cols) / n_cols_grid)

    fig, axes = plt.subplots(n_rows_grid, n_cols_grid,
                              figsize=(n_cols_grid * 4, n_rows_grid * 3))
    axes = axes.flatten()

    for i, col in enumerate(valid_cols):
        data = df[col].dropna()
        ax = axes[i]
        sns.histplot(data, kde=True, ax=ax, color="#2196F3", alpha=0.65,
                     line_kws={"linewidth": 1.5, "color": "#0D47A1"},
                     bins=40)
        skew = stats_df.loc[col, "skewness"]
        color = "#e74c3c" if abs(skew) > SKEW_HEAVY else (
                "#FF9800" if abs(skew) > SKEW_MODERATE else "#4caf50")
        ax.set_title(f"{col}\nskew={skew:.2f}", fontsize=7.5, color=color,
                     fontweight="bold" if abs(skew) > SKEW_MODERATE else "normal")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=6)
        sns.despine(ax=ax)

    # Hide unused axes
    for j in range(len(valid_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribution Summary Grid\n"
                 "(red=heavily skewed, orange=moderate, green=symmetric)",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_dir / "distributions_summary_grid.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    log("  [SAVED] distributions_summary_grid.png")

    # ── Identify skewed variables ────────────────────────────────────────────
    heavily_skewed  = stats_df[stats_df["skewness"].abs() > SKEW_HEAVY].index.tolist()
    moderately_skewed = stats_df[
        (stats_df["skewness"].abs() > SKEW_MODERATE) &
        (stats_df["skewness"].abs() <= SKEW_HEAVY)
    ].index.tolist()

    log(f"\n{sep2}")
    log("  DISTRIBUTION REPORT")
    log(sep2)
    log(f"\n  Heavily skewed (|skew| > {SKEW_HEAVY})  -- {len(heavily_skewed)} variables:")
    for col in heavily_skewed:
        skew = stats_df.loc[col, "skewness"]
        log(f"    {col:<22}  skew={skew:.3f}")

    log(f"\n  Moderately skewed ({SKEW_MODERATE} < |skew| <= {SKEW_HEAVY}) "
        f"-- {len(moderately_skewed)} variables:")
    for col in moderately_skewed:
        skew = stats_df.loc[col, "skewness"]
        log(f"    {col:<22}  skew={skew:.3f}")

    # ── Recommendations ──────────────────────────────────────────────────────
    log(f"\n  Observations:")
    log(f"    - Variables with heavy right skew (HOMA_IR, TG_HDL_ratio, etc.) are")
    log(f"      biologically expected. These distributions reflect metabolic")
    log(f"      heterogeneity in the population, not data quality errors.")
    log(f"    - These variables WILL distort distance-based clustering algorithms")
    log(f"      and may require log-transformation during the scaling stage.")
    log(f"    - No transformation is applied here.")
    log(f"\n  Recommendation:")
    if len(heavily_skewed) > 0:
        log(f"    Recommend creation of Decision 007: Log-transformation of heavily")
        log(f"    skewed variables before clustering. Candidates: {heavily_skewed}")
    else:
        log(f"    No additional preprocessing recommended based on distributions.")

    recommendation = (
        f"Recommend creation of Decision 007: Log-transformation of "
        f"{len(heavily_skewed)} heavily skewed variables before clustering."
        if heavily_skewed else
        "No additional preprocessing recommended based on distributions."
    )

    report_path = save_dir / "distribution_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [SAVED] {report_path}")
    print(sep)

    return {
        "heavily_skewed":    heavily_skewed,
        "moderately_skewed": moderately_skewed,
        "stats_df":          stats_df,
        "recommendation":    recommendation,
    }
