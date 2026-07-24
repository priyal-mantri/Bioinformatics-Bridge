"""
pipeline/eda/correlations.py
=============================
Phase 5 -- Correlation & Multicollinearity

Computes the Pearson correlation matrix for all feature variables,
identifies highly correlated pairs, and flags potential redundancy.

Saves:
    correlations/correlation_heatmap.png
    correlations/correlation_matrix.csv
    correlations/high_correlations.csv
    correlations/correlation_report.txt

Variables are NOT removed.
If removal appears scientifically justified, a new preprocessing decision
is recommended.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Correlation thresholds
HIGH_CORR_THRESHOLD = 0.75   # |r| >= 0.75 → high correlation (flag)
VERY_HIGH_THRESHOLD = 0.90   # |r| >= 0.90 → very high (strong redundancy)


def run_correlations(df: pd.DataFrame, output_dir: Path,
                     feature_cols: list[str]) -> dict:
    """
    Phase 5: Correlation & Multicollinearity.

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
    save_dir = output_dir / "correlations"
    save_dir.mkdir(parents=True, exist_ok=True)

    sep  = "=" * 65
    sep2 = "-" * 65
    lines = []

    def log(text=""):
        lines.append(text)
        print(text)

    log(sep)
    log("  PHASE 5 -- CORRELATION & MULTICOLLINEARITY")
    log(sep)

    # ── Compute Pearson correlation matrix ───────────────────────────────────
    # Use pairwise complete observations (each pair uses all rows where
    # BOTH variables are non-null)
    corr_matrix = df[feature_cols].corr(method="pearson", min_periods=30)

    corr_matrix.to_csv(save_dir / "correlation_matrix.csv")
    log(f"\n  Pearson correlation matrix computed.")
    log(f"  Variables included : {len(feature_cols)}")
    log(f"  Method             : pairwise complete observations (min_periods=30)")

    # ── Heatmap ──────────────────────────────────────────────────────────────
    log("\n  Generating correlation heatmap ...")

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # upper triangle

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
        square=True,
        linewidths=0.3,
        linecolor="#cccccc",
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix — Feature Variables",
                 fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(axis="both", labelsize=8)
    plt.tight_layout()
    fig.savefig(save_dir / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log("  [SAVED] correlation_heatmap.png")

    # ── Extract high-correlation pairs ───────────────────────────────────────
    high_pairs = []
    n = len(feature_cols)
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = feature_cols[i], feature_cols[j]
            r = corr_matrix.loc[c1, c2]
            if pd.notna(r) and abs(r) >= HIGH_CORR_THRESHOLD:
                high_pairs.append({
                    "variable_1": c1,
                    "variable_2": c2,
                    "pearson_r":  round(r, 4),
                    "abs_r":      round(abs(r), 4),
                    "level": ("VERY HIGH" if abs(r) >= VERY_HIGH_THRESHOLD
                              else "HIGH"),
                })

    high_pairs_df = pd.DataFrame(
        sorted(high_pairs, key=lambda x: -x["abs_r"])
    )
    if not high_pairs_df.empty:
        high_pairs_df.to_csv(save_dir / "high_correlations.csv", index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    log(f"\n{sep2}")
    log("  CORRELATION REPORT")
    log(sep2)

    very_high = [p for p in high_pairs if p["abs_r"] >= VERY_HIGH_THRESHOLD]
    high_only = [p for p in high_pairs if p["abs_r"] < VERY_HIGH_THRESHOLD]

    log(f"\n  Very high correlations (|r| >= {VERY_HIGH_THRESHOLD}) "
        f"-- {len(very_high)} pairs:")
    if very_high:
        for p in very_high:
            log(f"    {p['variable_1']:<22} <--> {p['variable_2']:<22}  "
                f"r={p['pearson_r']:+.4f}  [{p['level']}]")
    else:
        log("    None.")

    log(f"\n  High correlations ({HIGH_CORR_THRESHOLD} <= |r| < {VERY_HIGH_THRESHOLD}) "
        f"-- {len(high_only)} pairs:")
    if high_only:
        for p in high_only:
            log(f"    {p['variable_1']:<22} <--> {p['variable_2']:<22}  "
                f"r={p['pearson_r']:+.4f}  [{p['level']}]")
    else:
        log("    None.")

    # Biological interpretation of notable pairs
    log(f"\n  Biological interpretation of high-correlation pairs:")
    notable_notes = {
        frozenset({"LBXTC", "TC_HDL_ratio"}):
            "Expected: TC/HDL ratio is computed from TC -- structural correlation.",
        frozenset({"LBXSTR", "TG_HDL_ratio"}):
            "Expected: TG/HDL ratio is computed from LBXSTR -- structural correlation.",
        frozenset({"LBXIN", "HOMA_IR"}):
            "Expected: HOMA_IR = (Glucose * Insulin) / 405 -- structural correlation.",
        frozenset({"LBXGLU", "HOMA_IR"}):
            "Expected: HOMA_IR = (Glucose * Insulin) / 405 -- structural correlation.",
        frozenset({"BPXSY1", "Avg_Systolic_BP"}):
            "Expected: Avg_Systolic_BP is the mean of BPXSY1-3.",
        frozenset({"BPXDI1", "Avg_Diastolic_BP"}):
            "Expected: Avg_Diastolic_BP is the mean of BPXDI1-3.",
    }
    for p in high_pairs:
        pair_set = frozenset({p["variable_1"], p["variable_2"]})
        if pair_set in notable_notes:
            log(f"    {p['variable_1']:<20} <--> {p['variable_2']:<20}: "
                f"{notable_notes[pair_set]}")

    # ── Recommendations ──────────────────────────────────────────────────────
    recommendations = []

    # Detect structurally redundant pairs (derived from each other)
    structurally_redundant = [
        ("LBXTC",  "TC_HDL_ratio"),
        ("LBXSTR", "TG_HDL_ratio"),
        ("LBXIN",  "HOMA_IR"),
        ("LBXGLU", "HOMA_IR"),
    ]
    flagged_structural = [(c1, c2) for c1, c2 in structurally_redundant
                          if c1 in feature_cols and c2 in feature_cols]

    if flagged_structural:
        log(f"\n  Note on structurally redundant pairs:")
        log(f"    The following pairs include a raw variable AND a derived")
        log(f"    variable computed from it. Including both may double-weight")
        log(f"    this biological signal in the feature matrix.")
        for c1, c2 in flagged_structural:
            log(f"    {c1} + {c2}")
        recommendations.append(
            "Recommend creation of Decision 008: Feature selection — decide whether "
            "to include both raw and derived variables that are structurally correlated "
            "(e.g. LBXTC + TC_HDL_ratio, LBXIN + HOMA_IR). Retaining both double-weights "
            "those biological signals in PCA/clustering."
        )
    else:
        recommendations.append(
            "No structural redundancy detected. No variable removal recommended."
        )

    log(f"\n  Recommendations:")
    for rec in recommendations:
        log(f"    {rec}")

    report_path = save_dir / "correlation_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [SAVED] {report_path}")
    print(sep)

    return {
        "n_high_corr_pairs":    len(high_pairs),
        "n_very_high_pairs":    len(very_high),
        "very_high_pairs":      very_high,
        "high_pairs":           high_pairs,
        "flagged_structural":   flagged_structural,
        "recommendations":      recommendations,
    }
