"""
run_eda.py
==========
Entry point for the NHANES EDA pipeline.

Runs all 6 EDA phases in sequence and generates a final summary report.

EDA is EXPLORATORY ONLY.
No data is modified, no participants are removed, no variables are transformed.
If any phase discovers an issue that warrants preprocessing, it will
recommend a new preprocessing decision (Decision 007+).

Usage:
    python run_eda.py

Output:
    output/eda/
        overview/           -- Phase 1 files
        missingness/        -- Phase 2 files
        distributions/      -- Phase 3 files
        outliers/           -- Phase 4 files
        correlations/       -- Phase 5 files
        demographics/       -- Phase 6 files
        EDA_SUMMARY.md      -- Final summary report
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

from pipeline.config import OUTPUT_DIR
from pipeline.eda.overview       import run_overview
from pipeline.eda.missingness    import run_missingness
from pipeline.eda.distributions  import run_distributions
from pipeline.eda.outliers       import run_outliers
from pipeline.eda.correlations   import run_correlations
from pipeline.eda.demographics   import run_demographics


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PREPROCESSED_FILE = OUTPUT_DIR / "preprocessed.csv"
EDA_OUTPUT_DIR    = OUTPUT_DIR / "eda"


# ---------------------------------------------------------------------------
# Feature column definitions
# These are the variables that will go into the clustering feature matrix.
# Control, flag, and raw repeated BP columns are excluded from feature-level
# analysis (but included in the full dataset for the overview phase).
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Body Composition
    "BMXBMI", "BMXWAIST",
    "DXDTOBMD", "DXDTOPF", "DXDTOLE",
    # Cardiovascular (averaged readings)
    "Avg_Systolic_BP", "Avg_Diastolic_BP", "BPXPLS",
    # Glucose Metabolism
    "LBXGLU", "LBXIN", "HOMA_IR",
    # Lipid Metabolism
    "LBXTC", "LBDHDD", "LBXSTR", "TC_HDL_ratio", "TG_HDL_ratio",
    # Hepatic Function
    "LBXSATSI", "LBXSAL", "LBXSTP", "LBXSTB",
    # Renal Function
    "LBXSCR", "LBXSUA", "LBXSBU",
    # Electrolyte / Mineral
    "LBXSCA", "LBXSPH", "LBXSNASI", "LBXSKSI",
]

DEMOGRAPHIC_COLS = ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3"]


# ---------------------------------------------------------------------------
# EDA Summary generator
# ---------------------------------------------------------------------------

def _generate_eda_summary(results: dict, output_dir: Path) -> None:
    """
    Compile all phase results into a single EDA_SUMMARY.md file.
    """
    ov   = results["overview"]
    ms   = results["missingness"]
    dist = results["distributions"]
    out  = results["outliers"]
    corr = results["correlations"]
    demo = results["demographics"]

    all_recommendations = []
    if dist["heavily_skewed"]:
        all_recommendations.append(dist["recommendation"])
    for rec in out["recommendations"]:
        if "Decision 007" in rec:
            all_recommendations.append(rec)
    for rec in corr["recommendations"]:
        if "Decision 008" in rec:
            all_recommendations.append(rec)

    final_verdict = (
        "**Additional preprocessing recommended before clustering.**\n"
        "See recommendations below."
        if all_recommendations else
        "**Dataset ready for scaling and clustering.**"
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# EDA Summary Report",
        "## Research 2: The Bioinformatics Bridge",
        f"### Generated: {now}",
        "",
        "---",
        "",
        "## Final Recommendation",
        "",
        final_verdict,
        "",
        "---",
        "",
        "## Phase 1 — Dataset Overview",
        "",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| Rows | {ov['n_rows']:,} |",
        f"| Columns | {ov['n_cols']} |",
        f"| Feature columns | {ov['n_feature_cols']} |",
        f"| Demographic/control cols | {ov['n_demo_cols']} |",
        f"| Flag cols | {ov['n_flag_cols']} |",
        f"| Memory | {ov['mem_mb']} MB |",
        "",
        "---",
        "",
        "## Phase 2 — Missing Data Analysis",
        "",
        f"- **Total participants:** {ov['n_rows']:,}",
        f"- **Complete cases (all {ov['n_feature_cols']} features):** "
        f"{ms['n_complete_all']:,} "
        f"({ms['n_complete_all']/ov['n_rows']*100:.1f}%)",
        f"- **Complete cases (core features only, no DEXA/fasting):** "
        f"{ms['n_complete_core']:,} "
        f"({ms['n_complete_core']/ov['n_rows']*100:.1f}%)",
        "",
        "**Variables with high missingness (> 40%):**",
        "",
    ]

    for col in ms["high_missing_vars"]:
        pct = ms["high_missing_pcts"][col]
        note = "[expected — DEXA/fasting subsample]" if col in {
            "DXDTOBMD","DXDTOPF","DXDTOLE","DXAEXSTS",
            "LBXGLU","LBXIN","HOMA_IR","LBDINLC"
        } else "[INVESTIGATE]"
        lines.append(f"- `{col}`: {pct:.1f}%  {note}")

    if ms["unexpected_high"]:
        lines += [
            "",
            "> [!WARNING]",
            f"> Unexpected high missingness: {ms['unexpected_high']}",
        ]

    lines += [
        "",
        f"**{ms['recommendation']}**",
        "",
        "---",
        "",
        "## Phase 3 — Univariate Distributions",
        "",
        f"**Heavily skewed variables (|skew| > 2.0)  — {len(dist['heavily_skewed'])}:**",
        "",
    ]

    stats_df = dist["stats_df"]
    for col in dist["heavily_skewed"]:
        skew = stats_df.loc[col, "skewness"] if col in stats_df.index else "N/A"
        lines.append(f"- `{col}`: skew = {skew:.3f}")

    lines += [
        "",
        f"**Moderately skewed variables (1.0 < |skew| <= 2.0) — "
        f"{len(dist['moderately_skewed'])}:**",
        "",
    ]
    for col in dist["moderately_skewed"]:
        skew = stats_df.loc[col, "skewness"] if col in stats_df.index else "N/A"
        lines.append(f"- `{col}`: skew = {skew:.3f}")

    lines += [
        "",
        f"> [!IMPORTANT]",
        f"> {dist['recommendation']}",
        "",
        "---",
        "",
        "## Phase 4 — Outlier Investigation",
        "",
    ]

    outlier_df = out["outlier_df"]
    if out["vars_requiring_inv"]:
        lines.append("**Variables with extreme values requiring investigation:**")
        lines.append("")
        for col in out["vars_requiring_inv"]:
            row = outlier_df.loc[col]
            lines.append(
                f"- `{col}`: max={row['max_value']:.3f}, "
                f"extreme outliers={int(row['n_extreme_outliers'])}, "
                f"requires_investigation={int(row['n_requires_inv'])}"
            )
    else:
        lines.append("No variables with values outside biological plausibility ranges.")

    lines += [""]
    for rec in out["recommendations"]:
        lines.append(f"> [!IMPORTANT]")
        lines.append(f"> {rec}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Phase 5 — Correlation & Multicollinearity",
        "",
        f"- High correlation pairs (|r| >= 0.75): **{corr['n_high_corr_pairs']}**",
        f"- Very high correlation pairs (|r| >= 0.90): **{corr['n_very_high_pairs']}**",
        "",
        "**Very high correlation pairs (|r| >= 0.90):**",
        "",
    ]

    if corr["very_high_pairs"]:
        for p in corr["very_high_pairs"]:
            lines.append(
                f"- `{p['variable_1']}` ↔ `{p['variable_2']}`: "
                f"r = {p['pearson_r']:+.4f}"
            )
    else:
        lines.append("None.")

    if corr["flagged_structural"]:
        lines += [
            "",
            "**Structurally redundant pairs (derived from each other):**",
            "",
        ]
        for c1, c2 in corr["flagged_structural"]:
            lines.append(f"- `{c1}` + `{c2}`")

    lines += [""]
    for rec in corr["recommendations"]:
        lines.append(f"> [!IMPORTANT]")
        lines.append(f"> {rec}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Phase 6 — Demographic Profile",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Age range | {demo['age_range'][0]}–{demo['age_range'][1]} years |",
        f"| Age mean | {demo['age_mean']} years |",
        f"| Age median | {demo['age_median']:.0f} years |",
    ]

    for gender, count in demo["gender_split"].items():
        lines.append(
            f"| {gender} | {count:,} ({count/ov['n_rows']*100:.1f}%) |"
        )

    lines += [
        "",
        "**Ethnicity breakdown:**",
        "",
    ]
    for eth, count in demo["ethnicity_split"].items():
        lines.append(f"- {eth}: {count:,} ({count/ov['n_rows']*100:.1f}%)")

    lines += [
        "",
        f"> [!NOTE]",
        f"> {demo['recommendation']}",
        "",
        "---",
        "",
        "## Recommended New Preprocessing Decisions",
        "",
    ]

    if all_recommendations:
        for i, rec in enumerate(all_recommendations, start=7):
            lines.append(f"### Decision {i:03d}")
            lines.append("")
            lines.append(rec)
            lines.append("")
    else:
        lines.append("No new preprocessing decisions recommended at this time.")
        lines.append("")

    lines += [
        "---",
        "",
        "## Final Recommendation",
        "",
        final_verdict,
        "",
        "---",
        f"*Generated by run_eda.py · {now}*",
    ]

    summary_path = output_dir / "EDA_SUMMARY.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [SAVED] {summary_path}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:

    if not PREPROCESSED_FILE.exists():
        raise FileNotFoundError(
            f"[ERROR] {PREPROCESSED_FILE} not found.\n"
            f"        Run 'python run_preprocess.py' first."
        )

    print("\n" + "=" * 65)
    print("  NHANES EDA PIPELINE")
    print("  Research 2: The Bioinformatics Bridge")
    print("=" * 65)
    print(f"\n  Input  : {PREPROCESSED_FILE}")
    print(f"  Output : {EDA_OUTPUT_DIR}")

    # Create output subdirectories
    for subdir in ["overview", "missingness", "distributions",
                   "outliers", "correlations", "demographics"]:
        (EDA_OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(PREPROCESSED_FILE, low_memory=False)
    print(f"\n  Loaded : {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    # Validate feature columns exist
    missing_feats = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_feats:
        raise ValueError(
            f"[ERROR] The following feature columns are missing from the "
            f"preprocessed file:\n{missing_feats}\n"
            f"Check that run_preprocess.py has been run successfully."
        )

    results = {}

    # Phase 1 — Overview
    results["overview"] = run_overview(df, EDA_OUTPUT_DIR)

    # Phase 2 — Missingness
    results["missingness"] = run_missingness(df, EDA_OUTPUT_DIR, FEATURE_COLS)

    # Phase 3 — Distributions
    results["distributions"] = run_distributions(df, EDA_OUTPUT_DIR, FEATURE_COLS)

    # Phase 4 — Outliers
    results["outliers"] = run_outliers(df, EDA_OUTPUT_DIR, FEATURE_COLS)

    # Phase 5 — Correlations
    results["correlations"] = run_correlations(df, EDA_OUTPUT_DIR, FEATURE_COLS)

    # Phase 6 — Demographics
    results["demographics"] = run_demographics(df, EDA_OUTPUT_DIR, FEATURE_COLS)

    # Generate EDA_SUMMARY.md
    print("\n" + "=" * 65)
    print("  GENERATING EDA_SUMMARY.md")
    print("=" * 65)
    _generate_eda_summary(results, EDA_OUTPUT_DIR)

    print("\n" + "=" * 65)
    print("  EDA PIPELINE COMPLETE")
    print("=" * 65)
    print(f"\n  All outputs saved under: {EDA_OUTPUT_DIR}")
    print(f"  Final report           : {EDA_OUTPUT_DIR / 'EDA_SUMMARY.md'}\n")


if __name__ == "__main__":
    main()
