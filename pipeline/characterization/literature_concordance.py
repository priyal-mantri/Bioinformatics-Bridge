"""
Literature concordance module for Stage 8 phenotype characterization.
Implements the 4-dimension literature concordance evaluation anchored to
Research2_Final_Operationalization.md.

Dimensions:
1. Cluster Directional Displacement: Higher (Z > +0.20), Lower (Z < -0.20), Near cohort reference (|Z| <= 0.20)
2. Statistical Evidence: q < 0.05 or q >= 0.05, raw p-value, effect size (reported separately)
3. Literature Evidence: Tier A/B/C/D, reported literature direction, population/sex qualifications
4. Concordance Interpretation: Concordant, Discordant, Unclear / Insufficient Evidence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any

LITERATURE_OPERATIONALIZATION = {
    "BMXBMI": {
        "system": "Body composition",
        "evidence_tier": "B — indirect",
        "lit_direction": "Higher",
        "qualification": "General adult body-size phenotype proxy",
        "limitation": "Influenced by age, diet, activity; not direct lifelong Prakriti.",
    },
    "BMXWAIST": {
        "system": "Body composition",
        "evidence_tier": "B — indirect",
        "lit_direction": "Higher",
        "qualification": "Abdominal adiposity proxy",
        "limitation": "Plausible phenotype proxy, not an established biomarker.",
    },
    "DXDTOBMD": {
        "system": "Body composition",
        "evidence_tier": "C — exploratory",
        "lit_direction": "Exploratory / Unspecified",
        "qualification": "Total body bone mineral density",
        "limitation": "No direct Dosha-specific BMD association established in audited sources.",
    },
    "DXDTOPF": {
        "system": "Body composition",
        "evidence_tier": "B — indirect",
        "lit_direction": "Higher",
        "qualification": "Percent body fat",
        "limitation": "Measures current adiposity, not constitutional body type.",
    },
    "DXDTOLE": {
        "system": "Body composition",
        "evidence_tier": "B — indirect",
        "lit_direction": "Higher",
        "qualification": "Lean body mass / muscularity",
        "limitation": "Useful physiological dimension, no direct numerical Dosha mapping.",
    },
    "BPXPLS": {
        "system": "Cardiovascular / autonomic",
        "evidence_tier": "B — indirect",
        "lit_direction": "Higher",
        "qualification": "Resting pulse rate (pace/movement)",
        "limitation": "Single examination pulse affected by acute anxiety, caffeine, fitness.",
    },
    "Avg_Systolic_BP": {
        "system": "Cardiovascular / autonomic",
        "evidence_tier": "B — indirect",
        "lit_direction": "Exploratory / Unspecified",
        "qualification": "Vascular state",
        "limitation": "Audited literature did not establish clear baseline BP difference across healthy groups.",
    },
    "Avg_Diastolic_BP": {
        "system": "Cardiovascular / autonomic",
        "evidence_tier": "B — indirect",
        "lit_direction": "Exploratory / Unspecified",
        "qualification": "Vascular state",
        "limitation": "Medication, age, current health strongly affect measurement.",
    },
    "LBXGLU": {
        "system": "Glucose / metabolic",
        "evidence_tier": "B — indirect",
        "lit_direction": "Exploratory / Unspecified",
        "qualification": "Blood glucose handling",
        "limitation": "Current glycemic state, not subjective digestive capacity.",
    },
    "LBXIN": {
        "system": "Glucose / metabolic",
        "evidence_tier": "B — indirect",
        "lit_direction": "Exploratory / Unspecified",
        "qualification": "Endocrine glucose regulation",
        "limitation": "Physiological proxy, not direct digestive capacity.",
    },
    "LBXTC": {
        "system": "Lipid metabolism",
        "evidence_tier": "A — direct, male-specific",
        "lit_direction": "Higher",
        "qualification": "Male-specific (Kapha males)",
        "limitation": "Higher baseline TC reported in Kapha males; female differences absent.",
    },
    "LBDHDD": {
        "system": "Lipid metabolism",
        "evidence_tier": "A — direct, male-specific",
        "lit_direction": "Lower",
        "qualification": "Male-specific (Kapha males)",
        "limitation": "Lower HDL reported in Kapha males; do not generalize as universal.",
    },
    "LBXSTR": {
        "system": "Lipid metabolism",
        "evidence_tier": "A — direct, male-specific",
        "lit_direction": "Higher",
        "qualification": "Male-specific (Kapha males)",
        "limitation": "Higher TG reported in Kapha males; cited sample was young adults.",
    },
    "LBXSATSI": {
        "system": "Hepatic / biliary",
        "evidence_tier": "A — direct, male-specific",
        "lit_direction": "Higher",
        "qualification": "Male-specific ALT (Kapha males)",
        "limitation": "Higher ALT reported in Kapha males; retain sex qualification.",
    },
    "LBXSTB": {
        "system": "Hepatic / biliary",
        "evidence_tier": "B — indirect",
        "lit_direction": "Higher",
        "qualification": "Total bilirubin (Pitta biliary/pigmentation)",
        "limitation": "Functional alignment; empirical source showed no baseline elevation.",
    },
    "LBXSAL": {
        "system": "Hepatic / biliary",
        "evidence_tier": "C — weak/speculative",
        "lit_direction": "Exploratory / Unspecified",
        "qualification": "Albumin",
        "limitation": "General clinical relevance only.",
    },
    "LBXSTP": {
        "system": "Hepatic / biliary",
        "evidence_tier": "C — weak/speculative",
        "lit_direction": "Exploratory / Unspecified",
        "qualification": "Total protein",
        "limitation": "No direct association established.",
    },
    "LBXSUA": {
        "system": "Renal / waste clearance",
        "evidence_tier": "A — direct, male-specific",
        "lit_direction": "Higher",
        "qualification": "Male-specific (Kapha males)",
        "limitation": "Higher serum uric acid reported in Kapha males.",
    },
    "LBXSCR": {
        "system": "Renal / waste clearance",
        "evidence_tier": "B — indirect",
        "lit_direction": "Exploratory / Unspecified",
        "qualification": "Creatinine",
        "limitation": "No baseline difference established across groups.",
    },
    "LBXSBU": {
        "system": "Renal / waste clearance",
        "evidence_tier": "B — indirect",
        "lit_direction": "Exploratory / Unspecified",
        "qualification": "Blood urea nitrogen",
        "limitation": "Physiologically relevant, no direct baseline association.",
    },
    "LBXSPH": {
        "system": "Electrolytes / minerals",
        "evidence_tier": "B — indirect, female-specific",
        "lit_direction": "Higher",
        "qualification": "Female-specific (Pitta females)",
        "limitation": "Higher phosphorus reported in Pitta females; substantial variance.",
    },
    "LBXSCA": {
        "system": "Electrolytes / minerals",
        "evidence_tier": "C — weak/speculative",
        "lit_direction": "Exploratory / Unspecified",
        "qualification": "Calcium",
        "limitation": "Tightly homeostatically regulated; no established association.",
    },
    "LBXSNASI": {
        "system": "Electrolytes / minerals",
        "evidence_tier": "C — weak/speculative",
        "lit_direction": "Exploratory / Unspecified",
        "qualification": "Sodium",
        "limitation": "Strong homeostatic regulation.",
    },
    "LBXSKSI": {
        "system": "Electrolytes / minerals",
        "evidence_tier": "C — weak/speculative",
        "lit_direction": "Exploratory / Unspecified",
        "qualification": "Potassium",
        "limitation": "Not a Dosha biomarker; tightly regulated.",
    },
}


def evaluate_literature_concordance(
    profile_df: pd.DataFrame, stats_df: pd.DataFrame, cluster_id: int
) -> pd.DataFrame:
    """
    Evaluates 4-dimension literature concordance for a specific cluster.

    Dimensions:
    1. Cluster Directional Displacement (Z-score based)
    2. Statistical Evidence (p-value, q-value, effect size)
    3. Literature Evidence (Tier A/B/C/D, reported direction, qualifications)
    4. Concordance Interpretation (Concordant, Discordant, Unclear / Insufficient Evidence)
    """
    z_col = f"c{cluster_id}_zscore"
    if z_col not in profile_df.columns:
        raise KeyError(f"Z-score column '{z_col}' not found in profile DataFrame.")

    # Merge profile and stats DataFrames
    merged = profile_df.merge(stats_df, on="feature", how="inner")

    records = []
    for _, row in merged.iterrows():
        feat = row["feature"]
        z_val = row[z_col]
        p_val = row["p_primary"]
        q_val = row.get("q_primary_fdr", p_val)
        eff_name = row.get("effect_size_primary_name", "")
        eff_val = row.get("effect_size_primary_val", np.nan)

        # 1. Cluster Directional Displacement
        if z_val > 0.20:
            cluster_direction = "Higher"
        elif z_val < -0.20:
            cluster_direction = "Lower"
        else:
            cluster_direction = "Near cohort reference"

        # 2. Statistical Evidence
        stat_significance = "q < 0.05" if q_val < 0.05 else "q >= 0.05"

        # 3. Literature Evidence Baseline
        lit_meta = LITERATURE_OPERATIONALIZATION.get(
            feat,
            {
                "system": "Other",
                "evidence_tier": "D — unsupported",
                "lit_direction": "Exploratory / Unspecified",
                "qualification": "N/A",
                "limitation": "No literature mapping.",
            },
        )

        tier = lit_meta["evidence_tier"]
        lit_dir = lit_meta["lit_direction"]
        qualification = lit_meta["qualification"]
        limitation = lit_meta["limitation"]

        # 4. Concordance Interpretation
        # Rules:
        # - "Unclear / Insufficient Evidence" when:
        #   * cluster_direction == "Near cohort reference" (|Z| <= 0.20)
        #   * lit_dir == "Exploratory / Unspecified" or Tier starts with 'C' or 'D'
        #   * population/sex qualification prevents interpretation
        # - Otherwise:
        #   * "Concordant" if cluster_direction == lit_dir
        #   * "Discordant" if cluster_direction != lit_dir (opposite direction)
        # Note: q >= 0.05 is reported in Statistical Evidence and does NOT force Concordance to "Unclear".

        is_tier_cd = tier.startswith("C") or tier.startswith("D")
        is_near_ref = cluster_direction == "Near cohort reference"
        is_unspecified_lit = lit_dir == "Exploratory / Unspecified"

        if is_near_ref or is_tier_cd or is_unspecified_lit:
            concordance = "Unclear / Insufficient Evidence"
        elif cluster_direction == lit_dir:
            concordance = "Concordant"
        else:
            concordance = "Discordant"

        records.append({
            "cluster_id": cluster_id,
            "feature": feat,
            "system": lit_meta["system"],
            "evidence_tier": tier,
            "qualification": qualification,
            "literature_direction": lit_dir,
            "cluster_zscore": z_val,
            "cluster_direction": cluster_direction,
            "p_value": p_val,
            "q_value_fdr": q_val,
            "stat_significance": stat_significance,
            "effect_size_name": eff_name,
            "effect_size_val": eff_val,
            "concordance": concordance,
            "limitation_qualification": limitation,
        })

    concordance_df = pd.DataFrame(records)
    return concordance_df
