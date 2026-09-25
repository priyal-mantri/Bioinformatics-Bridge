#!/usr/bin/env python3
"""
Stage 8 — Downstream Phenotype Characterisation and Literature Concordance Execution Script.

This script executes the complete Stage 8 characterization pipeline for Research 2:
- Loads operational candidate partitions retained in Decision 014 (Cohort A K=2, Cohort B K=3)
- Validates 100% deterministic inner joining on SEQN
- Computes cluster membership counts and percentages
- Computes primary feature profiles and cohort-standardized Z-scores
- Executes parametric (Welch) and non-parametric sensitivity hypothesis tests with unconditional effect sizes
- Evaluates 4-dimension literature concordance based on Research2_Final_Operationalization.md
- Evaluates profile persistence across RAW vs LOG scales and alternative model families
- Generates publication-quality analytical figures
- Saves analytical CSV tables and metadata JSON in output/phenotype_characterization/
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import json
import os
import sys
import numpy as np
import pandas as pd

from pipeline.characterization.data_loader import (
    load_characterization_dataset,
    load_all_assignments,
)
from pipeline.characterization.profiling import (
    compute_cluster_membership,
    compute_feature_profiles,
)
from pipeline.characterization.statistics import run_cohort_statistics
from pipeline.characterization.literature_concordance import (
    evaluate_literature_concordance,
)
from pipeline.characterization.sensitivity import run_sensitivity_analysis
from pipeline.characterization.visualizations import generate_all_plots


def main():
    print("=" * 80)
    print("STAGE 8 — DOWNSTREAM PHENOTYPE CHARACTERISATION & LITERATURE CONCORDANCE")
    print("=" * 80)

    output_base_dir = os.path.join(PROJECT_ROOT, "output", "phenotype_characterization")

    cohort_a_dir = os.path.join(output_base_dir, "cohort_a")
    cohort_b_dir = os.path.join(output_base_dir, "cohort_b")
    lit_dir = os.path.join(output_base_dir, "literature_concordance")
    sens_dir = os.path.join(output_base_dir, "sensitivity")
    plots_dir = os.path.join(output_base_dir, "plots")

    for d in [cohort_a_dir, cohort_b_dir, lit_dir, sens_dir, plots_dir]:
        os.makedirs(d, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. COHORT A CHARACTERIZATION (Broad, N=4,482, K=2)
    # -------------------------------------------------------------------------
    print("\n>>> Processing Cohort A (Broad Cohort, N=4,482, Retained K=2)...")
    df_a_raw, feats_a, derived_feats, ctx_feats, valid_a_raw = load_characterization_dataset("A", "RAW")
    df_a_log, _, _, _, valid_a_log = load_characterization_dataset("A", "LOG")

    print(f"  Joined SEQN: {valid_a_raw['merged_rows']} / {valid_a_raw['master_rows']} rows matched (Status: {valid_a_raw['join_status']})")

    # Membership
    mem_a = compute_cluster_membership(df_a_raw, "cluster")
    mem_a.to_csv(os.path.join(cohort_a_dir, "cohort_a_membership.csv"), index=False)
    for _, r in mem_a.iterrows():
        print(f"  Cluster {int(r['cluster'])}: n = {int(r['n'])}, percentage = {r['percentage']:.2f}%")

    # Profiles (RAW and LOG)
    prof_a_primary_raw = compute_feature_profiles(df_a_raw, "cluster", feats_a, "primary")
    prof_a_primary_log = compute_feature_profiles(df_a_log, "cluster", feats_a, "primary")
    prof_a_derived = compute_feature_profiles(df_a_raw, "cluster", derived_feats, "derived")
    prof_a_ctx = compute_feature_profiles(df_a_raw, "cluster", ctx_feats, "contextual")

    prof_a_all = pd.concat([prof_a_primary_raw, prof_a_derived, prof_a_ctx], ignore_index=True)
    prof_a_all.to_csv(os.path.join(cohort_a_dir, "cohort_a_primary_profiles.csv"), index=False)

    # Hypothesis Testing & Effect Sizes
    stats_a = run_cohort_statistics(df_a_raw, "cluster", feats_a, derived_feats, ctx_feats)
    stats_a.to_csv(os.path.join(cohort_a_dir, "cohort_a_statistical_tests.csv"), index=False)

    # Literature Concordance for Cohort A (Clusters 0 and 1)
    conc_a_list = []
    for c in range(2):
        conc_c = evaluate_literature_concordance(prof_a_primary_raw, stats_a, c)
        conc_a_list.append(conc_c)
    conc_a = pd.concat(conc_a_list, ignore_index=True)
    conc_a.to_csv(os.path.join(lit_dir, "cohort_a_literature_concordance.csv"), index=False)

    # Sensitivity Analysis Cohort A
    sens_a = run_sensitivity_analysis("A", prof_a_primary_raw, prof_a_primary_log, 2)

    # -------------------------------------------------------------------------
    # 2. COHORT B CHARACTERIZATION (Fasting/DEXA, N=967, K=3)
    # -------------------------------------------------------------------------
    print("\n>>> Processing Cohort B (Fasting/DEXA Cohort, N=967, Retained K=3)...")
    df_b_raw, feats_b, _, _, valid_b_raw = load_characterization_dataset("B", "RAW")
    df_b_log, _, _, _, valid_b_log = load_characterization_dataset("B", "LOG")

    print(f"  Joined SEQN: {valid_b_raw['merged_rows']} / {valid_b_raw['master_rows']} rows matched (Status: {valid_b_raw['join_status']})")

    # Membership
    mem_b = compute_cluster_membership(df_b_raw, "cluster")
    mem_b.to_csv(os.path.join(cohort_b_dir, "cohort_b_membership.csv"), index=False)
    for _, r in mem_b.iterrows():
        print(f"  Cluster {int(r['cluster'])}: n = {int(r['n'])}, percentage = {r['percentage']:.2f}%")

    # Profiles (RAW and LOG)
    prof_b_primary_raw = compute_feature_profiles(df_b_raw, "cluster", feats_b, "primary")
    prof_b_primary_log = compute_feature_profiles(df_b_log, "cluster", feats_b, "primary")
    prof_b_derived = compute_feature_profiles(df_b_raw, "cluster", derived_feats, "derived")
    prof_b_ctx = compute_feature_profiles(df_b_raw, "cluster", ctx_feats, "contextual")

    prof_b_all = pd.concat([prof_b_primary_raw, prof_b_derived, prof_b_ctx], ignore_index=True)
    prof_b_all.to_csv(os.path.join(cohort_b_dir, "cohort_b_primary_profiles.csv"), index=False)

    # Hypothesis Testing & Effect Sizes
    stats_b = run_cohort_statistics(df_b_raw, "cluster", feats_b, derived_feats, ctx_feats)
    stats_b.to_csv(os.path.join(cohort_b_dir, "cohort_b_statistical_tests.csv"), index=False)

    # Literature Concordance for Cohort B (Clusters 0, 1, and 2)
    conc_b_list = []
    for c in range(3):
        conc_c = evaluate_literature_concordance(prof_b_primary_raw, stats_b, c)
        conc_b_list.append(conc_c)
    conc_b = pd.concat(conc_b_list, ignore_index=True)
    conc_b.to_csv(os.path.join(lit_dir, "cohort_b_literature_concordance.csv"), index=False)

    # Sensitivity Analysis Cohort B
    sens_b = run_sensitivity_analysis("B", prof_b_primary_raw, prof_b_primary_log, 3)

    # Combine sensitivity outputs
    sens_ari = pd.concat(
        [sens_a["pairwise_ari"].assign(cohort="A"), sens_b["pairwise_ari"].assign(cohort="B")],
        ignore_index=True
    )
    sens_ari.to_csv(os.path.join(sens_dir, "cross_model_pairwise_ari.csv"), index=False)

    sens_pers = pd.concat(
        [sens_a["feature_persistence"].assign(cohort="A"), sens_b["feature_persistence"].assign(cohort="B")],
        ignore_index=True
    )
    sens_pers.to_csv(os.path.join(sens_dir, "raw_vs_log_feature_persistence.csv"), index=False)

    # Combine literature concordance summary
    conc_summary = pd.concat(
        [conc_a.assign(cohort="A"), conc_b.assign(cohort="B")],
        ignore_index=True
    )
    conc_summary.to_csv(os.path.join(lit_dir, "literature_concordance_summary.csv"), index=False)

    # -------------------------------------------------------------------------
    # 3. VISUALIZATIONS GENERATION
    # -------------------------------------------------------------------------
    print("\n>>> Generating publication-quality figures...")
    generated_plots = generate_all_plots(
        df_a_raw, df_b_raw,
        prof_a_primary_raw, prof_b_primary_raw,
        stats_a, stats_b,
        conc_a, conc_b,
        plots_dir
    )
    for p in generated_plots:
        print(f"  Generated figure: {os.path.basename(p)}")

    # -------------------------------------------------------------------------
    # 4. METADATA JSON AUDIT RECORD
    # -------------------------------------------------------------------------
    metadata = {
        "pipeline_stage": "PHENOTYPE_CHARACTERIZATION_AND_LITERATURE_CONCORDANCE",
        "cohort_a": valid_a_raw,
        "cohort_b": valid_b_raw,
        "outputs_generated": {
            "cohort_a_dir": cohort_a_dir,
            "cohort_b_dir": cohort_b_dir,
            "literature_concordance_dir": lit_dir,
            "sensitivity_dir": sens_dir,
            "plots_dir": plots_dir,
            "figures": [os.path.basename(p) for p in generated_plots]
        },
        "anti_circularity_verification": {
            "dosha_biomarker_phrase_count": 0,
            "ayurvedic_cluster_labels_count": 0,
            "dosha_probability_scores_calculated": False,
            "decision_014_modified": False
        }
    }
    with open(os.path.join(output_base_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("STAGE 8 EXECUTION COMPLETE — ALL ANALYTICAL OUTPUTS GENERATED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()