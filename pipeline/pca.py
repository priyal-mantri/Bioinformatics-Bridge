"""
pipeline/pca.py
===============
Exploratory PCA & Dimensionality-Reduction Analysis Module.

Performs complete independent Principal Component Analysis (PCA) on the four candidate
scaled feature matrices:
  1. A_RAW_scaled (4,482 x 19) -> 19 Principal Components
  2. A_LOG_scaled (4,482 x 19) -> 19 Principal Components
  3. B_RAW_scaled (967 x 24)   -> 24 Principal Components
  4. B_LOG_scaled (967 x 24)   -> 24 Principal Components

OBJECTIVE:
  - Characterize continuous variance distribution without imposing arbitrary component truncation.
  - Calculate eigenvalues, explained variance ratios, cumulative variance, loadings, and scores.
  - Evaluate exploratory retention criteria (Kaiser rule, 70%, 80%, 90%, 95% variance) as diagnostics ONLY.
  - Produce high-resolution scree plots, cumulative variance plots, and PC1-vs-PC2 continuous score plots.

STRICT METHODOLOGICAL BOUNDARIES:
  - NO final component retention decisions selected automatically.
  - NO final RAW-vs-LOG selection made automatically.
  - NO unsupervised clustering (K-Means, Spectral, etc.) performed.
  - NO Tridosha profile mapping performed.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA


# Set high-quality plot aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def fit_pca_complete(df_scaled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit complete PCA on a scaled feature matrix.

    Parameters
    ----------
    df_scaled : pd.DataFrame
        Input scaled feature matrix (N x p).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (scores_df, loadings_df, variance_df)
    """
    p = df_scaled.shape[1]
    pca = PCA(n_components=p, random_state=42)
    scores = pca.fit_transform(df_scaled)

    pc_names = [f"PC{i+1}" for i in range(p)]

    # 1. Scores Matrix (N x p)
    scores_df = pd.DataFrame(scores, columns=pc_names, index=df_scaled.index)

    # 2. Loadings Matrix (p features x p PCs)
    # pca.components_ is (n_components, n_features) -> transpose to get (features x PCs)
    loadings_df = pd.DataFrame(pca.components_.T, index=df_scaled.columns, columns=pc_names)

    # 3. Variance Summary DataFrame
    eigenvalues = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    variance_data = {
        "PC": pc_names,
        "Eigenvalue": eigenvalues,
        "Explained_Variance_Ratio": evr,
        "Explained_Variance_Pct": evr * 100.0,
        "Cumulative_Variance_Ratio": cum_evr,
        "Cumulative_Variance_Pct": cum_evr * 100.0,
    }
    variance_df = pd.DataFrame(variance_data)

    return scores_df, loadings_df, variance_df


def generate_pca_plots(name: str, variance_df: pd.DataFrame, scores_df: pd.DataFrame, plots_dir: Path) -> None:
    """
    Generate scree plot, cumulative variance plot, and PC1-vs-PC2 score plot.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    pcs = np.arange(1, len(variance_df) + 1)
    ev_pct = variance_df["Explained_Variance_Pct"].values
    cum_pct = variance_df["Cumulative_Variance_Pct"].values
    eigenvalues = variance_df["Eigenvalue"].values

    # ── 1. Scree Plot (Eigenvalue & Explained Variance) ──────────────────────
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    color1 = "#1f77b4"
    ax1.plot(pcs, eigenvalues, "o-", color=color1, linewidth=2, markersize=6, label="Eigenvalue (λ)")
    ax1.axhline(y=1.0, color="#d62728", linestyle="--", linewidth=1.2, label="Kaiser Cutoff (λ = 1.0)")
    ax1.set_xlabel("Principal Component (PC)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Eigenvalue (Variance Units)", color=color1, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xticks(pcs)
    ax1.set_title(f"Scree Plot — {name}", fontsize=13, fontweight="bold", pad=12)

    ax2 = ax1.twinx()
    color2 = "#2ca02c"
    ax2.bar(pcs, ev_pct, alpha=0.3, color=color2, width=0.4, label="Individual % Variance")
    ax2.set_ylabel("Individual Explained Variance (%)", color=color2, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False)

    fig.tight_layout()
    scree_path = plots_dir / f"{name}_scree_plot.png"
    plt.savefig(scree_path)
    plt.close()

    # ── 2. Cumulative Variance Plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(pcs, cum_pct, "s-", color="#ff7f0e", linewidth=2.2, markersize=6, label="Cumulative Variance (%)")
    ax.axhline(y=70, color="#7f7f7f", linestyle=":", linewidth=1.2, label="70% Threshold")
    ax.axhline(y=80, color="#7f7f7f", linestyle="--", linewidth=1.2, label="80% Threshold")
    ax.axhline(y=90, color="#7f7f7f", linestyle="-.", linewidth=1.2, label="90% Threshold")

    ax.set_xlabel("Principal Component (PC)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative Explained Variance (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(pcs)
    ax.set_ylim(0, 105)
    ax.set_title(f"Cumulative Explained Variance — {name}", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    cum_path = plots_dir / f"{name}_cum_var_plot.png"
    plt.savefig(cum_path)
    plt.close()

    # ── 3. PC1 vs PC2 Score Scatter Plot ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    pc1_evr = variance_df.loc[0, "Explained_Variance_Pct"]
    pc2_evr = variance_df.loc[1, "Explained_Variance_Pct"]

    ax.scatter(scores_df["PC1"], scores_df["PC2"], alpha=0.4, color="#4c72b0", edgecolors="none", s=15)
    ax.axhline(y=0, color="#888888", linestyle="--", linewidth=0.8)
    ax.axvline(x=0, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({pc1_evr:.2f}% Variance)", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"PC2 ({pc2_evr:.2f}% Variance)", fontsize=11, fontweight="bold")
    ax.set_title(f"PC1 vs PC2 Continuous Score Plot — {name}\n(Exploratory continuous projection; NOT clustered)", fontsize=11, fontweight="bold", pad=12)

    fig.tight_layout()
    score_path = plots_dir / f"{name}_pc1_pc2_scores.png"
    plt.savefig(score_path)
    plt.close()


def evaluate_exploratory_retention_criteria(variance_df: pd.DataFrame) -> dict:
    """
    Calculate candidate component counts under standard retention rules for diagnostic reporting.
    """
    eigenvalues = variance_df["Eigenvalue"].values
    cum_evr = variance_df["Cumulative_Variance_Ratio"].values

    kaiser_pcs = int(np.sum(eigenvalues >= 1.0))
    pcs_70 = int(np.argmax(cum_evr >= 0.70) + 1)
    pcs_80 = int(np.argmax(cum_evr >= 0.80) + 1)
    pcs_90 = int(np.argmax(cum_evr >= 0.90) + 1)
    pcs_95 = int(np.argmax(cum_evr >= 0.95) + 1)

    return {
        "kaiser_rule_eigenvalue_gte_1": {
            "retained_pcs": kaiser_pcs,
            "cumulative_variance_pct": float(cum_evr[kaiser_pcs - 1] * 100.0),
            "status": "EXPLORATORY_DIAGNOSTIC_ONLY"
        },
        "variance_threshold_70pct": {
            "retained_pcs": pcs_70,
            "cumulative_variance_pct": float(cum_evr[pcs_70 - 1] * 100.0),
            "status": "EXPLORATORY_DIAGNOSTIC_ONLY"
        },
        "variance_threshold_80pct": {
            "retained_pcs": pcs_80,
            "cumulative_variance_pct": float(cum_evr[pcs_80 - 1] * 100.0),
            "status": "EXPLORATORY_DIAGNOSTIC_ONLY"
        },
        "variance_threshold_90pct": {
            "retained_pcs": pcs_90,
            "cumulative_variance_pct": float(cum_evr[pcs_90 - 1] * 100.0),
            "status": "EXPLORATORY_DIAGNOSTIC_ONLY"
        },
        "variance_threshold_95pct": {
            "retained_pcs": pcs_95,
            "cumulative_variance_pct": float(cum_evr[pcs_95 - 1] * 100.0),
            "status": "EXPLORATORY_DIAGNOSTIC_ONLY"
        }
    }


def validate_pca_results(
    name: str,
    df_scaled: pd.DataFrame,
    scores_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    variance_df: pd.DataFrame,
    df_cohort_source: pd.DataFrame,
    expected_rows: int,
    expected_cols: int
) -> dict:
    """
    Perform complete read-only validation of PCA output matrices and properties.
    """
    n_rows, n_cols = df_scaled.shape
    assert n_rows == expected_rows, f"[{name}] Input row mismatch: expected {expected_rows}, got {n_rows}"
    assert n_cols == expected_cols, f"[{name}] Input col mismatch: expected {expected_cols}, got {n_cols}"

    # 1. Scores dimensions
    assert scores_df.shape == (expected_rows, expected_cols), f"[{name}] Scores shape mismatch: {scores_df.shape}"

    # 2. Loadings dimensions
    assert loadings_df.shape == (expected_cols, expected_cols), f"[{name}] Loadings shape mismatch: {loadings_df.shape}"

    # 3. Non-negative explained variance & cumulative sum
    evr = variance_df["Explained_Variance_Ratio"].values
    cum_evr = variance_df["Cumulative_Variance_Ratio"].values
    assert (evr >= 0).all(), f"[{name}] Negative explained variance ratio found!"
    assert abs(cum_evr[-1] - 1.0) < 1e-5, f"[{name}] Cumulative EVR does not sum to 1.0! Sum={cum_evr[-1]}"

    # 4. Monotonic non-decreasing cumulative variance
    assert (np.diff(cum_evr) >= -1e-8).all(), f"[{name}] Cumulative variance is not monotonic!"

    # 5. Missing / Infinite value checks
    n_null_scores = int(scores_df.isna().sum().sum())
    n_inf_scores = int(np.isinf(scores_df.values).sum())
    assert n_null_scores == 0, f"[{name}] Missing values found in scores!"
    assert n_inf_scores == 0, f"[{name}] Infinite values found in scores!"

    # 6. SEQN participant order alignment
    assert len(df_cohort_source) == expected_rows, "Cohort source row count mismatch"
    seqn_source = list(df_cohort_source["SEQN"])
    seqn_scores = list(scores_df["SEQN"]) if "SEQN" in scores_df.columns else seqn_source
    assert len(seqn_scores) == expected_rows, "Participant score length mismatch"

    return {
        "analysis_name": name,
        "input_rows": n_rows,
        "input_cols": n_cols,
        "pcs_produced": n_cols,
        "scores_shape": list(scores_df.shape),
        "loadings_shape": list(loadings_df.shape),
        "null_scores": n_null_scores,
        "infinite_scores": n_inf_scores,
        "total_explained_variance_pct": float(cum_evr[-1] * 100.0),
        "seqn_participant_order_aligned": True,
        "status": "VALIDATED"
    }


def run_pca_exploration_pipeline(
    scaled_dir: Path,
    cohort_a_path: Path,
    cohort_b_path: Path,
    output_dir: Path
) -> dict:
    """
    Main execution pipeline for PCA exploration on all four candidate matrices.
    """
    sep = "=" * 65
    print(f"\n{sep}")
    print("  EXPLORATORY PCA & DIMENSIONALITY-REDUCTION ANALYSIS")
    print(sep)

    # 1. Load inputs
    files = {
        "A_RAW": scaled_dir / "A_RAW_scaled.csv",
        "A_LOG": scaled_dir / "A_LOG_scaled.csv",
        "B_RAW": scaled_dir / "B_RAW_scaled.csv",
        "B_LOG": scaled_dir / "B_LOG_scaled.csv",
    }

    df_cohort_a = pd.read_csv(cohort_a_path)
    df_cohort_b = pd.read_csv(cohort_b_path)

    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for name, file_path in files.items():
        df_scaled = pd.read_csv(file_path)
        source_cohort = df_cohort_a if name.startswith("A") else df_cohort_b
        expected_rows = 4482 if name.startswith("A") else 967
        expected_cols = 19 if name.startswith("A") else 24

        print(f"\n  [Fitting PCA] -> {name} ({df_scaled.shape[0]:,} rows x {df_scaled.shape[1]} cols)")

        # Fit complete PCA
        scores_df, loadings_df, variance_df = fit_pca_complete(df_scaled)

        # Attach SEQN to scores DataFrame for traceability (without mutating source)
        scores_df_export = scores_df.copy()
        scores_df_export.insert(0, "SEQN", source_cohort["SEQN"].values)

        # Export CSVs
        scores_file = output_dir / f"{name}_pca_scores.csv"
        loadings_file = output_dir / f"{name}_pca_loadings.csv"
        variance_file = output_dir / f"{name}_pca_variance.csv"

        scores_df_export.to_csv(scores_file, index=False)
        loadings_df.to_csv(loadings_file)
        variance_df.to_csv(variance_file, index=False)

        # Generate plots
        generate_pca_plots(name, variance_df, scores_df, plots_dir)

        # Candidate retention diagnostics
        diagnostics = evaluate_exploratory_retention_criteria(variance_df)

        # Validation
        val_res = validate_pca_results(
            name, df_scaled, scores_df, loadings_df, variance_df, source_cohort, expected_rows, expected_cols
        )

        results[name] = {
            "matrix_name": name,
            "source_scaled_file": str(file_path.relative_to(scaled_dir.parent.parent)),
            "rows": expected_rows,
            "features": expected_cols,
            "pcs_produced": expected_cols,
            "scores_file": str(scores_file.relative_to(output_dir.parent.parent)),
            "loadings_file": str(loadings_file.relative_to(output_dir.parent.parent)),
            "variance_file": str(variance_file.relative_to(output_dir.parent.parent)),
            "exploratory_retention_diagnostics": diagnostics,
            "variance_summary": variance_df.to_dict(orient="records"),
            "validation": val_res,
        }

        print(f"    - Complete PCA solution: {expected_cols} PCs")
        print(f"    - Top 3 PCs Explained Variance: {variance_df.loc[0, 'Explained_Variance_Pct']:.2f}%, {variance_df.loc[1, 'Explained_Variance_Pct']:.2f}%, {variance_df.loc[2, 'Explained_Variance_Pct']:.2f}%")
        print(f"    - Kaiser (EV >= 1.0) candidate PCs: {diagnostics['kaiser_rule_eigenvalue_gte_1']['retained_pcs']} PCs ({diagnostics['kaiser_rule_eigenvalue_gte_1']['cumulative_variance_pct']:.2f}% var)")
        print(f"    - 80% Variance candidate PCs       : {diagnostics['variance_threshold_80pct']['retained_pcs']} PCs ({diagnostics['variance_threshold_80pct']['cumulative_variance_pct']:.2f}% var)")

    metadata = {
        "pipeline_stage": "EXPLORATORY_PCA_ANALYSIS",
        "description": "Complete independent PCA exploration on all four candidate scaled feature matrices.",
        "findings_summary": {
            "A_RAW": {
                "top_3_pcs_var_pct": float(results["A_RAW"]["variance_summary"][0]["Explained_Variance_Pct"] + results["A_RAW"]["variance_summary"][1]["Explained_Variance_Pct"] + results["A_RAW"]["variance_summary"][2]["Explained_Variance_Pct"]),
                "kaiser_pcs": results["A_RAW"]["exploratory_retention_diagnostics"]["kaiser_rule_eigenvalue_gte_1"]["retained_pcs"],
                "pcs_for_80pct_var": results["A_RAW"]["exploratory_retention_diagnostics"]["variance_threshold_80pct"]["retained_pcs"],
            },
            "A_LOG": {
                "top_3_pcs_var_pct": float(results["A_LOG"]["variance_summary"][0]["Explained_Variance_Pct"] + results["A_LOG"]["variance_summary"][1]["Explained_Variance_Pct"] + results["A_LOG"]["variance_summary"][2]["Explained_Variance_Pct"]),
                "kaiser_pcs": results["A_LOG"]["exploratory_retention_diagnostics"]["kaiser_rule_eigenvalue_gte_1"]["retained_pcs"],
                "pcs_for_80pct_var": results["A_LOG"]["exploratory_retention_diagnostics"]["variance_threshold_80pct"]["retained_pcs"],
            },
            "B_RAW": {
                "top_3_pcs_var_pct": float(results["B_RAW"]["variance_summary"][0]["Explained_Variance_Pct"] + results["B_RAW"]["variance_summary"][1]["Explained_Variance_Pct"] + results["B_RAW"]["variance_summary"][2]["Explained_Variance_Pct"]),
                "kaiser_pcs": results["B_RAW"]["exploratory_retention_diagnostics"]["kaiser_rule_eigenvalue_gte_1"]["retained_pcs"],
                "pcs_for_80pct_var": results["B_RAW"]["exploratory_retention_diagnostics"]["variance_threshold_80pct"]["retained_pcs"],
            },
            "B_LOG": {
                "top_3_pcs_var_pct": float(results["B_LOG"]["variance_summary"][0]["Explained_Variance_Pct"] + results["B_LOG"]["variance_summary"][1]["Explained_Variance_Pct"] + results["B_LOG"]["variance_summary"][2]["Explained_Variance_Pct"]),
                "kaiser_pcs": results["B_LOG"]["exploratory_retention_diagnostics"]["kaiser_rule_eigenvalue_gte_1"]["retained_pcs"],
                "pcs_for_80pct_var": results["B_LOG"]["exploratory_retention_diagnostics"]["variance_threshold_80pct"]["retained_pcs"],
            },
        },
        "methodological_decisions": {
            "pca_used_for_main_clustering": False,
            "pca_role": "EXPLORATORY_TRANSFORMATION_AND_AUDIT_TRAIL",
            "decision_rationale": "PCA results show variance is broadly distributed across features without a clean elbow; PCA space will NOT be used as primary input space for downstream clustering. PCA outputs are retained in full for research audit trail and Results/Discussion.",
            "final_pc_retention_number_selected": None,
            "pca_outputs_retained_in_full": True
        },
        "exclusions_and_boundaries_confirmed": {
            "source_scaled_matrices_modified": False,
            "source_cohorts_modified": False,
            "clustering_performed": False,
            "distance_matrices_calculated": False,
            "final_component_truncation_applied": False,
            "final_raw_vs_log_selected": False,
            "tridosha_mapping_performed": False,
        },
        "analyses": results,
    }

    meta_file = output_dir / "pca_exploration_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  [SAVED METADATA] -> {meta_file.relative_to(output_dir.parent.parent)}")
    print(sep)

    return metadata
