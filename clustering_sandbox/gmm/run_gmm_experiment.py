"""
clustering_sandbox/gmm/run_gmm_experiment.py
=============================================
Exploratory Gaussian Mixture Model (GMM) Sandbox Experiment (Experiment 3).

Evaluates GMM clustering (full covariance, n_init=25, seed=42) across K = 2..7
for four candidate standardized feature matrices:
  1. A_RAW_scaled (4,482 x 19)
  2. A_LOG_scaled (4,482 x 19)
  3. B_RAW_scaled (967 x 24)
  4. B_LOG_scaled (967 x 24)

STRICT METHODOLOGICAL BOUNDARIES:
  - Quantitative exploratory evidence only.
  - NO optimal K selected.
  - NO "winner" declared.
  - NO subjective evaluation labels ('good', 'weak', 'best') assigned.
  - NO Tridosha / Prakriti mapping performed.
  - Full standardized feature space used as GMM input (PCA used strictly for 2D visualization plots).
  - Main research decision log NOT modified.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


RANDOM_SEED = 42
N_INIT = 25
COVARIANCE_TYPE = "full"
K_RANGE = [2, 3, 4, 5, 6, 7]


def compute_entropy(proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes Shannon entropy H_i and normalized entropy H_norm_i for each row of probabilities.
    H_i = - sum_k p_ik log(p_ik)
    H_norm_i = H_i / log(K)
    """
    n_samples, k = proba.shape
    # Avoid log(0)
    eps = 1e-15
    proba_safe = np.clip(proba, eps, 1.0)
    # Normalize inside safe array to ensure sum to 1
    proba_safe = proba_safe / proba_safe.sum(axis=1, keepdims=True)
    
    entropy_i = -np.sum(proba_safe * np.log(proba_safe), axis=1)
    max_entropy = np.log(k)
    norm_entropy_i = entropy_i / max_entropy if max_entropy > 0 else np.zeros_like(entropy_i)
    
    return entropy_i, norm_entropy_i


def run_gmm_sweep_for_matrix(
    name: str,
    df_scaled: pd.DataFrame,
    df_cohort_source: pd.DataFrame,
    gmm_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict]:
    """
    Run GMM sweep (K=2..7) on a single scaled matrix.
    """
    metrics_dir = gmm_dir / "metrics"
    assignments_dir = gmm_dir / "assignments"
    probabilities_dir = gmm_dir / "probabilities"
    parameters_dir = gmm_dir / "model_parameters"

    n_samples, n_features = df_scaled.shape
    X = df_scaled.values
    seqn_series = df_cohort_source["SEQN"].values

    metrics_records = []
    assignments_dict = {"SEQN": seqn_series}
    param_records = []

    for k in K_RANGE:
        t0 = time.time()
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=COVARIANCE_TYPE,
            n_init=N_INIT,
            random_state=RANDOM_SEED
        )
        
        warn_msg = None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gmm.fit(X)
            if len(w) > 0:
                warn_msg = "; ".join([str(item.message) for item in w])

        converged = bool(gmm.converged_)
        n_iter = int(gmm.n_iter_)
        avg_log_likelihood = float(gmm.score(X))
        total_log_likelihood = float(avg_log_likelihood * n_samples)
        bic = float(gmm.bic(X))
        aic = float(gmm.aic(X))

        # Assignments & Soft Probabilities
        labels = gmm.predict(X)
        proba = gmm.predict_proba(X)  # shape (N, K)
        
        # Max posterior probability per participant
        max_proba = np.max(proba, axis=1)
        mean_max_pos = float(np.mean(max_proba))
        median_max_pos = float(np.median(max_proba))
        
        prop_ge_50 = float(np.mean(max_proba >= 0.50))
        prop_ge_60 = float(np.mean(max_proba >= 0.60))
        prop_ge_70 = float(np.mean(max_proba >= 0.70))
        prop_ge_80 = float(np.mean(max_proba >= 0.80))
        prop_ge_90 = float(np.mean(max_proba >= 0.90))

        # Entropy
        entropy_raw, entropy_norm = compute_entropy(proba)
        mean_ent = float(np.mean(entropy_raw))
        median_ent = float(np.median(entropy_raw))
        std_ent = float(np.std(entropy_raw))
        ent_p5 = float(np.percentile(entropy_raw, 5))
        ent_p25 = float(np.percentile(entropy_raw, 25))
        ent_p75 = float(np.percentile(entropy_raw, 75))
        ent_p95 = float(np.percentile(entropy_raw, 95))

        mean_norm_ent = float(np.mean(entropy_norm))
        median_norm_ent = float(np.median(entropy_norm))

        # Internal Hard-Assignment Metrics
        sil = float(silhouette_score(X, labels, random_state=RANDOM_SEED))
        ch = float(calinski_harabasz_score(X, labels))
        db = float(davies_bouldin_score(X, labels))

        # Cluster Counts & Proportions
        unique_lbls, counts = np.unique(labels, return_counts=True)
        counts_dict = {int(lbl): int(cnt) for lbl, cnt in zip(unique_lbls, counts)}
        # Ensure all k clusters are represented in dictionary
        for c in range(k):
            if c not in counts_dict:
                counts_dict[c] = 0
        
        props = [counts_dict[c] / n_samples for c in range(k)]
        min_prop = float(min(props))
        max_prop = float(max(props))

        # Save Probabilities CSV per K
        prob_cols = {f"P_cluster_{c+1}": proba[:, c] for c in range(k)}
        prob_df = pd.DataFrame({"SEQN": seqn_series, **prob_cols, "max_posterior": max_proba, "entropy_raw": entropy_raw, "entropy_norm": entropy_norm})
        prob_csv = probabilities_dir / f"{name}_gmm_probabilities_K{k}.csv"
        prob_df.to_csv(prob_csv, index=False)

        # Save Component Parameters JSON
        param_data = {
            "matrix_name": name,
            "K": k,
            "covariance_type": COVARIANCE_TYPE,
            "weights": gmm.weights_.tolist(),
            "means": gmm.means_.tolist(),
            "covariances": gmm.covariances_.tolist(),
            "converged": converged,
            "n_iter": n_iter,
            "lower_bound": float(gmm.lower_bound_)
        }
        param_json = parameters_dir / f"{name}_gmm_parameters_K{k}.json"
        with open(param_json, "w") as f:
            json.dump(param_data, f, indent=2)

        metrics_records.append({
            "matrix_name": name,
            "N": n_samples,
            "feature_count": n_features,
            "K": k,
            "covariance_type": COVARIANCE_TYPE,
            "BIC": bic,
            "AIC": aic,
            "log_likelihood": total_log_likelihood,
            "average_log_likelihood": avg_log_likelihood,
            "silhouette": sil,
            "CH": ch,
            "DB": db,
            "min_cluster_proportion": min_prop,
            "max_cluster_proportion": max_prop,
            "cluster_counts": json.dumps(counts_dict),
            "cluster_proportions": json.dumps([round(p, 4) for p in props]),
            "mean_max_posterior": mean_max_pos,
            "median_max_posterior": median_max_pos,
            "proportion_max_posterior_ge_0.50": prop_ge_50,
            "proportion_max_posterior_ge_0.60": prop_ge_60,
            "proportion_max_posterior_ge_0.70": prop_ge_70,
            "proportion_max_posterior_ge_0.80": prop_ge_80,
            "proportion_max_posterior_ge_0.90": prop_ge_90,
            "mean_entropy": mean_ent,
            "median_entropy": median_ent,
            "std_entropy": std_ent,
            "entropy_p5": ent_p5,
            "entropy_p25": ent_p25,
            "entropy_p75": ent_p75,
            "entropy_p95": ent_p95,
            "mean_norm_entropy": mean_norm_ent,
            "median_norm_entropy": median_norm_ent,
            "convergence_status": "CONVERGED" if converged else "FAILED",
            "n_iter": n_iter,
            "warnings": warn_msg if warn_msg else "None",
            "random_state": RANDOM_SEED,
            "n_init": N_INIT,
            "sklearn_version": sklearn.__version__,
            "runtime_seconds": float(time.time() - t0)
        })

        assignments_dict[f"K{k}_cluster"] = labels
        assignments_dict[f"K{k}_max_posterior"] = max_proba
        assignments_dict[f"K{k}_norm_entropy"] = entropy_norm
        param_records.append(param_data)

    metrics_df = pd.DataFrame(metrics_records)
    assignments_df = pd.DataFrame(assignments_dict)

    summary_dict = {
        "matrix_name": name,
        "n_samples": n_samples,
        "n_features": n_features,
        "runs": metrics_records
    }

    return metrics_df, assignments_df, param_records, summary_dict


def generate_gmm_plots(
    name: str,
    metrics_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    df_scaled: pd.DataFrame,
    plots_dir: Path
) -> None:
    """
    Generates all required plots for a single matrix:
      A. BIC vs K
      B. AIC vs K
      C. Average Log-Likelihood vs K
      D. Silhouette Score vs K (plus CH, DB)
      E. Cluster Size Distribution vs K
      F. Assignment Confidence Summary vs K
      G. Entropy Summary vs K
      H. 2D PCA Projections (K=2,3,4) showing posterior assignment structure
    """
    ks = metrics_df["K"].values

    # 1. BIC, AIC, Average Log-Likelihood Panel
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    axes[0].plot(ks, metrics_df["BIC"], "o-", color="#1f77b4", linewidth=2, markersize=7)
    axes[0].set_title(f"{name} — BIC vs K", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Components (K)")
    axes[0].set_ylabel("BIC (Lower is better fit)")
    axes[0].set_xticks(ks)

    axes[1].plot(ks, metrics_df["AIC"], "s-", color="#ff7f0e", linewidth=2, markersize=7)
    axes[1].set_title(f"{name} — AIC vs K", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Components (K)")
    axes[1].set_ylabel("AIC (Lower is better fit)")
    axes[1].set_xticks(ks)

    axes[2].plot(ks, metrics_df["average_log_likelihood"], "^-", color="#2ca02c", linewidth=2, markersize=7)
    axes[2].set_title(f"{name} — Avg Log-Likelihood vs K", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Number of Components (K)")
    axes[2].set_ylabel("Avg Log-Likelihood per sample")
    axes[2].set_xticks(ks)

    plt.tight_layout()
    plt.savefig(plots_dir / f"{name}_bic_aic_ll.png", dpi=300)
    plt.close()

    # 2. Internal Metrics Panel (Silhouette, CH, DB)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(ks, metrics_df["silhouette"], "o-", color="#9467bd", linewidth=2, markersize=7)
    axes[0].set_title(f"{name} — Silhouette Score vs K", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Number of Components (K)")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_xticks(ks)

    axes[1].plot(ks, metrics_df["CH"], "s-", color="#8c564b", linewidth=2, markersize=7)
    axes[1].set_title(f"{name} — Calinski-Harabasz Index vs K", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Components (K)")
    axes[1].set_ylabel("Calinski-Harabasz Score")
    axes[1].set_xticks(ks)

    axes[2].plot(ks, metrics_df["DB"], "d-", color="#e377c2", linewidth=2, markersize=7)
    axes[2].set_title(f"{name} — Davies-Bouldin Index vs K", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Number of Components (K)")
    axes[2].set_ylabel("Davies-Bouldin Index (Lower is better)")
    axes[2].set_xticks(ks)

    plt.tight_layout()
    plt.savefig(plots_dir / f"{name}_internal_metrics.png", dpi=300)
    plt.close()

    # 3. Cluster Size Proportions vs K
    fig, ax = plt.subplots(figsize=(9, 5))
    bar_width = 0.12
    colors = sns.color_palette("tab10", 7)

    for i, row in metrics_df.iterrows():
        k = int(row["K"])
        props = json.loads(row["cluster_proportions"])
        x_offsets = np.linspace(-bar_width*(k-1)/2, bar_width*(k-1)/2, k) + k
        for c, (prop, x_pos) in enumerate(zip(props, x_offsets)):
            ax.bar(x_pos, prop, width=bar_width*0.9, color=colors[c % len(colors)], edgecolor="black", linewidth=0.5)

    ax.set_xticks(ks)
    ax.set_title(f"{name} — Component Proportions vs K", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Components (K)")
    ax.set_ylabel("Proportion of Sample")
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(plots_dir / f"{name}_cluster_sizes.png", dpi=300)
    plt.close()

    # 4. Assignment Confidence Summary vs K
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, metrics_df["mean_max_posterior"], "o-", label="Mean Max Posterior", linewidth=2.5, color="#1f77b4")
    ax.plot(ks, metrics_df["median_max_posterior"], "s--", label="Median Max Posterior", linewidth=2, color="#aec7e8")
    ax.plot(ks, metrics_df["proportion_max_posterior_ge_0.50"], "^:", label="Prop P(max) >= 0.50", linewidth=1.5, color="#2ca02c")
    ax.plot(ks, metrics_df["proportion_max_posterior_ge_0.70"], "v:", label="Prop P(max) >= 0.70", linewidth=1.5, color="#ff7f0e")
    ax.plot(ks, metrics_df["proportion_max_posterior_ge_0.90"], "d:", label="Prop P(max) >= 0.90", linewidth=1.5, color="#d62728")

    ax.set_xticks(ks)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{name} — Assignment Confidence & Posterior Thresholds vs K", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Components (K)")
    ax.set_ylabel("Posterior Probability / Proportion")
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(plots_dir / f"{name}_assignment_confidence.png", dpi=300)
    plt.close()

    # 5. Entropy Summary vs K
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, metrics_df["mean_norm_entropy"], "o-", label="Mean Normalized Entropy H/log(K)", linewidth=2.5, color="#7f7f7f")
    ax.plot(ks, metrics_df["median_norm_entropy"], "s--", label="Median Normalized Entropy", linewidth=2, color="#bcbd22")
    ax.plot(ks, metrics_df["mean_entropy"], "^:", label="Mean Raw Entropy (nats)", linewidth=1.5, color="#17becf")

    ax.set_xticks(ks)
    ax.set_title(f"{name} — Posterior Assignment Entropy vs K", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Components (K)")
    ax.set_ylabel("Entropy")
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(plots_dir / f"{name}_entropy_summary.png", dpi=300)
    plt.close()

    # 6. 2D PCA Projections (Visualization Only) for K=2, 3, 4
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(df_scaled.values)
    var_exp = pca.explained_variance_ratio_

    for k in [2, 3, 4]:
        labels = assignments_df[f"K{k}_cluster"].values
        max_pos = assignments_df[f"K{k}_max_posterior"].values

        fig, ax = plt.subplots(figsize=(8, 6.5))
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=labels,
            cmap="tab10",
            alpha=np.clip(max_pos ** 2, 0.2, 0.95),  # Opacity proportional to posterior certainty squared
            s=18,
            edgecolors="none"
        )
        ax.set_title(
            f"{name} — GMM K={k} Soft Component Assignments in 2D PCA Space\n"
            f"[VISUALIZATION ONLY — GMM fitted on full {df_scaled.shape[1]}D space]\n"
            f"PCA Explained Variance: PC1={var_exp[0]:.1%}, PC2={var_exp[1]:.1%}",
            fontsize=10, fontweight="bold"
        )
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1%})")
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1%})")
        
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(k))
        cbar.set_label("Assigned GMM Component Index", rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"{name}_pca_posterior_K{k}.png", dpi=300)
        plt.close()


def generate_cross_matrix_plots(summary_df: pd.DataFrame, plots_dir: Path) -> None:
    """
    Generates cross-matrix comparison plots for BIC, Silhouette, Mean Max Posterior, and Mean Normalized Entropy.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    matrices = summary_df["matrix_name"].unique()
    palette = {"A_RAW": "#1f77b4", "A_LOG": "#ff7f0e", "B_RAW": "#2ca02c", "B_LOG": "#d62728"}

    # 1. BIC vs K across matrices
    for mat in matrices:
        sub = summary_df[summary_df["matrix_name"] == mat]
        axes[0, 0].plot(sub["K"], sub["BIC"], "o-", label=mat, color=palette.get(mat, None), linewidth=2)
    axes[0, 0].set_title("BIC vs K (Lower is better model fit)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("K")
    axes[0, 0].set_ylabel("BIC")
    axes[0, 0].legend()

    # 2. Silhouette vs K across matrices
    for mat in matrices:
        sub = summary_df[summary_df["matrix_name"] == mat]
        axes[0, 1].plot(sub["K"], sub["silhouette"], "s-", label=mat, color=palette.get(mat, None), linewidth=2)
    axes[0, 1].set_title("Silhouette Score vs K (Hard Assignments)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("K")
    axes[0, 1].set_ylabel("Silhouette Score")
    axes[0, 1].legend()

    # 3. Mean Max Posterior vs K
    for mat in matrices:
        sub = summary_df[summary_df["matrix_name"] == mat]
        axes[1, 0].plot(sub["K"], sub["mean_max_posterior"], "^-", label=mat, color=palette.get(mat, None), linewidth=2)
    axes[1, 0].set_title("Mean Max Posterior Probability vs K", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("K")
    axes[1, 0].set_ylabel("Mean Max Posterior")
    axes[1, 0].set_ylim(0.4, 1.02)
    axes[1, 0].legend()

    # 4. Mean Normalized Entropy vs K
    for mat in matrices:
        sub = summary_df[summary_df["matrix_name"] == mat]
        axes[1, 1].plot(sub["K"], sub["mean_norm_entropy"], "d-", label=mat, color=palette.get(mat, None), linewidth=2)
    axes[1, 1].set_title("Mean Normalized Entropy H / log(K) vs K", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("K")
    axes[1, 1].set_ylabel("Mean Normalized Entropy")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "gmm_cross_matrix_comparison.png", dpi=300)
    plt.close()


def run_gmm_pipeline(base_dir: Path) -> dict:
    """
    Main orchestration function for GMM Exploratory Sandbox Experiment.
    """
    t_start = time.time()
    gmm_dir = base_dir / "clustering_sandbox" / "gmm"
    
    metrics_dir = gmm_dir / "metrics"
    assignments_dir = gmm_dir / "assignments"
    probabilities_dir = gmm_dir / "probabilities"
    parameters_dir = gmm_dir / "model_parameters"
    plots_dir = gmm_dir / "plots"
    metadata_dir = gmm_dir / "metadata"

    for d in [metrics_dir, assignments_dir, probabilities_dir, parameters_dir, plots_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    scaled_dir = base_dir / "output" / "scaled_matrices"
    cohort_a_path = base_dir / "output" / "analysis_cohort_a_broad.csv"
    cohort_b_path = base_dir / "output" / "analysis_cohort_b_fasting.csv"

    matrices = {
        "A_RAW": (scaled_dir / "A_RAW_scaled.csv", cohort_a_path),
        "A_LOG": (scaled_dir / "A_LOG_scaled.csv", cohort_a_path),
        "B_RAW": (scaled_dir / "B_RAW_scaled.csv", cohort_b_path),
        "B_LOG": (scaled_dir / "B_LOG_scaled.csv", cohort_b_path),
    }

    all_metrics_list = []
    all_summaries = {}

    print("=================================================================")
    print("  EXPLORATORY CLUSTERING SANDBOX — GMM EXPERIMENT (EXPERIMENT 3)")
    print("=================================================================")

    for name, (scaled_file, cohort_file) in matrices.items():
        df_scaled = pd.read_csv(scaled_file)
        df_cohort = pd.read_csv(cohort_file)

        print(f"\n  [Running GMM Sweep K=2..7] -> {name} ({df_scaled.shape[0]:,} rows x {df_scaled.shape[1]} features)")

        metrics_df, assignments_df, param_records, summary = run_gmm_sweep_for_matrix(
            name, df_scaled, df_cohort, gmm_dir
        )

        # Export CSVs
        metrics_csv = metrics_dir / f"{name}_gmm_metrics.csv"
        assignments_csv = assignments_dir / f"{name}_gmm_assignments.csv"

        metrics_df.to_csv(metrics_csv, index=False)
        assignments_df.to_csv(assignments_csv, index=False)

        # Generate per-matrix plots
        generate_gmm_plots(name, metrics_df, assignments_df, df_scaled, plots_dir)

        all_metrics_list.append(metrics_df)
        all_summaries[name] = summary

        print(f"    - Completed 6 GMM fits (K=2..7, covariance_type='full', n_init=25)")
        print(f"    - Metrics saved to: {metrics_csv.relative_to(base_dir)}")
        print(f"    - Assignments saved to: {assignments_csv.relative_to(base_dir)}")

    # Combine all cross-matrix metrics into single summary CSV
    cross_matrix_df = pd.concat(all_metrics_list, ignore_index=True)
    cross_matrix_csv = metrics_dir / "gmm_cross_matrix_summary.csv"
    cross_matrix_df.to_csv(cross_matrix_csv, index=False)
    print(f"\n  [CROSS-MATRIX SUMMARY] -> Saved to {cross_matrix_csv.relative_to(base_dir)}")

    # Generate cross-matrix comparison plots
    generate_cross_matrix_plots(cross_matrix_df, plots_dir)

    elapsed = time.time() - t_start

    # Create metadata JSON
    metadata = {
        "experiment": "EXPLORATORY_CLUSTERING_SANDBOX_GMM",
        "status": "EXPERIMENTAL_SANDBOX_ONLY",
        "description": "Exploratory Gaussian Mixture Model (GMM) clustering sweep across K=2..7 for 4 candidate standardized feature matrices.",
        "methodological_boundaries": {
            "optimal_k_selected": False,
            "winner_declared": False,
            "subjective_labels_assigned": False,
            "biological_interpretation_performed": False,
            "tridosha_mapping_performed": False,
            "pca_used_as_input": False,
            "pca_used_for_visualization_only": True,
            "formal_decision_log_modified": False
        },
        "configuration": {
            "k_range": K_RANGE,
            "covariance_type": COVARIANCE_TYPE,
            "n_init": N_INIT,
            "random_state": RANDOM_SEED,
            "sklearn_version": sklearn.__version__,
            "total_runs_completed": len(cross_matrix_df),
            "total_runtime_seconds": round(elapsed, 2)
        },
        "matrices_analyzed": list(matrices.keys()),
        "runs_summary": {
            name: [
                {
                    "K": int(r["K"]),
                    "BIC": round(r["BIC"], 2),
                    "AIC": round(r["AIC"], 2),
                    "avg_log_likelihood": round(r["average_log_likelihood"], 4),
                    "silhouette": round(r["silhouette"], 4),
                    "mean_max_posterior": round(r["mean_max_posterior"], 4),
                    "mean_norm_entropy": round(r["mean_norm_entropy"], 4),
                    "converged": r["convergence_status"] == "CONVERGED"
                }
                for r in summ["runs"]
            ]
            for name, summ in all_summaries.items()
        }
    }

    meta_file = metadata_dir / "gmm_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Write README.md inside clustering_sandbox/gmm/
    readme_content = f"""# Gaussian Mixture Model (GMM) Exploratory Clustering Sandbox (Experiment 3)

## Research Question & Purpose

This directory contains the quantitative exploratory evidence for probabilistic **Gaussian Mixture Models (GMM)** ($K = 2 \dots 7$) fitted on the four candidate standardized feature matrices:

1. `A_RAW_scaled.csv` ($N = 4,482$, $p = 19$)
2. `A_LOG_scaled.csv` ($N = 4,482$, $p = 19$)
3. `B_RAW_scaled.csv` ($N = 967$, $p = 24$)
4. `B_LOG_scaled.csv` ($N = 967$, $p = 24$)

The core methodological question is:
> *What component structure and probabilistic assignment certainty does a full-covariance Gaussian Mixture Model uncover in the standardized biological feature space, and how does its behavior compare with deterministic K-Means partitions?*

> ⚠️ **EXPERIMENTAL SANDBOX ONLY**: These outputs are part of the **`experiment/clustering-sandbox`** branch. They are strictly quantitative diagnostic outputs and have **NOT** been selected as formal methodological decisions for the main research pipeline.

---

## Experimental & Model Configuration

- **Model Specification**: Scikit-Learn `GaussianMixture(n_components=K, covariance_type="full", n_init={N_INIT}, random_state={RANDOM_SEED})`.
- **Covariance Structure**: `"full"` covariance was chosen as the primary specification to allow full non-spherical component orientations in the standardized feature space without imposing diagonal independence.
- **K Sweep Range**: $K \in [2, 3, 4, 5, 6, 7]$ (24 total GMM fits across the 4 matrices).
- **Software**: Scikit-Learn `{sklearn.__version__}`.

---

## Quantitative Diagnostics & Metrics Reported

1. **Model Fit Diagnostics**:
   - **Log-Likelihood** (Total and Average per participant)
   - **Bayesian Information Criterion (BIC)**
   - **Akaike Information Criterion (AIC)**
   > *Note: BIC/AIC measure parametric likelihood fit under Gaussian assumptions with complexity penalties. Lowest BIC/AIC is a parametric diagnostic, NOT an absolute proof of biological ground truth.*

2. **Hard-Assignment Metrics**:
   - **Silhouette Score**
   - **Calinski-Harabasz (CH) Index**
   - **Davies-Bouldin (DB) Index**
   - **Component Sizes & Proportions** ($\min$ and $\max$ cluster proportions)

3. **Soft Assignment & Uncertainty Summaries**:
   - Participant posterior probability matrix $P(c_k \mid x_i)$ for $k=1 \dots K$.
   - Maximum posterior probability $P_{{\max, i}} = \max_k P(c_k \mid x_i)$.
   - Mean and median $P_{{\max}}$.
   - Cumulative proportion of participants exceeding confidence thresholds: $P_{{\max}} \ge 0.50, 0.60, 0.70, 0.80, 0.90$.

4. **Assignment Entropy**:
   - Participant Shannon entropy $H_i = -\sum_k p_{{ik}} \log(p_{{ik}})$.
   - Normalized entropy $H_{{{\text{{norm}}, i}}} = H_i / \log(K) \in [0, 1]$.
   - Summary percentiles (5th, 25th, 50th, 75th, 95th).

5. **Component Parameters**:
   - Component mixing weights $\pi_k$.
   - Component feature means $\boldsymbol{{\mu}}_k$.
   - Component full covariance matrices $\boldsymbol{{\Sigma}}_k$.

---

## Methodological Boundaries Confirmed

- ⚠️ **No Optimal K Selected**: No threshold or BIC/AIC minimum was used to declare a "winning" $K$.
- ⚠️ **No Subjective Labels**: Qualitative descriptors (e.g. "strong", "weak", "good", "poor") were **not** assigned.
- ⚠️ **Full Standardized Space**: Fitting was executed strictly on full 19D / 24D standardized feature space.
- ⚠️ **PCA Role**: 2D PCA coordinate projections were computed **strictly for rendering 2D visualization scatter plots** with posterior transparency overlay. PCA coordinates were **never** passed as model inputs.
- ⚠️ **Cohort & Transform Separation**: Cohort A ($N=4,482$) and Cohort B ($N=967$), as well as RAW and LOG representations, were kept 100% separate.

---

*Generated by `clustering_sandbox/gmm/run_gmm_experiment.py`*
"""

    readme_file = gmm_dir / "README.md"
    with open(readme_file, "w") as f:
        f.write(readme_content)

    print(f"\n  [SAVED METADATA] -> {meta_file.relative_to(base_dir)}")
    print(f"  [SAVED README]   -> {readme_file.relative_to(base_dir)}")
    print(f"  [TOTAL RUNTIME]  -> {elapsed:.2f} seconds")
    print("=================================================================")

    return metadata


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    run_gmm_pipeline(base_dir)
