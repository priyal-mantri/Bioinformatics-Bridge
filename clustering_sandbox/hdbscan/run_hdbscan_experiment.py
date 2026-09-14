"""
clustering_sandbox/hdbscan/run_hdbscan_experiment.py
======================================================
Experiment 5: HDBSCAN Fine-Scale Density Clustering Sandbox

PURPOSE:
  Perform a fine-scale sensitivity analysis to determine whether smaller,
  persistent density structures exist across the four Z-score standardized
  candidate matrices (A_RAW, A_LOG, B_RAW, B_LOG).

METHODOLOGY:
  - Algorithm: sklearn.cluster.HDBSCAN (scikit-learn 1.4.1.post1)
  - Metric: Euclidean distance (in full 19D or 24D standardized feature space)
  - Selection Method: "eom" (Excess of Mass)
  - PCA Role: PCA is used ONLY for 2D post-hoc visualization (Decision 013).

PARAMETER GRID (48 Total Fits):
  - Cohort A (N=4,482): min_cluster_size in [10, 15, 25, 50] x min_samples in [5, 10, None (default)]
  - Cohort B (N=967):   min_cluster_size in [5, 10, 15, 25]  x min_samples in [3, 5, 10]

METRICS & NOISE EXCLUSION RULE:
  - Noise observations (label -1) are excluded when computing internal validation metrics
    (Silhouette, Calinski-Harabasz, Davies-Bouldin). Metrics are computed only if
    number_of_clusters >= 2 and number_of_non_noise_samples > number_of_clusters.
  - Reported noise fraction and cluster counts explicitly characterize point rejection.

STABILITY ASSESSMENT:
  - Pairwise Adjusted Rand Index (ARI) and Jaccard similarity across neighboring
    parameter settings and RAW vs LOG representations.
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
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    jaccard_score
)
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Parameter Grids
# --------------------------------------------------------------------------- #
GRID_COHORT_A = {
    "min_cluster_sizes": [10, 15, 25, 50],
    "min_samples_list": [5, 10, None]  # None means default = min_cluster_size
}

GRID_COHORT_B = {
    "min_cluster_sizes": [5, 10, 15, 25],
    "min_samples_list": [3, 5, 10]
}

RANDOM_STATE = 42

# Styling
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#444444"
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["figure.dpi"] = 300


# =========================================================================== #
# Core Fitting & Evaluation
# =========================================================================== #

def run_hdbscan_fit(
    X: np.ndarray,
    matrix_name: str,
    cohort_name: str,
    min_cluster_size: int,
    min_samples: int | None
) -> tuple[dict, np.ndarray]:
    """Fit a single HDBSCAN configuration and compute evaluation metrics."""
    t0 = time.time()
    n_samples, n_features = X.shape

    min_samples_param = min_samples if min_samples is not None else min_cluster_size
    min_samples_setting_str = "default" if min_samples is None else str(min_samples)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        n_jobs=1
    )
    labels = clusterer.fit_predict(X)
    probs = clusterer.probabilities_
    elapsed = time.time() - t0

    is_noise = (labels == -1)
    n_noise = int(np.sum(is_noise))
    p_noise = float(n_noise / n_samples)
    n_non_noise = n_samples - n_noise

    unique_clusters = np.unique(labels[~is_noise]) if n_non_noise > 0 else np.array([])
    n_clusters = int(len(unique_clusters))

    if n_clusters > 0:
        counts = pd.Series(labels[~is_noise]).value_counts().to_dict()
        cluster_sizes_json = json.dumps({int(lbl): int(cnt) for lbl, cnt in sorted(counts.items())})
        proportions_json = json.dumps([round(float(cnt / n_samples), 4) for cnt in sorted(counts.values())])
        min_sz = int(min(counts.values()))
        max_sz = int(max(counts.values()))
    else:
        cluster_sizes_json = json.dumps({})
        proportions_json = json.dumps([])
        min_sz, max_sz = 0, 0

    # Probability metrics
    if n_non_noise > 0:
        mean_prob_non_noise = float(np.mean(probs[~is_noise]))
        median_prob_non_noise = float(np.median(probs[~is_noise]))
    else:
        mean_prob_non_noise = np.nan
        median_prob_non_noise = np.nan

    mean_prob_all = float(np.mean(probs))

    # Internal metrics on NON-NOISE points only
    if n_clusters >= 2 and n_non_noise > n_clusters:
        X_non_noise = X[~is_noise]
        labels_non_noise = labels[~is_noise]
        try:
            sil_non_noise = float(silhouette_score(X_non_noise, labels_non_noise))
            ch_non_noise = float(calinski_harabasz_score(X_non_noise, labels_non_noise))
            db_non_noise = float(davies_bouldin_score(X_non_noise, labels_non_noise))
        except Exception:
            sil_non_noise, ch_non_noise, db_non_noise = np.nan, np.nan, np.nan
    else:
        sil_non_noise, ch_non_noise, db_non_noise = np.nan, np.nan, np.nan

    metric_rec = {
        "matrix_name": matrix_name,
        "cohort": cohort_name,
        "N": n_samples,
        "n_features": n_features,
        "min_cluster_size": min_cluster_size,
        "min_samples_setting": min_samples_setting_str,
        "min_samples_value": min_samples_param,
        "n_clusters_excluding_noise": n_clusters,
        "n_noise_points": n_noise,
        "noise_fraction": round(p_noise, 4),
        "n_non_noise_points": n_non_noise,
        "non_noise_fraction": round(1.0 - p_noise, 4),
        "cluster_sizes": cluster_sizes_json,
        "cluster_proportions": proportions_json,
        "min_cluster_size_actual": min_sz,
        "max_cluster_size_actual": max_sz,
        "silhouette_non_noise": round(sil_non_noise, 4) if not np.isnan(sil_non_noise) else None,
        "calinski_harabasz_non_noise": round(ch_non_noise, 2) if not np.isnan(ch_non_noise) else None,
        "davies_bouldin_non_noise": round(db_non_noise, 3) if not np.isnan(db_non_noise) else None,
        "mean_prob_non_noise": round(mean_prob_non_noise, 4) if not np.isnan(mean_prob_non_noise) else None,
        "median_prob_non_noise": round(median_prob_non_noise, 4) if not np.isnan(median_prob_non_noise) else None,
        "mean_prob_all": round(mean_prob_all, 4),
        "runtime_seconds": round(elapsed, 3),
        "sklearn_version": sklearn.__version__
    }

    return metric_rec, labels


# =========================================================================== #
# Stability & Cross-Configuration Comparison
# =========================================================================== #

def compute_stability_matrix(
    assignments_df: pd.DataFrame,
    config_labels: list[str]
) -> pd.DataFrame:
    """Compute pairwise Adjusted Rand Index (ARI) across parameter configurations."""
    n_configs = len(config_labels)
    ari_matrix = np.ones((n_configs, n_configs))

    for i in range(n_configs):
        for j in range(i + 1, n_configs):
            l1 = assignments_df[config_labels[i]].values
            l2 = assignments_df[config_labels[j]].values
            ari = adjusted_rand_score(l1, l2)
            ari_matrix[i, j] = round(float(ari), 4)
            ari_matrix[j, i] = round(float(ari), 4)

    return pd.DataFrame(ari_matrix, index=config_labels, columns=config_labels)


# =========================================================================== #
# Plotting Functions
# =========================================================================== #

def create_experiment_plots(df_summary: pd.DataFrame, plots_dir: Path) -> None:
    """Create visualization plots for HDBSCAN Fine-Scale Experiment."""
    matrix_names = ["A_RAW", "A_LOG", "B_RAW", "B_LOG"]

    # Plot 1: Summary of Clusters & Noise Fraction
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    for col_idx, name in enumerate(matrix_names):
        sub_df = df_summary[df_summary["matrix_name"] == name]

        # Top row: Clusters Found vs min_cluster_size
        ax_top = axes[0, col_idx]
        for ms_set in sub_df["min_samples_setting"].unique():
            sub_ms = sub_df[sub_df["min_samples_setting"] == ms_set]
            ax_top.plot(
                sub_ms["min_cluster_size"], sub_ms["n_clusters_excluding_noise"],
                "o-", label=f"ms={ms_set}", linewidth=1.8, markersize=5
            )
        ax_top.set_title(f"{name}\nClusters Found (excl. noise)", fontsize=10, fontweight="bold")
        ax_top.set_xlabel("min_cluster_size")
        ax_top.set_ylabel("Clusters Found")
        ax_top.legend(fontsize=8)

        # Bottom row: Noise Fraction vs min_cluster_size
        ax_bot = axes[1, col_idx]
        for ms_set in sub_df["min_samples_setting"].unique():
            sub_ms = sub_df[sub_df["min_samples_setting"] == ms_set]
            ax_bot.plot(
                sub_ms["min_cluster_size"], sub_ms["noise_fraction"] * 100,
                "s--", label=f"ms={ms_set}", linewidth=1.8, markersize=5
            )
        ax_bot.set_title(f"{name}\nNoise Percentage (%)", fontsize=10, fontweight="bold")
        ax_bot.set_xlabel("min_cluster_size")
        ax_bot.set_ylabel("Noise %")
        ax_bot.set_ylim(-5, 105)
        ax_bot.legend(fontsize=8)

    plt.suptitle(
        "Experiment 5: HDBSCAN Fine-Scale Density Clustering Parameter Grid\n"
        "(Clusters Found and Noise Percentage across min_cluster_size and min_samples)",
        fontsize=12, fontweight="bold", y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plots_dir / "hdbscan_noise_and_clusters_grid.png", dpi=300)
    plt.close()

    print("  [PLOTS] Main parameter grid summary plot created.")


def create_pca_posthoc_plots(
    base_dir: Path,
    assignments_dict: dict,
    plots_dir: Path
) -> None:
    """Generate PCA 2D post-hoc visualizations for selected configurations."""
    scaled_dir = base_dir / "output" / "scaled_matrices"

    matrices_configs = {
        "A_RAW": (scaled_dir / "A_RAW_scaled.csv", ["mcs10_ms5", "mcs15_ms5", "mcs25_ms5", "mcs50_ms5"]),
        "A_LOG": (scaled_dir / "A_LOG_scaled.csv", ["mcs10_ms5", "mcs15_ms5", "mcs25_ms5", "mcs50_ms5"]),
        "B_RAW": (scaled_dir / "B_RAW_scaled.csv", ["mcs5_ms3", "mcs10_ms3", "mcs15_ms3", "mcs25_ms3"]),
        "B_LOG": (scaled_dir / "B_LOG_scaled.csv", ["mcs5_ms3", "mcs10_ms3", "mcs15_ms3", "mcs25_ms3"]),
    }

    for mat_name, (scaled_path, config_cols) in matrices_configs.items():
        df_scaled = pd.read_csv(scaled_path)
        X = df_scaled.values.astype(np.float64)
        df_ass = assignments_dict[mat_name]

        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X)
        var_exp = pca.explained_variance_ratio_ * 100

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        axes = axes.flatten()

        for idx, col in enumerate(config_cols):
            ax = axes[idx]
            labels = df_ass[col].values
            is_noise = (labels == -1)

            # Noise points in gray
            ax.scatter(
                X_pca[is_noise, 0], X_pca[is_noise, 1],
                c="#aaaaaa", label="Noise (-1)", s=10, alpha=0.4, edgecolors="none"
            )
            # Non-noise points in tab10
            if np.sum(~is_noise) > 0:
                ax.scatter(
                    X_pca[~is_noise, 0], X_pca[~is_noise, 1],
                    c=labels[~is_noise], cmap="tab10", s=14, alpha=0.7, edgecolors="none"
                )

            n_cl = len(np.unique(labels[~is_noise])) if np.sum(~is_noise) > 0 else 0
            n_ns = np.sum(is_noise)
            pct_ns = (n_ns / len(X)) * 100

            ax.set_title(
                f"{mat_name} ({col})\n"
                f"Clusters: {n_cl} | Noise: {n_ns}/{len(X)} ({pct_ns:.1f}%)",
                fontsize=10, fontweight="bold"
            )
            ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var)")
            ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var)")

        plt.suptitle(
            f"HDBSCAN Cluster Assignments Visualized on PCA Projection ({mat_name})\n"
            "[POST-HOC VISUALIZATION ONLY — Clustering was run in full standardized feature space]",
            fontsize=11, fontweight="bold", y=0.99
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(plots_dir / f"hdbscan_pca_projections_{mat_name}.png", dpi=300)
        plt.close()

    print("  [PLOTS] All PCA 2D post-hoc projection plots created.")


# =========================================================================== #
# Main Driver
# =========================================================================== #

def run_hdbscan_experiment(base_dir: Path) -> None:
    exp_dir = base_dir / "clustering_sandbox" / "hdbscan"
    metrics_dir = exp_dir / "metrics"
    assignments_dir = exp_dir / "cluster_assignments"
    plots_dir = exp_dir / "plots"
    metadata_dir = exp_dir / "metadata"
    for d in [metrics_dir, assignments_dir, plots_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    scaled_dir = base_dir / "output" / "scaled_matrices"
    cohort_a_path = base_dir / "output" / "analysis_cohort_a_broad.csv"
    cohort_b_path = base_dir / "output" / "analysis_cohort_b_fasting.csv"

    df_cohort_a = pd.read_csv(cohort_a_path)
    df_cohort_b = pd.read_csv(cohort_b_path)

    matrices = {
        "A_RAW": (scaled_dir / "A_RAW_scaled.csv", df_cohort_a, "Cohort_A", GRID_COHORT_A),
        "A_LOG": (scaled_dir / "A_LOG_scaled.csv", df_cohort_a, "Cohort_A", GRID_COHORT_A),
        "B_RAW": (scaled_dir / "B_RAW_scaled.csv", df_cohort_b, "Cohort_B", GRID_COHORT_B),
        "B_LOG": (scaled_dir / "B_LOG_scaled.csv", df_cohort_b, "Cohort_B", GRID_COHORT_B),
    }

    all_metrics_records = []
    assignments_by_matrix = {}
    metadata_summary = {
        "experiment": "EXPERIMENT_5_HDBSCAN_FINE_SCALE_DENSITY_SANDBOX",
        "hdbscan_implementation": f"sklearn.cluster.HDBSCAN (scikit-learn {sklearn.__version__})",
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "grid_cohort_a": GRID_COHORT_A,
        "grid_cohort_b": GRID_COHORT_B,
        "total_fits_executed": 48,
        "matrices_summary": {}
    }

    t_start = time.time()

    print("=" * 70)
    print("  EXPERIMENT 5: HDBSCAN FINE-SCALE DENSITY CLUSTERING")
    print("  Evaluating 48 total parameter configurations across 4 matrices")
    print("=" * 70)

    for mat_name, (scaled_path, df_cohort, cohort_name, grid) in matrices.items():
        df_scaled = pd.read_csv(scaled_path)
        X = df_scaled.values.astype(np.float64)
        n, p = X.shape
        seqn = df_cohort["SEQN"].values if "SEQN" in df_cohort.columns else np.arange(n)

        print(f"\n{'─'*60}")
        print(f"  Matrix: {mat_name} | Cohort: {cohort_name} | N={n:,} | p={p}")
        print(f"{'─'*60}")

        mat_metrics = []
        assignments_dict = {"SEQN": seqn}

        for mcs in grid["min_cluster_sizes"]:
            for ms in grid["min_samples_list"]:
                ms_str = "default" if ms is None else str(ms)
                col_name = f"mcs{mcs}_ms{ms_str}"

                metric_rec, labels = run_hdbscan_fit(X, mat_name, cohort_name, mcs, ms)
                mat_metrics.append(metric_rec)
                all_metrics_records.append(metric_rec)
                assignments_dict[col_name] = labels

                n_cl = metric_rec["n_clusters_excluding_noise"]
                n_ns = metric_rec["n_noise_points"]
                p_ns = metric_rec["noise_fraction"] * 100
                sil = metric_rec["silhouette_non_noise"]
                sil_str = f"{sil:.4f}" if sil is not None else "N/A"

                print(
                    f"    mcs={mcs:2d} | ms={ms_str:>7s} -> Clusters={n_cl:2d} | "
                    f"Noise={n_ns:4d} ({p_ns:5.1f}%) | Non-Noise Sil={sil_str}"
                )

        df_mat_metrics = pd.DataFrame(mat_metrics)
        df_mat_metrics.to_csv(metrics_dir / f"{mat_name}_hdbscan_metrics.csv", index=False)

        df_mat_assignments = pd.DataFrame(assignments_dict)
        df_mat_assignments.to_csv(assignments_dir / f"{mat_name}_hdbscan_assignments.csv", index=False)
        assignments_by_matrix[mat_name] = df_mat_assignments

        metadata_summary["matrices_summary"][mat_name] = {
            "N": n,
            "p": p,
            "total_configs": len(mat_metrics),
            "configs_with_clusters": int(sum(1 for m in mat_metrics if m["n_clusters_excluding_noise"] > 0)),
            "max_clusters_found": int(max(m["n_clusters_excluding_noise"] for m in mat_metrics)),
            "min_noise_fraction": float(min(m["noise_fraction"] for m in mat_metrics)),
            "max_noise_fraction": float(max(m["noise_fraction"] for m in mat_metrics)),
        }

    # Save Cross-Matrix Metrics Summary
    df_metrics_cross = pd.DataFrame(all_metrics_records)
    df_metrics_cross.to_csv(metrics_dir / "hdbscan_cross_matrix_summary.csv", index=False)

    # Compute Stability Summary (ARI between parameter settings within each matrix)
    stability_records = []
    for mat_name, df_ass in assignments_by_matrix.items():
        config_cols = [c for c in df_ass.columns if c != "SEQN"]
        df_ari = compute_stability_matrix(df_ass, config_cols)

        # Average ARI across distinct config pairs
        upper_tri = df_ari.values[np.triu_indices_from(df_ari.values, k=1)]
        mean_ari = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 1.0
        min_ari = float(np.min(upper_tri)) if len(upper_tri) > 0 else 1.0
        max_ari = float(np.max(upper_tri)) if len(upper_tri) > 0 else 1.0

        stability_records.append({
            "matrix_name": mat_name,
            "mean_pairwise_ARI": round(mean_ari, 4),
            "min_pairwise_ARI": round(min_ari, 4),
            "max_pairwise_ARI": round(max_ari, 4),
        })

    df_stability = pd.DataFrame(stability_records)
    df_stability.to_csv(metrics_dir / "hdbscan_stability_summary.csv", index=False)

    # Create Plots
    print(f"\n{'─'*60}")
    print("  Generating visualization plots ...")
    create_experiment_plots(df_metrics_cross, plots_dir)
    create_pca_posthoc_plots(base_dir, assignments_by_matrix, plots_dir)

    metadata_summary["total_runtime_seconds"] = round(time.time() - t_start, 2)
    with open(metadata_dir / "hdbscan_experiment_metadata.json", "w") as f:
        json.dump(metadata_summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 5 COMPLETE")
    print(f"  Total Runtime: {metadata_summary['total_runtime_seconds']:.2f}s")
    print(f"  Outputs saved in: clustering_sandbox/hdbscan/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    run_hdbscan_experiment(base_dir)
