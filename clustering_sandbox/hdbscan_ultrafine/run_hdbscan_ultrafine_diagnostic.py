"""
clustering_sandbox/hdbscan_ultrafine/run_hdbscan_ultrafine_diagnostic.py
==========================================================================
READ-ONLY HDBSCAN Ultra-Fine Scale Sensitivity Diagnostic.

PURPOSE:
  Evaluate HDBSCAN behavior at ultra-fine parameter scales to determine whether
  smaller density thresholds (min_cluster_size in {3, 5, 7, 10}, min_samples in {2, 3, 5})
  reveal persistent, reproducible local density structures or merely increasingly
  fragmented micro-clusters.

THIS SCRIPT DOES NOT:
  - Select parameters to force 3 clusters or biological claims.
  - Modify the master methodology decision log.
  - Commit anything.
  - Perform Ayurvedic/dosha comparisons.

METHODOLOGY & CONSTRAINTS:
  - Algorithm: sklearn.cluster.HDBSCAN (scikit-learn 1.4.1.post1)
  - Metric: Euclidean distance in full 19D or 24D standardized feature space.
  - Cluster Selection Method: "eom" (Excess of Mass)
  - PCA Role: PCA is used ONLY for 2D post-hoc visualization (Decision 013).

PARAMETER GRID (48 Total Fits):
  - Cohort A (N=4,482, 19D): min_cluster_size in {3, 5, 7, 10} x min_samples in {2, 3, 5} (12 fits)
  - Cohort B (N=967,  24D): min_cluster_size in {3, 5, 7, 10} x min_samples in {2, 3, 5} (12 fits)

METRICS & NOISE EXCLUSION RULE:
  - Internal metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin) are computed
    STRICTLY on non-noise observations (label != -1) when number_of_clusters >= 2
    and number_of_non_noise_observations > number_of_clusters.
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
    adjusted_rand_score
)
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Grid Configurations
# --------------------------------------------------------------------------- #
MIN_CLUSTER_SIZES = [3, 5, 7, 10]
MIN_SAMPLES_LIST = [2, 3, 5]
RANDOM_STATE = 42

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

def evaluate_ultrafine_config(
    X: np.ndarray,
    matrix_name: str,
    mcs: int,
    ms: int
) -> tuple[dict, np.ndarray]:
    """Fit HDBSCAN with ultra-fine parameters and compute metrics."""
    t0 = time.time()
    n_samples, n_features = X.shape

    clusterer = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
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
    p_non_noise = float(n_non_noise / n_samples)

    unique_clusters = np.unique(labels[~is_noise]) if n_non_noise > 0 else np.array([])
    n_clusters = int(len(unique_clusters))

    if n_clusters > 0:
        counts = pd.Series(labels[~is_noise]).value_counts().to_dict()
        cluster_sizes_json = json.dumps({int(lbl): int(cnt) for lbl, cnt in sorted(counts.items())})

        # Non-noise cluster proportions
        non_noise_props = [float(cnt / n_non_noise) for cnt in counts.values()]
        cohort_props = [float(cnt / n_samples) for cnt in counts.values()]

        min_sz = int(min(counts.values()))
        max_sz = int(max(counts.values()))
        min_prop_cohort = round(float(min_sz / n_samples), 5)
        max_prop_cohort = round(float(max_sz / n_samples), 5)
        mean_prop_non_noise = round(float(np.mean(non_noise_props)), 5)
    else:
        cluster_sizes_json = json.dumps({})
        min_sz, max_sz = 0, 0
        min_prop_cohort, max_prop_cohort = 0.0, 0.0
        mean_prop_non_noise = 0.0

    # Probability stats
    if n_non_noise > 0:
        mean_prob_non_noise = float(np.mean(probs[~is_noise]))
        median_prob_non_noise = float(np.median(probs[~is_noise]))
    else:
        mean_prob_non_noise = np.nan
        median_prob_non_noise = np.nan

    mean_prob_all = float(np.mean(probs))

    # Internal validation metrics on NON-NOISE points
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

    rec = {
        "matrix": matrix_name,
        "N": n_samples,
        "dimensionality": n_features,
        "min_cluster_size": mcs,
        "min_samples": ms,
        "number_of_clusters_excluding_noise": n_clusters,
        "number_of_noise_points": n_noise,
        "noise_fraction": round(p_noise, 4),
        "number_of_non_noise_points": n_non_noise,
        "non_noise_fraction": round(p_non_noise, 4),
        "cluster_sizes": cluster_sizes_json,
        "minimum_cluster_size_actual": min_sz,
        "maximum_cluster_size_actual": max_sz,
        "min_cluster_proportion_cohort": min_prop_cohort,
        "max_cluster_proportion_cohort": max_prop_cohort,
        "mean_cluster_proportion_non_noise": mean_prop_non_noise,
        "silhouette_non_noise": round(sil_non_noise, 4) if not np.isnan(sil_non_noise) else None,
        "calinski_harabasz_non_noise": round(ch_non_noise, 2) if not np.isnan(ch_non_noise) else None,
        "davies_bouldin_non_noise": round(db_non_noise, 3) if not np.isnan(db_non_noise) else None,
        "mean_prob_non_noise": round(mean_prob_non_noise, 4) if not np.isnan(mean_prob_non_noise) else None,
        "median_prob_non_noise": round(median_prob_non_noise, 4) if not np.isnan(median_prob_non_noise) else None,
        "mean_prob_all": round(mean_prob_all, 4),
        "runtime_seconds": round(elapsed, 3),
        "sklearn_version": sklearn.__version__
    }

    return rec, labels


# =========================================================================== #
# Stability & Cross-Configuration Metrics
# =========================================================================== #

def compute_stability_metrics(
    assignments_df: pd.DataFrame,
    config_cols: list[str]
) -> pd.DataFrame:
    """
    Compute pairwise ARI and non-noise overlap proportion between configurations.
    """
    records = []
    n_cols = len(config_cols)

    for i in range(n_cols):
        c1 = config_cols[i]
        l1 = assignments_df[c1].values

        for j in range(i + 1, n_cols):
            c2 = config_cols[j]
            l2 = assignments_df[c2].values

            # ARI on full cohort (including noise as label -1)
            ari_full = adjusted_rand_score(l1, l2)

            # Non-noise overlap: points that are non-noise in BOTH configurations
            both_non_noise = (l1 != -1) & (l2 != -1)
            n_both_non_noise = int(np.sum(both_non_noise))

            if n_both_non_noise > 1:
                ari_non_noise_overlap = adjusted_rand_score(l1[both_non_noise], l2[both_non_noise])
                ari_nn_val = round(float(ari_non_noise_overlap), 4)
            else:
                ari_nn_val = None

            records.append({
                "config_1": c1,
                "config_2": c2,
                "ari_full_cohort": round(float(ari_full), 4),
                "n_both_non_noise": n_both_non_noise,
                "ari_non_noise_overlap": ari_nn_val
            })

    return pd.DataFrame(records)


# =========================================================================== #
# Visualizations
# =========================================================================== #

def create_ultrafine_plots(df_summary: pd.DataFrame, plots_dir: Path) -> None:
    matrix_names = ["A_RAW", "A_LOG", "B_RAW", "B_LOG"]

    # ---- Plot 1: Cluster Count vs min_cluster_size ------------------------- #
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, name in enumerate(matrix_names):
        ax = axes[idx]
        sub = df_summary[df_summary["matrix"] == name]

        for ms in MIN_SAMPLES_LIST:
            sub_ms = sub[sub["min_samples"] == ms]
            ax.plot(
                sub_ms["min_cluster_size"], sub_ms["number_of_clusters_excluding_noise"],
                "o-", label=f"min_samples={ms}", linewidth=2, markersize=6
            )
        ax.set_title(f"{name} — Non-Noise Clusters Found", fontsize=11, fontweight="bold")
        ax.set_xlabel("min_cluster_size")
        ax.set_ylabel("Clusters Found (excl. noise)")
        ax.set_xticks(MIN_CLUSTER_SIZES)
        ax.legend(fontsize=9)

    plt.suptitle(
        "HDBSCAN Ultra-Fine Diagnostic 1: Non-Noise Clusters Found vs min_cluster_size\n"
        "(Evaluated across min_samples ∈ {2, 3, 5})",
        fontsize=12, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plots_dir / "hdbscan_ultrafine_clusters_vs_params.png", dpi=300)
    plt.close()

    # ---- Plot 2: Noise Percentage vs min_cluster_size ---------------------- #
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, name in enumerate(matrix_names):
        ax = axes[idx]
        sub = df_summary[df_summary["matrix"] == name]

        for ms in MIN_SAMPLES_LIST:
            sub_ms = sub[sub["min_samples"] == ms]
            ax.plot(
                sub_ms["min_cluster_size"], sub_ms["noise_fraction"] * 100,
                "s--", label=f"min_samples={ms}", linewidth=2, markersize=6
            )
        ax.set_title(f"{name} — Noise Percentage (%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("min_cluster_size")
        ax.set_ylabel("Noise Percentage (%)")
        ax.set_xticks(MIN_CLUSTER_SIZES)
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=9)

    plt.suptitle(
        "HDBSCAN Ultra-Fine Diagnostic 2: Noise Percentage vs min_cluster_size\n"
        "[High noise indicates diffuse data space; lower noise requires permissive min_samples]",
        fontsize=12, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plots_dir / "hdbscan_ultrafine_noise_vs_params.png", dpi=300)
    plt.close()

    # ---- Plot 3: Non-Noise Participant Count vs min_cluster_size ----------- #
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, name in enumerate(matrix_names):
        ax = axes[idx]
        sub = df_summary[df_summary["matrix"] == name]

        for ms in MIN_SAMPLES_LIST:
            sub_ms = sub[sub["min_samples"] == ms]
            ax.plot(
                sub_ms["min_cluster_size"], sub_ms["number_of_non_noise_points"],
                "^--", label=f"min_samples={ms}", linewidth=2, markersize=6
            )
        ax.set_title(f"{name} — Non-Noise Participant Count", fontsize=11, fontweight="bold")
        ax.set_xlabel("min_cluster_size")
        ax.set_ylabel("Non-Noise Participants")
        ax.set_xticks(MIN_CLUSTER_SIZES)
        ax.legend(fontsize=9)

    plt.suptitle(
        "HDBSCAN Ultra-Fine Diagnostic 3: Non-Noise Participant Count vs min_cluster_size\n"
        "(Shows how many participants remain in valid clusters as parameters scale)",
        fontsize=12, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plots_dir / "hdbscan_ultrafine_non_noise_count.png", dpi=300)
    plt.close()

    print("  [PLOTS] Ultra-fine summary plots 1, 2, 3 created.")


def create_pca_projections(
    base_dir: Path,
    assignments_dict: dict,
    plots_dir: Path
) -> None:
    """Generate PCA 2D post-hoc projections for representative ultra-fine configs."""
    scaled_dir = base_dir / "output" / "scaled_matrices"

    matrices_configs = {
        "A_RAW": (scaled_dir / "A_RAW_scaled.csv", ["mcs3_ms2", "mcs5_ms2", "mcs7_ms2", "mcs10_ms2"]),
        "A_LOG": (scaled_dir / "A_LOG_scaled.csv", ["mcs3_ms2", "mcs5_ms2", "mcs7_ms2", "mcs10_ms2"]),
        "B_RAW": (scaled_dir / "B_RAW_scaled.csv", ["mcs3_ms2", "mcs5_ms2", "mcs7_ms2", "mcs10_ms2"]),
        "B_LOG": (scaled_dir / "B_LOG_scaled.csv", ["mcs3_ms2", "mcs5_ms2", "mcs7_ms2", "mcs10_ms2"]),
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

            # Gray noise points
            ax.scatter(
                X_pca[is_noise, 0], X_pca[is_noise, 1],
                c="#aaaaaa", label="Noise (-1)", s=10, alpha=0.4, edgecolors="none"
            )
            # Non-noise clusters
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
            f"HDBSCAN Ultra-Fine Cluster Assignments on PCA Projection ({mat_name})\n"
            "[POST-HOC VISUALIZATION ONLY — Clustering was run in full standardized feature space]",
            fontsize=11, fontweight="bold", y=0.99
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(plots_dir / f"hdbscan_ultrafine_pca_projections_{mat_name}.png", dpi=300)
        plt.close()

    print("  [PLOTS] All PCA 2D post-hoc projection plots created.")


# =========================================================================== #
# Main Driver
# =========================================================================== #

def run_hdbscan_ultrafine_diagnostics(base_dir: Path) -> None:
    diag_dir = base_dir / "clustering_sandbox" / "hdbscan_ultrafine"
    metrics_dir = diag_dir / "metrics"
    assignments_dir = diag_dir / "cluster_assignments"
    plots_dir = diag_dir / "plots"
    metadata_dir = diag_dir / "metadata"
    for d in [metrics_dir, assignments_dir, plots_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    scaled_dir = base_dir / "output" / "scaled_matrices"
    cohort_a_path = base_dir / "output" / "analysis_cohort_a_broad.csv"
    cohort_b_path = base_dir / "output" / "analysis_cohort_b_fasting.csv"

    df_cohort_a = pd.read_csv(cohort_a_path)
    df_cohort_b = pd.read_csv(cohort_b_path)

    matrices = {
        "A_RAW": (scaled_dir / "A_RAW_scaled.csv", df_cohort_a),
        "A_LOG": (scaled_dir / "A_LOG_scaled.csv", df_cohort_a),
        "B_RAW": (scaled_dir / "B_RAW_scaled.csv", df_cohort_b),
        "B_LOG": (scaled_dir / "B_LOG_scaled.csv", df_cohort_b),
    }

    all_metrics_records = []
    assignments_by_matrix = {}

    t_start = time.time()

    print("=" * 70)
    print("  HDBSCAN ULTRA-FINE SCALE SENSITIVITY DIAGNOSTIC")
    print("  Evaluating min_cluster_size ∈ {3,5,7,10} × min_samples ∈ {2,3,5}")
    print("=" * 70)

    for mat_name, (scaled_path, df_cohort) in matrices.items():
        df_scaled = pd.read_csv(scaled_path)
        X = df_scaled.values.astype(np.float64)
        n, p = X.shape
        seqn = df_cohort["SEQN"].values if "SEQN" in df_cohort.columns else np.arange(n)

        print(f"\n{'─'*60}")
        print(f"  Matrix: {mat_name} | N={n:,} | p={p}")
        print(f"{'─'*60}")

        mat_metrics = []
        assignments_dict = {"SEQN": seqn}

        for mcs in MIN_CLUSTER_SIZES:
            for ms in MIN_SAMPLES_LIST:
                col_name = f"mcs{mcs}_ms{ms}"

                rec, labels = evaluate_ultrafine_config(X, mat_name, mcs, ms)
                mat_metrics.append(rec)
                all_metrics_records.append(rec)
                assignments_dict[col_name] = labels

                n_cl = rec["number_of_clusters_excluding_noise"]
                n_ns = rec["number_of_noise_points"]
                p_ns = rec["noise_fraction"] * 100
                sil = rec["silhouette_non_noise"]
                sil_str = f"{sil:.4f}" if sil is not None else "N/A"

                print(
                    f"    mcs={mcs:2d} | ms={ms:2d} -> Clusters={n_cl:2d} | "
                    f"Noise={n_ns:4d} ({p_ns:5.1f}%) | Non-Noise Sil={sil_str}"
                )

        df_mat_metrics = pd.DataFrame(mat_metrics)
        df_mat_metrics.to_csv(metrics_dir / f"{mat_name}_hdbscan_ultrafine_metrics.csv", index=False)

        df_mat_assignments = pd.DataFrame(assignments_dict)
        df_mat_assignments.to_csv(assignments_dir / f"{mat_name}_hdbscan_ultrafine_assignments.csv", index=False)
        assignments_by_matrix[mat_name] = df_mat_assignments

    # Cross-Matrix Summary CSV
    df_metrics_cross = pd.DataFrame(all_metrics_records)
    df_metrics_cross.to_csv(metrics_dir / "hdbscan_ultrafine_cross_matrix_summary.csv", index=False)

    # Stability Summary CSV (pairwise ARI between config pairs)
    stability_list = []
    for mat_name, df_ass in assignments_by_matrix.items():
        config_cols = [c for c in df_ass.columns if c != "SEQN"]
        df_stab = compute_stability_metrics(df_ass, config_cols)
        df_stab.insert(0, "matrix", mat_name)
        stability_list.append(df_stab)

    df_stability_all = pd.concat(stability_list, ignore_index=True)
    df_stability_all.to_csv(metrics_dir / "hdbscan_ultrafine_stability_summary.csv", index=False)

    # Create Plots
    print(f"\n{'─'*60}")
    print("  Generating diagnostic plots ...")
    create_ultrafine_plots(df_metrics_cross, plots_dir)
    create_pca_projections(base_dir, assignments_by_matrix, plots_dir)

    metadata = {
        "diagnostic": "HDBSCAN_ULTRAFINE_SCALE_SENSITIVITY_DIAGNOSTIC",
        "hdbscan_implementation": f"sklearn.cluster.HDBSCAN (scikit-learn {sklearn.__version__})",
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "min_cluster_sizes_tested": MIN_CLUSTER_SIZES,
        "min_samples_tested": MIN_SAMPLES_LIST,
        "total_configurations_tested": len(all_metrics_records),
        "total_runtime_seconds": round(time.time() - t_start, 2)
    }

    with open(metadata_dir / "hdbscan_ultrafine_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  ULTRA-FINE DIAGNOSTIC COMPLETE")
    print(f"  Total Runtime: {metadata['total_runtime_seconds']:.2f} seconds")
    print(f"  Outputs saved in: clustering_sandbox/hdbscan_ultrafine/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    run_hdbscan_ultrafine_diagnostics(base_dir)
