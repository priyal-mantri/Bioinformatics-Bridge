"""
clustering_sandbox/hdbscan_diagnostics/run_hdbscan_calibration_diagnostic.py
================================================================================
READ-ONLY HDBSCAN Parameter-Calibration Diagnostic — Experiment 5 Pre-Stage.

PURPOSE:
  Empirically characterize HDBSCAN clustering behavior across a range of
  density parameters (min_cluster_size, min_samples) for all four Z-score
  standardized candidate feature matrices (A_RAW, A_LOG, B_RAW, B_LOG).

THIS SCRIPT DOES NOT:
  - Select a single "optimal" HDBSCAN parameter configuration.
  - Declare a winning model or biological cluster count.
  - Relate results to Ayurvedic Tridosha/Prakriti.
  - Modify the master methodology decision log.
  - Commit anything.

METHODOLOGY & CONSTRAINTS:
  - HDBSCAN Implementation: sklearn.cluster.HDBSCAN (scikit-learn 1.4.1.post1)
  - Distance Metric: Euclidean
  - Cluster Selection Method: "eom" (Excess of Mass)
  - Clustering Space: Full standardized feature space (19D for A, 24D for B).
  - PCA Role: PCA is used ONLY for 2D post-hoc visualization (Decision 013).

PARAMETER GRID:
  - Cohort A (N=4,482): min_cluster_size in {25, 50, 100, 200} (~0.5% - 4.5% of N)
  - Cohort B (N=967):   min_cluster_size in {10, 25, 50, 100} (~1.0% - 10.3% of N)
  - min_samples in {None (default = min_cluster_size), 5, 15}

NOISE EXCLUSION FOR INTERNAL METRICS:
  Noise points (label -1) are explicitly EXCLUDED when calculating Silhouette,
  Calinski-Harabasz, and Davies-Bouldin scores to prevent treating noise as an
  ordinary biological cluster.
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
    davies_bouldin_score
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Parameter Grids
# --------------------------------------------------------------------------- #
GRID_COHORT_A = {
    "min_cluster_sizes": [25, 50, 100, 200],
    "min_samples_list": [None, 5, 15]
}

GRID_COHORT_B = {
    "min_cluster_sizes": [10, 25, 50, 100],
    "min_samples_list": [None, 5, 15]
}

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
# Helper Functions
# =========================================================================== #

def evaluate_hdbscan_config(
    X: np.ndarray,
    matrix_name: str,
    min_cluster_size: int,
    min_samples: int | None
) -> dict:
    """
    Fit sklearn.cluster.HDBSCAN with specified min_cluster_size and min_samples,
    and compute detailed diagnostic metrics.
    """
    t0 = time.time()
    n_samples, n_features = X.shape

    min_samples_param = min_samples if min_samples is not None else min_cluster_size
    min_samples_setting_str = "default (min_cluster_size)" if min_samples is None else str(min_samples)

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

    # Separate non-noise vs noise points
    is_noise = (labels == -1)
    n_noise = int(np.sum(is_noise))
    p_noise = float(n_noise / n_samples)
    n_non_noise = n_samples - n_noise
    p_non_noise = float(n_non_noise / n_samples)

    # Unique non-noise clusters
    non_noise_labels = labels[~is_noise]
    unique_clusters = np.unique(non_noise_labels) if len(non_noise_labels) > 0 else np.array([])
    n_clusters = int(len(unique_clusters))

    # Cluster sizes
    if n_clusters > 0:
        counts = pd.Series(non_noise_labels).value_counts().to_dict()
        cluster_sizes_json = json.dumps({int(lbl): int(cnt) for lbl, cnt in sorted(counts.items())})
        min_sz = int(min(counts.values()))
        max_sz = int(max(counts.values()))
        min_prop = round(float(min_sz / n_samples), 4)
        max_prop = round(float(max_sz / n_samples), 4)
    else:
        cluster_sizes_json = json.dumps({})
        min_sz, max_sz = 0, 0
        min_prop, max_prop = 0.0, 0.0

    # Membership probability stats
    if n_non_noise > 0:
        mean_prob_non_noise = float(np.mean(probs[~is_noise]))
        median_prob_non_noise = float(np.median(probs[~is_noise]))
    else:
        mean_prob_non_noise = np.nan
        median_prob_non_noise = np.nan

    mean_prob_all = float(np.mean(probs))

    # Internal metrics on non-noise points (if n_clusters >= 2 and n_non_noise > n_clusters)
    if n_clusters >= 2 and n_non_noise > n_clusters:
        X_non_noise = X[~is_noise]
        try:
            sil_non_noise = float(silhouette_score(X_non_noise, non_noise_labels))
            ch_non_noise = float(calinski_harabasz_score(X_non_noise, non_noise_labels))
            db_non_noise = float(davies_bouldin_score(X_non_noise, non_noise_labels))
        except Exception as e:
            sil_non_noise, ch_non_noise, db_non_noise = np.nan, np.nan, np.nan
    else:
        sil_non_noise, ch_non_noise, db_non_noise = np.nan, np.nan, np.nan

    return {
        "matrix_name": matrix_name,
        "N": n_samples,
        "n_features": n_features,
        "min_cluster_size": min_cluster_size,
        "min_samples_setting": min_samples_setting_str,
        "min_samples_value": min_samples_param,
        "n_clusters_found": n_clusters,
        "n_noise_samples": n_noise,
        "noise_proportion": round(p_noise, 4),
        "n_non_noise_samples": n_non_noise,
        "non_noise_proportion": round(p_non_noise, 4),
        "cluster_sizes": cluster_sizes_json,
        "min_cluster_size_actual": min_sz,
        "max_cluster_size_actual": max_sz,
        "min_cluster_proportion_actual": min_prop,
        "max_cluster_proportion_actual": max_prop,
        "mean_membership_prob_non_noise": round(mean_prob_non_noise, 4) if not np.isnan(mean_prob_non_noise) else None,
        "median_membership_prob_non_noise": round(median_prob_non_noise, 4) if not np.isnan(median_prob_non_noise) else None,
        "mean_membership_prob_all": round(mean_prob_all, 4),
        "silhouette_non_noise": round(sil_non_noise, 4) if not np.isnan(sil_non_noise) else None,
        "calinski_harabasz_non_noise": round(ch_non_noise, 2) if not np.isnan(ch_non_noise) else None,
        "davies_bouldin_non_noise": round(db_non_noise, 3) if not np.isnan(db_non_noise) else None,
        "sklearn_version": sklearn.__version__,
        "runtime_seconds": round(elapsed, 3)
    }


# =========================================================================== #
# Plotting Functions
# =========================================================================== #

def create_calibration_plots(df_summary: pd.DataFrame, plots_dir: Path) -> None:
    """Create diagnostic visualizations for the HDBSCAN parameter sweep."""
    matrix_names = ["A_RAW", "A_LOG", "B_RAW", "B_LOG"]
    colors = {"A_RAW": "#1f77b4", "A_LOG": "#ff7f0e", "B_RAW": "#2ca02c", "B_LOG": "#d62728"}

    # ---- Plot 1: Number of Clusters vs min_cluster_size -------------------- #
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, name in enumerate(matrix_names):
        ax = axes[idx]
        sub_mat = df_summary[df_summary["matrix_name"] == name]

        for ms_setting in sub_mat["min_samples_setting"].unique():
            sub_ms = sub_mat[sub_mat["min_samples_setting"] == ms_setting]
            label_str = f"min_samples={ms_setting}"
            ax.plot(
                sub_ms["min_cluster_size"], sub_ms["n_clusters_found"],
                "o-", label=label_str, linewidth=1.8, markersize=6
            )

        ax.set_title(f"{name} — Number of Clusters Found", fontsize=11, fontweight="bold")
        ax.set_xlabel("min_cluster_size")
        ax.set_ylabel("Clusters Found (excl. noise)")
        ax.legend(fontsize=8)

    plt.suptitle(
        "HDBSCAN Calibration Diagnostic 1: Number of Clusters vs min_cluster_size\n"
        "(Evaluated across min_samples choices)",
        fontsize=12, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plots_dir / "hdbscan_diag1_n_clusters_vs_params.png", dpi=300)
    plt.close()

    # ---- Plot 2: Noise Proportion vs min_cluster_size ----------------------- #
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, name in enumerate(matrix_names):
        ax = axes[idx]
        sub_mat = df_summary[df_summary["matrix_name"] == name]

        for ms_setting in sub_mat["min_samples_setting"].unique():
            sub_ms = sub_mat[sub_mat["min_samples_setting"] == ms_setting]
            label_str = f"min_samples={ms_setting}"
            ax.plot(
                sub_ms["min_cluster_size"], sub_ms["noise_proportion"] * 100,
                "s--", label=label_str, linewidth=1.8, markersize=6
            )

        ax.set_title(f"{name} — Noise Percentage (%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("min_cluster_size")
        ax.set_ylabel("Noise Percentage (%)")
        ax.set_ylim(-2, 102)
        ax.legend(fontsize=8)

    plt.suptitle(
        "HDBSCAN Calibration Diagnostic 2: Noise Percentage vs min_cluster_size\n"
        "[High noise indicates diffuse data space; low noise indicates dense aggregation]",
        fontsize=12, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plots_dir / "hdbscan_diag2_noise_proportion_vs_params.png", dpi=300)
    plt.close()

    print("  [PLOTS] Calibration plots 1 & 2 created.")


def create_pca_representative_plots(
    base_dir: Path,
    plots_dir: Path
) -> None:
    """
    Generate PCA 2D post-hoc visualizations for representative HDBSCAN parameter settings.
    A Cohort representative: min_cluster_size=50, min_samples=None
    B Cohort representative: min_cluster_size=25, min_samples=None
    """
    scaled_dir = base_dir / "output" / "scaled_matrices"
    matrices = {
        "A_RAW": (scaled_dir / "A_RAW_scaled.csv", 50, None),
        "A_LOG": (scaled_dir / "A_LOG_scaled.csv", 50, None),
        "B_RAW": (scaled_dir / "B_RAW_scaled.csv", 25, None),
        "B_LOG": (scaled_dir / "B_LOG_scaled.csv", 25, None),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (name, (path, mcs, ms)) in enumerate(matrices.items()):
        ax = axes[idx]
        df_scaled = pd.read_csv(path)
        X = df_scaled.values.astype(np.float64)

        clusterer = HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean", cluster_selection_method="eom")
        labels = clusterer.fit_predict(X)

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        var_exp = pca.explained_variance_ratio_ * 100

        # Noise points in black/gray, clusters in tab10
        is_noise = (labels == -1)
        ax.scatter(
            X_pca[is_noise, 0], X_pca[is_noise, 1],
            c="#aaaaaa", label="Noise (-1)", s=10, alpha=0.4, edgecolors="none"
        )
        if np.sum(~is_noise) > 0:
            scatter = ax.scatter(
                X_pca[~is_noise, 0], X_pca[~is_noise, 1],
                c=labels[~is_noise], cmap="tab10", s=14, alpha=0.7, edgecolors="none"
            )

        n_cl = len(np.unique(labels[~is_noise])) if np.sum(~is_noise) > 0 else 0
        n_ns = np.sum(is_noise)
        pct_ns = (n_ns / len(X)) * 100

        ax.set_title(
            f"{name} (mcs={mcs}, ms={ms or mcs})\n"
            f"{n_cl} Clusters | Noise: {n_ns}/{len(X)} ({pct_ns:.1f}%)",
            fontsize=10, fontweight="bold"
        )
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var)")
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var)")

    plt.suptitle(
        "HDBSCAN Calibration Diagnostic 3: Representative PCA Post-Hoc Projections\n"
        "[Gray = Noise (-1) | Colored = Non-noise clusters | PCA for visualization ONLY]",
        fontsize=11, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plots_dir / "hdbscan_diag4_pca_projections_rep.png", dpi=300)
    plt.close()

    print("  [PLOTS] Representative PCA plot created.")


# =========================================================================== #
# Main Driver
# =========================================================================== #

def run_hdbscan_diagnostics(base_dir: Path) -> None:
    diag_dir = base_dir / "clustering_sandbox" / "hdbscan_diagnostics"
    metrics_dir = diag_dir / "metrics"
    plots_dir = diag_dir / "plots"
    metadata_dir = diag_dir / "metadata"
    for d in [metrics_dir, plots_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    scaled_dir = base_dir / "output" / "scaled_matrices"
    matrices = {
        "A_RAW": (scaled_dir / "A_RAW_scaled.csv", GRID_COHORT_A),
        "A_LOG": (scaled_dir / "A_LOG_scaled.csv", GRID_COHORT_A),
        "B_RAW": (scaled_dir / "B_RAW_scaled.csv", GRID_COHORT_B),
        "B_LOG": (scaled_dir / "B_LOG_scaled.csv", GRID_COHORT_B),
    }

    all_records = []
    matrix_summaries = {}

    print("=" * 70)
    print("  HDBSCAN PARAMETER-CALIBRATION DIAGNOSTIC")
    print("  Read-only. Evaluating min_cluster_size x min_samples grid. No final choices.")
    print("=" * 70)

    t_start = time.time()

    for name, (filepath, grid) in matrices.items():
        df_scaled = pd.read_csv(filepath)
        X = df_scaled.values.astype(np.float64)
        n, p = X.shape
        print(f"\n{'─'*60}")
        print(f"  Matrix: {name} | N={n:,} | p={p}")
        print(f"{'─'*60}")

        mat_records = []
        for mcs in grid["min_cluster_sizes"]:
            for ms in grid["min_samples_list"]:
                rec = evaluate_hdbscan_config(X, name, mcs, ms)
                mat_records.append(rec)
                all_records.append(rec)

                ms_str = f"{ms}" if ms is not None else f"{mcs} (default)"
                print(
                    f"    mcs={mcs:3d} | ms={ms_str:>16s} -> Clusters={rec['n_clusters_found']:2d} | "
                    f"Noise={rec['n_noise_samples']:4d} ({rec['noise_proportion']*100:5.1f}%) | "
                    f"Sil(non-noise)={rec['silhouette_non_noise'] if rec['silhouette_non_noise'] is not None else 'N/A'}"
                )

        df_mat = pd.DataFrame(mat_records)
        df_mat.to_csv(metrics_dir / f"{name}_hdbscan_calibration.csv", index=False)
        matrix_summaries[name] = mat_records

    df_summary = pd.DataFrame(all_records)
    df_summary.to_csv(metrics_dir / "diag1_hdbscan_calibration_summary.csv", index=False)

    print(f"\n{'─'*60}")
    print("  Generating diagnostic plots ...")
    create_calibration_plots(df_summary, plots_dir)
    create_pca_representative_plots(base_dir, plots_dir)

    metadata = {
        "diagnostic": "HDBSCAN_PARAMETER_CALIBRATION_DIAGNOSTIC",
        "hdbscan_implementation": f"sklearn.cluster.HDBSCAN (scikit-learn {sklearn.__version__})",
        "status": "COMPLETED_READ_ONLY",
        "description": "Pre-Experiment 5 parameter calibration diagnostic.",
        "grids_evaluated": {
            "Cohort_A": GRID_COHORT_A,
            "Cohort_B": GRID_COHORT_B
        },
        "total_configurations_tested": len(all_records),
        "total_runtime_seconds": round(time.time() - t_start, 2)
    }

    with open(metadata_dir / "hdbscan_calibration_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  CALIBRATION DIAGNOSTIC COMPLETE")
    print(f"  Total Runtime: {metadata['total_runtime_seconds']:.2f} seconds")
    print(f"  Outputs saved in: clustering_sandbox/hdbscan_diagnostics/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    run_hdbscan_diagnostics(base_dir)
