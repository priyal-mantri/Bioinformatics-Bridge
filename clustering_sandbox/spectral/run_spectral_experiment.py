"""
clustering_sandbox/spectral/run_spectral_experiment.py
======================================================
Experiment 4: Spectral Clustering Exploratory Sandbox (kNN Affinity)

PURPOSE:
  Evaluate the structure found by Spectral Clustering operating in the full
  standardized feature space across four NHANES 2017-2018 candidate matrices
  (A_RAW, A_LOG, B_RAW, B_LOG) for cluster counts K = 2, 3, 4, 5, 6, 7.

FIXED METHODOLOGY (LOCKED):
  - Affinity Construction: k-Nearest Neighbor (kNN) graph
  - Fixed Graph Construction Parameter: k_neighbors = 10
  - Symmetrization Rule: W = (A + A^T) / 2 where A is the binary directed kNN matrix.
  - Laplacian: Normalized symmetric Laplacian L_sym = I - D^{-1/2} W D^{-1/2}.
  - Spectral Embedding: Ng-Jordan-Weiss (NJW) row-normalized L2 embedding.
  - Final Partitioning: KMeans(n_clusters=K, random_state=42, n_init=25) on spectral embedding.
  - Feature Space: Full standardized feature space (19D for A, 24D for B).
  - PCA Role: PCA is used ONLY for 2D post-hoc visualization (Decision 013).

EFFICIENCY OPTIMIZATION:
  - kNN graph construction and normalized Laplacian eigendecomposition (first 8 eigenvectors)
    are performed EXACTLY ONCE per candidate matrix.
  - The K=2..7 sweep is performed by slicing the precomputed spectral embedding rows
    and running K-Means. This avoids redundant rebuilds of the graph or eigensolver.
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
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.decomposition import PCA
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Experiment Settings
# --------------------------------------------------------------------------- #
K_SWEEP = [2, 3, 4, 5, 6, 7]
K_NEIGHBORS = 10
RANDOM_STATE = 42
N_INIT = 25

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

def build_spectral_embedding(X: np.ndarray, k_neighbors: int = 10, n_components: int = 8):
    """
    Construct kNN graph, symmetrize via W = (A + A^T) / 2, build normalized
    symmetric Laplacian L_sym, and compute the smallest n_components eigenvectors.

    Returns:
        eigenvalues: np.ndarray of shape (n_components,)
        eigenvectors: np.ndarray of shape (N, n_components)
        is_connected: bool
        n_components_graph: int
    """
    n_samples = X.shape[0]

    # kNN graph (directed binary connectivity, self-neighbors excluded)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm="auto", metric="euclidean", n_jobs=1)
    nbrs.fit(X)
    A = nbrs.kneighbors_graph(mode="connectivity")  # sparse binary

    # Symmetrization
    W = (A + A.T) / 2.0

    # Connectivity check
    n_comp_graph, labels = connected_components(csgraph=W, directed=False, return_labels=True)
    is_connected = bool(n_comp_graph == 1)

    # Normalized symmetric Laplacian L_sym = I - D^{-1/2} W D^{-1/2}
    L_sym = laplacian(W, normed=True)

    # Compute smallest n_components eigenvalues/eigenvectors
    n_to_compute = min(n_components, n_samples - 2)
    eigenvalues, eigenvectors = eigsh(L_sym, k=n_to_compute, which="SM", tol=1e-8)

    # Sort in ascending eigenvalue order
    idx = np.argsort(np.real(eigenvalues))
    eigenvalues = np.real(eigenvalues[idx])
    eigenvectors = np.real(eigenvectors[:, idx])

    return eigenvalues, eigenvectors, is_connected, int(n_comp_graph)


def run_spectral_sweep_for_matrix(
    df_scaled: pd.DataFrame,
    df_cohort: pd.DataFrame,
    matrix_name: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Perform Spectral Clustering K=2..7 sweep for a single candidate matrix.
    """
    t0 = time.time()
    X = df_scaled.values.astype(np.float64)
    n_samples, n_features = X.shape
    seqn = df_cohort["SEQN"].values if "SEQN" in df_cohort.columns else np.arange(n_samples)

    print(f"\n{'─'*60}")
    print(f"  Running Spectral Clustering for Matrix: {matrix_name}")
    print(f"  N={n_samples:,} participants, p={n_features} features, k_neighbors={K_NEIGHBORS}")
    print(f"{'─'*60}")

    # Build spectral embedding components once
    max_k_components = max(K_SWEEP) + 1  # top 8 eigenvectors for K=7
    eigenvalues, eigenvectors, is_connected, n_graph_comp = build_spectral_embedding(
        X, k_neighbors=K_NEIGHBORS, n_components=max_k_components
    )

    print(f"  Graph connectivity check: {n_graph_comp} connected component(s) -> Fully Connected: {is_connected}")
    print(f"  Laplacian Eigenvalues (λ₁..λ₈): {[round(float(v), 5) for v in eigenvalues[:8]]}")
    fiedler_val = float(eigenvalues[1])
    print(f"  Fiedler value (λ₂): {fiedler_val:.5f}")

    metrics_list = []
    assignments_dict = {"SEQN": seqn}

    for K in K_SWEEP:
        t_k0 = time.time()

        # Extract top K eigenvectors
        U_K = eigenvectors[:, :K]

        # Ng-Jordan-Weiss row normalization (L2 norm per row)
        row_norms = np.linalg.norm(U_K, axis=1, keepdims=True)
        # Avoid division by zero
        row_norms[row_norms == 0] = 1e-12
        Y_K = U_K / row_norms

        # KMeans partitioning on normalized spectral embedding
        km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = km.fit_predict(Y_K)
        inertia_embedding = float(km.inertia_)

        # Internal clustering metrics in FULL standardized feature space X
        sil = float(silhouette_score(X, labels))
        ch = float(calinski_harabasz_score(X, labels))
        db = float(davies_bouldin_score(X, labels))

        # Cluster size distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        counts_dict = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)}
        proportions = [round(float(cnt / n_samples), 4) for cnt in counts]
        min_sz = int(np.min(counts))
        max_sz = int(np.max(counts))
        min_prop = float(min_sz / n_samples)
        max_prop = float(max_sz / n_samples)
        ratio_max_min = float(max_sz / min_sz) if min_sz > 0 else np.nan

        # Eigengap at K (gap between lambda_K and lambda_{K-1}) and gap to next (lambda_{K+1} - lambda_K)
        # Note: 1-indexed K corresponds to 0-indexed eigenvector indices
        gap_prev = float(eigenvalues[K-1] - eigenvalues[K-2]) if K >= 2 else np.nan
        gap_next = float(eigenvalues[K] - eigenvalues[K-1]) if K < len(eigenvalues) else np.nan

        metric_rec = {
            "matrix_name": matrix_name,
            "K": K,
            "n_samples": n_samples,
            "n_features": n_features,
            "k_neighbors": K_NEIGHBORS,
            "silhouette_score": sil,
            "calinski_harabasz_score": ch,
            "davies_bouldin_score": db,
            "embedding_kmeans_inertia": inertia_embedding,
            "cluster_sizes": json.dumps(counts_dict),
            "cluster_proportions": json.dumps(proportions),
            "min_cluster_size": min_sz,
            "max_cluster_size": max_sz,
            "min_cluster_proportion": round(min_prop, 4),
            "max_cluster_proportion": round(max_prop, 4),
            "max_min_cluster_ratio": round(ratio_max_min, 4),
            "fiedler_lambda2": round(fiedler_val, 6),
            "eigengap_K_minus_1_to_K": round(gap_prev, 6),
            "eigengap_K_to_K_plus_1": round(gap_next, 6),
            "random_seed": RANDOM_STATE,
            "n_init": N_INIT,
            "sklearn_version": sklearn.__version__,
            "runtime_k_seconds": round(time.time() - t_k0, 3)
        }
        metrics_list.append(metric_rec)
        assignments_dict[f"K{K}_cluster"] = labels

        print(
            f"    K={K}: Silhouette={sil:.4f} | CH={ch:.1f} | DB={db:.3f} | "
            f"Proportions={[round(p, 3) for p in proportions]} | Min Size={min_sz}"
        )

    df_metrics = pd.DataFrame(metrics_list)
    df_assignments = pd.DataFrame(assignments_dict)

    matrix_meta = {
        "matrix_name": matrix_name,
        "n_samples": n_samples,
        "n_features": n_features,
        "k_neighbors": K_NEIGHBORS,
        "is_fully_connected": is_connected,
        "n_graph_components": n_graph_comp,
        "fiedler_value_lambda2": round(fiedler_val, 6),
        "laplacian_eigenvalues_top8": [round(float(v), 6) for v in eigenvalues[:8]],
        "laplacian_eigengaps_top7": [round(float(eigenvalues[i+1] - eigenvalues[i]), 6) for i in range(7)],
        "runtime_total_seconds": round(time.time() - t0, 2)
    }

    return df_metrics, df_assignments, matrix_meta


# =========================================================================== #
# Visualization Functions
# =========================================================================== #

def plot_metrics_vs_k(df_metrics_all: pd.DataFrame, plots_dir: Path) -> None:
    """Plot Sil, CH, DB, and Min Cluster Proportion vs K across matrices."""
    matrix_names = ["A_RAW", "A_LOG", "B_RAW", "B_LOG"]
    colors = {"A_RAW": "#1f77b4", "A_LOG": "#ff7f0e", "B_RAW": "#2ca02c", "B_LOG": "#d62728"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Silhouette Score (higher is better)
    ax = axes[0, 0]
    for name in matrix_names:
        sub = df_metrics_all[df_metrics_all["matrix_name"] == name]
        ax.plot(sub["K"], sub["silhouette_score"], "o-", color=colors[name], label=name, linewidth=2)
    ax.set_title("Silhouette Score vs K (Higher is better)", fontsize=11, fontweight="bold")
    ax.set_xlabel("K (Cluster Count)")
    ax.set_ylabel("Silhouette Score")
    ax.set_xticks(K_SWEEP)
    ax.legend()

    # Panel 2: Calinski-Harabasz Index (higher is better)
    ax = axes[0, 1]
    for name in matrix_names:
        sub = df_metrics_all[df_metrics_all["matrix_name"] == name]
        ax.plot(sub["K"], sub["calinski_harabasz_score"], "o-", color=colors[name], label=name, linewidth=2)
    ax.set_title("Calinski-Harabasz Index vs K (Higher is better)", fontsize=11, fontweight="bold")
    ax.set_xlabel("K (Cluster Count)")
    ax.set_ylabel("Calinski-Harabasz Score")
    ax.set_xticks(K_SWEEP)
    ax.legend()

    # Panel 3: Davies-Bouldin Index (lower is better)
    ax = axes[1, 0]
    for name in matrix_names:
        sub = df_metrics_all[df_metrics_all["matrix_name"] == name]
        ax.plot(sub["K"], sub["davies_bouldin_score"], "o-", color=colors[name], label=name, linewidth=2)
    ax.set_title("Davies-Bouldin Index vs K (Lower is better)", fontsize=11, fontweight="bold")
    ax.set_xlabel("K (Cluster Count)")
    ax.set_ylabel("Davies-Bouldin Score")
    ax.set_xticks(K_SWEEP)
    ax.legend()

    # Panel 4: Min Cluster Proportion vs K
    ax = axes[1, 1]
    for name in matrix_names:
        sub = df_metrics_all[df_metrics_all["matrix_name"] == name]
        ax.plot(sub["K"], sub["min_cluster_proportion"], "s--", color=colors[name], label=name, linewidth=2)
    ax.axhline(y=0.01, color="red", linestyle=":", label="1% Threshold")
    ax.set_title("Minimum Cluster Proportion vs K", fontsize=11, fontweight="bold")
    ax.set_xlabel("K (Cluster Count)")
    ax.set_ylabel("Min Cluster Proportion")
    ax.set_xticks(K_SWEEP)
    ax.legend()

    plt.suptitle(
        "Experiment 4: Spectral Clustering Evaluation Metrics across K=2..7\n"
        "(Fixed k_neighbors=10, Full Standardized Feature Space)",
        fontsize=12, fontweight="bold", y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plots_dir / "spectral_metrics_vs_k.png", dpi=300)
    plt.close()


def plot_pca_projections_for_matrix(
    df_scaled: pd.DataFrame,
    df_assignments: pd.DataFrame,
    matrix_name: str,
    plots_dir: Path
) -> None:
    """
    Project full standardized feature space onto top 2 PCs for 2D post-hoc visualization ONLY.
    """
    X = df_scaled.values
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, K in enumerate(K_SWEEP):
        ax = axes[idx]
        labels = df_assignments[f"K{K}_cluster"].values
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=labels, cmap="tab10", s=8, alpha=0.6, edgecolors="none"
        )
        ax.set_title(f"{matrix_name} — K={K}", fontsize=10, fontweight="bold")
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% var)")
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% var)")

    plt.suptitle(
        f"Spectral Clustering Assignments Visualized on PCA Projection ({matrix_name})\n"
        "[POST-HOC VISUALIZATION ONLY — Clustering was run in full standardized feature space]",
        fontsize=11, fontweight="bold", y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plots_dir / f"spectral_pca_projections_{matrix_name}.png", dpi=300)
    plt.close()


# =========================================================================== #
# Main Driver
# =========================================================================== #

def run_spectral_experiment(base_dir: Path) -> None:
    exp_dir = base_dir / "clustering_sandbox" / "spectral"
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
        "A_RAW": (scaled_dir / "A_RAW_scaled.csv", df_cohort_a),
        "A_LOG": (scaled_dir / "A_LOG_scaled.csv", df_cohort_a),
        "B_RAW": (scaled_dir / "B_RAW_scaled.csv", df_cohort_b),
        "B_LOG": (scaled_dir / "B_LOG_scaled.csv", df_cohort_b),
    }

    metrics_all_list = []
    metadata_summary = {
        "experiment": "EXPERIMENT_4_SPECTRAL_CLUSTERING_SANDBOX",
        "fixed_affinity_method": "k-Nearest-Neighbors (kNN)",
        "fixed_k_neighbors": K_NEIGHBORS,
        "symmetrization_rule": "W = (A + A^T) / 2",
        "spectral_embedding": "Ng-Jordan-Weiss (NJW) row L2 normalized eigenvectors",
        "partitioning_algorithm": "KMeans(random_state=42, n_init=25)",
        "k_clusters_sweep": K_SWEEP,
        "matrices_evaluated": list(matrices.keys()),
        "matrices_metadata": {}
    }

    t_start = time.time()

    for matrix_name, (scaled_path, df_cohort) in matrices.items():
        df_scaled = pd.read_csv(scaled_path)
        df_metrics, df_assignments, mat_meta = run_spectral_sweep_for_matrix(
            df_scaled, df_cohort, matrix_name
        )

        metrics_all_list.append(df_metrics)
        metadata_summary["matrices_metadata"][matrix_name] = mat_meta

        # Save individual metrics CSV
        df_metrics.to_csv(metrics_dir / f"{matrix_name}_spectral_metrics.csv", index=False)

        # Save cluster assignments CSV
        df_assignments.to_csv(assignments_dir / f"{matrix_name}_spectral_assignments.csv", index=False)

        # Generate PCA post-hoc plot
        plot_pca_projections_for_matrix(df_scaled, df_assignments, matrix_name, plots_dir)

    # Combine all metrics into cross-matrix summary
    df_metrics_cross = pd.concat(metrics_all_list, ignore_index=True)
    df_metrics_cross.to_csv(metrics_dir / "spectral_cross_matrix_summary.csv", index=False)

    # Create combined metric plots
    plot_metrics_vs_k(df_metrics_cross, plots_dir)

    metadata_summary["total_runtime_seconds"] = round(time.time() - t_start, 2)

    with open(metadata_dir / "spectral_experiment_metadata.json", "w") as f:
        json.dump(metadata_summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 4 COMPLETE")
    print(f"  Total Runtime: {metadata_summary['total_runtime_seconds']:.2f}s")
    print(f"  Outputs saved in: clustering_sandbox/spectral/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    run_spectral_experiment(base_dir)
