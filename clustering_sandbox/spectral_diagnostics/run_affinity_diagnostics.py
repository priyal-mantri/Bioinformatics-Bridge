"""
clustering_sandbox/spectral_diagnostics/run_affinity_diagnostics.py
=====================================================================
READ-ONLY Spectral Affinity Calibration Diagnostic — Pre-Experiment 4.

PURPOSE:
  Characterize the empirical distance geometry and kNN graph connectivity
  properties of the four standardized feature matrices, in order to
  establish principled, data-driven hyperparameter reference values for
  the subsequent Spectral Clustering experiment.

THIS SCRIPT DOES NOT:
  - Perform Spectral Clustering.
  - Produce cluster assignments.
  - Select a final K.
  - Choose a final affinity method (RBF vs kNN).
  - Modify the master methodology decision log.
  - Commit anything.

DIAGNOSTICS PERFORMED:
  1. Pairwise squared Euclidean distance distribution (sampled for N=4,482)
  2. RBF reference gamma: gamma_ref = 1 / (2 * median_squared_distance)
     NOTE: At the median distance, this gives affinity exp(-0.5) ≈ 0.607,
     NOT 0.5. The median heuristic sets gamma so that HALF of all pairwise
     distances produce affinity > exp(-0.5) ≈ 0.607.
  3. kNN graph connectivity sweep: k = 5, 7, 10, 12, 15, 20
     Symmetrization: W = (A + A^T) / 2 (averaged, preserving weight scale)
  4. First 10 normalized graph Laplacian eigenvalues and eigengaps at a
     single diagnostic k selected by explicit, documented rule.

SAMPLING STRATEGY FOR COHORT A (N=4,482):
  Full N×N pairwise distance matrix has 4,482×4,481/2 ≈ 10.04 million
  unique pairs, occupying ~80 MB as float64. This is materialized once
  for statistics using scipy.spatial.distance.pdist (condensed form) —
  which is both memory-efficient and exact.
  For the distance percentile statistics, pdist is preferred over sampling
  because it avoids reproducibility questions about the sample draw and
  is feasible in RAM for N=4,482.

GRAPH SYMMETRIZATION RULE (documented once, applied consistently):
  The sklearn NearestNeighbors constructs a directed kNN adjacency matrix A.
  We symmetrize using W = (A + A^T) / 2. This preserves the original
  similarity weights rather than binarizing and is consistent with
  spectral theory requirements for an undirected weighted graph.

METHODOLOGICAL CONSTRAINTS:
  - gamma_ref is reported per matrix; it is NOT shared across matrices.
  - kNN minimum connectivity k is reported per matrix; it is NOT assumed.
  - Eigengap values are treated as evidence about graph partition structure,
    NOT as proof of the biologically correct number of clusters.
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
import scipy.spatial.distance as spdist
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DIAG_K_VALUES = [5, 7, 10, 12, 15, 20]   # kNN sweep values

# For the Laplacian eigenvalue diagnostic, select the SMALLEST k from
# DIAG_K_VALUES that produces a fully connected graph (n_components == 1)
# for that matrix. If no tested k achieves connectivity, use the largest.
# This rule is explicit and does not depend on the clustering result.

N_EIGENVALUES = 10   # Number of Laplacian eigenvalues to compute

# --------------------------------------------------------------------------- #
# Matplotlib styling
# --------------------------------------------------------------------------- #
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
# DIAGNOSTIC 1 — Pairwise squared Euclidean distance distribution
# =========================================================================== #

def compute_distance_distribution(X: np.ndarray, matrix_name: str) -> dict:
    """
    Compute full condensed pairwise squared Euclidean distance distribution
    using scipy.spatial.distance.pdist.

    For N=4,482: N*(N-1)/2 ≈ 10.04M pairs → condensed float64 array ≈ 80 MB.
    For N=967: N*(N-1)/2 ≈ 467K pairs → condensed float64 array ≈ 3.7 MB.
    Both are tractable in RAM without random sampling.
    """
    n_samples, n_features = X.shape
    n_pairs = n_samples * (n_samples - 1) // 2

    print(f"    Computing {n_pairs:,} pairwise squared distances for {matrix_name} ...")
    t0 = time.time()

    # pdist computes condensed pairwise distances; squaring gives squared Euclidean
    dists_flat = spdist.pdist(X, metric="sqeuclidean")

    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.2f}s. Condensed array: {dists_flat.nbytes / 1e6:.1f} MB")

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    pct_values = np.percentile(dists_flat, percentiles)
    pct_dict = {f"p{p}": float(v) for p, v in zip(percentiles, pct_values)}

    result = {
        "matrix_name": matrix_name,
        "N": n_samples,
        "n_features": n_features,
        "n_pairs": int(n_pairs),
        "mean_sq_dist": float(np.mean(dists_flat)),
        "median_sq_dist": float(np.median(dists_flat)),
        "std_sq_dist": float(np.std(dists_flat)),
        "min_sq_dist": float(np.min(dists_flat)),
        "max_sq_dist": float(np.max(dists_flat)),
        "elapsed_seconds": round(elapsed, 2),
    }
    result.update(pct_dict)

    # Theoretical expected squared distance for iid standardized features:
    # E[||x_i - x_j||^2] = 2p (each of p features contributes Var(x_ik - x_jk) = 2)
    result["theoretical_mean_sq_dist_iid"] = float(2 * n_features)

    # RBF reference gamma: gamma_ref = 1 / (2 * median)
    # At median distance: S = exp(-gamma_ref * median) = exp(-0.5) ≈ 0.6065
    # (NOT 0.5 — a common misstatement)
    gamma_ref = 1.0 / (2.0 * result["median_sq_dist"])
    result["gamma_ref"] = float(gamma_ref)

    # RBF affinity values at representative distance percentiles under gamma_ref
    rbf_affinities = {}
    for pct_label in ["p5", "p10", "p25", "p50", "p75", "p90", "p95"]:
        d2 = result[pct_label]
        aff = float(np.exp(-gamma_ref * d2))
        rbf_affinities[pct_label] = round(aff, 6)
    result["rbf_affinities_at_percentiles"] = rbf_affinities

    del dists_flat  # free memory
    return result


# =========================================================================== #
# DIAGNOSTIC 2 — kNN graph connectivity sweep
# =========================================================================== #

def compute_knn_connectivity(
    X: np.ndarray,
    matrix_name: str,
    k_values: list[int]
) -> dict:
    """
    Construct kNN affinity graphs for each k in k_values and report
    graph connectivity statistics.

    Symmetrization: W = (A + A^T) / 2
    This preserves weight scale and is applied consistently across all k.
    """
    n_samples = X.shape[0]
    connectivity_records = []

    for k in k_values:
        t0 = time.time()

        nbrs = NearestNeighbors(
            n_neighbors=k,
            algorithm="auto",
            metric="euclidean",
            n_jobs=1
        )
        nbrs.fit(X)
        A = nbrs.kneighbors_graph(mode="connectivity")  # sparse binary

        # Symmetrize: W = (A + A^T) / 2
        W = (A + A.T) / 2.0

        n_comp, comp_labels = connected_components(
            csgraph=W, directed=False, return_labels=True
        )

        comp_sizes = pd.Series(comp_labels).value_counts().values
        largest = int(comp_sizes.max())
        smallest = int(comp_sizes.min())

        record = {
            "matrix_name": matrix_name,
            "N": n_samples,
            "k": k,
            "n_connected_components": int(n_comp),
            "largest_component_size": largest,
            "largest_component_proportion": round(largest / n_samples, 4),
            "smallest_component_size": smallest,
            "is_fully_connected": bool(n_comp == 1),
            "elapsed_seconds": round(time.time() - t0, 3)
        }
        connectivity_records.append(record)

        status = "CONNECTED" if n_comp == 1 else f"DISCONNECTED ({n_comp} components)"
        print(f"      k={k:2d}: {status} | largest={largest} | smallest={smallest}")

    return {
        "matrix_name": matrix_name,
        "symmetrization_rule": "W = (A + A^T) / 2",
        "k_sweep": connectivity_records
    }


# =========================================================================== #
# DIAGNOSTIC 3 — Normalized Laplacian eigenvalue spectrum
# =========================================================================== #

def select_diagnostic_k(connectivity_result: dict) -> int:
    """
    Select the diagnostic k for eigenvalue computation.
    Rule: Choose the SMALLEST k in DIAG_K_VALUES that produces a fully
    connected graph (n_components == 1) for this matrix.
    If no tested k achieves full connectivity, select the largest k tested
    and document this clearly.
    This rule is deterministic, does not depend on clustering results,
    and does not optimize for any particular eigenvalue profile.
    """
    for rec in connectivity_result["k_sweep"]:
        if rec["is_fully_connected"]:
            return rec["k"]
    # None connected → use largest k
    return connectivity_result["k_sweep"][-1]["k"]


def compute_laplacian_eigenvalues(
    X: np.ndarray,
    matrix_name: str,
    k: int,
    n_eigenvalues: int = 10
) -> dict:
    """
    Build kNN graph with specified k, compute the normalized symmetric
    graph Laplacian, and extract the n_eigenvalues smallest eigenvalues
    and their successive differences (eigengaps).

    Normalized symmetric Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}

    Eigenvalue ordering: 0 = lambda_1 ≤ lambda_2 ≤ ... ≤ lambda_N ≤ 2
    For a fully connected graph, lambda_1 = 0 (trivial eigenvector = const).
    lambda_2 > 0 (algebraic connectivity / Fiedler value).
    """
    t0 = time.time()
    n_samples = X.shape[0]

    # Build kNN graph and symmetrize
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=1)
    nbrs.fit(X)
    A = nbrs.kneighbors_graph(mode="connectivity")
    W = (A + A.T) / 2.0

    # Normalized symmetric Laplacian via scipy
    L_sym = laplacian(W, normed=True)

    # Compute smallest n_eigenvalues + 1 (to compute n_eigenvalues eigengaps)
    n_to_compute = min(n_eigenvalues + 1, n_samples - 2)
    eigenvalues, _ = eigsh(L_sym, k=n_to_compute, which="SM", tol=1e-8)
    eigenvalues = np.sort(np.real(eigenvalues))

    eigenvalues_list = eigenvalues[:n_eigenvalues].tolist()

    # Eigengaps: Delta_i = lambda_{i+1} - lambda_i
    eigengaps = []
    for i in range(len(eigenvalues_list) - 1):
        gap = eigenvalues_list[i + 1] - eigenvalues_list[i]
        eigengaps.append(round(float(gap), 6))

    elapsed = time.time() - t0

    return {
        "matrix_name": matrix_name,
        "diagnostic_k_used": k,
        "k_selection_rule": (
            "Smallest k in sweep [5,7,10,12,15,20] yielding fully connected graph; "
            "if none connected, largest k in sweep."
        ),
        "n_eigenvalues_computed": len(eigenvalues_list),
        "eigenvalues": [round(v, 6) for v in eigenvalues_list],
        "eigengaps": eigengaps,
        "fiedler_value_lambda2": round(float(eigenvalues_list[1]), 6)
                                 if len(eigenvalues_list) > 1 else None,
        "elapsed_seconds": round(elapsed, 2)
    }


# =========================================================================== #
# PLOTS
# =========================================================================== #

def create_diagnostic_plots(
    dist_results: list[dict],
    conn_results: list[dict],
    eigen_results: list[dict],
    plots_dir: Path
) -> None:
    """
    Create three sets of diagnostic plots:
      1. Pairwise squared distance distribution summaries (box-plot style)
      2. kNN connectivity vs k (number of components)
      3. Laplacian eigenvalue spectrum and eigengaps
    """
    matrix_order = ["A_RAW", "A_LOG", "B_RAW", "B_LOG"]
    colors = {"A_RAW": "#1f77b4", "A_LOG": "#ff7f0e",
              "B_RAW": "#2ca02c", "B_LOG": "#d62728"}

    # ---- Plot 1: Distance distribution summary ----------------------------- #
    fig, ax = plt.subplots(figsize=(10, 5))
    pcts = ["p5", "p25", "p50", "p75", "p95"]
    x = np.arange(len(pcts))
    width = 0.18

    for i, name in enumerate(matrix_order):
        dr = next(d for d in dist_results if d["matrix_name"] == name)
        vals = [dr[p] for p in pcts]
        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               color=colors[name], label=name, alpha=0.85, edgecolor="black",
               linewidth=0.5)

    ax.axhline(y=38.0, color="#1f77b4", linestyle="--", linewidth=0.8,
               alpha=0.5, label="Theoretical iid E[d²]=2p (A, p=19)")
    ax.axhline(y=48.0, color="#2ca02c", linestyle="--", linewidth=0.8,
               alpha=0.5, label="Theoretical iid E[d²]=2p (B, p=24)")

    ax.set_xticks(x)
    ax.set_xticklabels(["5th pct", "25th pct", "Median", "75th pct", "95th pct"])
    ax.set_title(
        "Pairwise Squared Euclidean Distance Percentiles\n"
        "All pairs computed (exact, no sampling) | "
        "Dashes = theoretical iid expected distance",
        fontsize=10, fontweight="bold"
    )
    ax.set_ylabel("Squared Euclidean Distance")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plots_dir / "diag1_distance_distribution.png", dpi=300)
    plt.close()

    # ---- Plot 2: kNN connectivity vs k ------------------------------------- #
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in matrix_order:
        cr = next(c for c in conn_results if c["matrix_name"] == name)
        ks = [r["k"] for r in cr["k_sweep"]]
        n_comps = [r["n_connected_components"] for r in cr["k_sweep"]]
        ax.plot(ks, n_comps, "o-", color=colors[name], linewidth=2,
                markersize=7, label=name)

    ax.axhline(y=1, color="black", linestyle="--", linewidth=1.2,
               alpha=0.8, label="Fully connected threshold (1 component)")
    ax.set_xlabel("k (number of nearest neighbors)")
    ax.set_ylabel("Number of Connected Components")
    ax.set_title(
        "kNN Graph Connectivity vs k\n"
        "Symmetrization: W = (A + A^T) / 2",
        fontsize=10, fontweight="bold"
    )
    ax.set_xticks(DIAG_K_VALUES)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "diag2_knn_connectivity.png", dpi=300)
    plt.close()

    # ---- Plot 3: Laplacian eigenvalue spectrum and eigengaps --------------- #
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for col, name in enumerate(matrix_order):
        er = next(e for e in eigen_results if e["matrix_name"] == name)
        evals = er["eigenvalues"]
        egaps = er["eigengaps"]
        idx = list(range(1, len(evals) + 1))

        # Top row: eigenvalues
        ax_top = axes[0, col]
        ax_top.plot(idx, evals, "o-", color=colors[name], linewidth=2, markersize=6)
        ax_top.set_title(
            f"{name}\nNorm. Laplacian Eigenvalues\n"
            f"(k={er['diagnostic_k_used']}, λ₂={er['fiedler_value_lambda2']:.4f})",
            fontsize=9, fontweight="bold"
        )
        ax_top.set_xlabel("Eigenvalue Index")
        ax_top.set_ylabel("λ")
        ax_top.set_xticks(idx)

        # Bottom row: eigengaps
        ax_bot = axes[1, col]
        gap_idx = list(range(1, len(egaps) + 1))
        ax_bot.bar(gap_idx, egaps, color=colors[name], alpha=0.8,
                   edgecolor="black", linewidth=0.5)
        ax_bot.set_title(
            f"{name}\nEigengaps Δλ_i = λ_{{i+1}} - λ_i",
            fontsize=9, fontweight="bold"
        )
        ax_bot.set_xlabel("i (gap between λᵢ and λᵢ₊₁)")
        ax_bot.set_ylabel("Eigengap")
        ax_bot.set_xticks(gap_idx)

    plt.suptitle(
        "Normalized Graph Laplacian Eigenvalue Spectrum & Eigengaps\n"
        "[DIAGNOSTIC ONLY — These are evidence about graph structure, "
        "NOT proof of the biologically correct number of clusters]",
        fontsize=10, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "diag3_laplacian_eigenvalues.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    print("  [PLOTS] All 3 diagnostic plots saved.")


# =========================================================================== #
# REPORT WRITER
# =========================================================================== #

def write_csv_reports(
    dist_results: list[dict],
    conn_results: list[dict],
    eigen_results: list[dict],
    metrics_dir: Path
) -> None:
    """Write structured CSV summary tables for each diagnostic."""

    # CSV 1: Distance distribution summary
    rows = []
    for dr in dist_results:
        rows.append({
            "matrix": dr["matrix_name"],
            "N": dr["N"],
            "n_features": dr["n_features"],
            "n_pairs": dr["n_pairs"],
            "mean_sq_dist": round(dr["mean_sq_dist"], 4),
            "median_sq_dist": round(dr["median_sq_dist"], 4),
            "std_sq_dist": round(dr["std_sq_dist"], 4),
            "min_sq_dist": round(dr["min_sq_dist"], 4),
            "max_sq_dist": round(dr["max_sq_dist"], 4),
            "p5_sq_dist": round(dr["p5"], 4),
            "p10_sq_dist": round(dr["p10"], 4),
            "p25_sq_dist": round(dr["p25"], 4),
            "p50_sq_dist_median": round(dr["p50"], 4),
            "p75_sq_dist": round(dr["p75"], 4),
            "p90_sq_dist": round(dr["p90"], 4),
            "p95_sq_dist": round(dr["p95"], 4),
            "theoretical_iid_mean": dr["theoretical_mean_sq_dist_iid"],
            "gamma_ref": round(dr["gamma_ref"], 6),
            "rbf_at_p5": dr["rbf_affinities_at_percentiles"]["p5"],
            "rbf_at_p25": dr["rbf_affinities_at_percentiles"]["p25"],
            "rbf_at_p50_median": dr["rbf_affinities_at_percentiles"]["p50"],
            "rbf_at_p75": dr["rbf_affinities_at_percentiles"]["p75"],
            "rbf_at_p95": dr["rbf_affinities_at_percentiles"]["p95"],
        })
    pd.DataFrame(rows).to_csv(
        metrics_dir / "diag1_distance_distribution.csv", index=False
    )

    # CSV 2: kNN connectivity
    rows2 = []
    for cr in conn_results:
        for rec in cr["k_sweep"]:
            rows2.append({
                "matrix": rec["matrix_name"],
                "N": rec["N"],
                "k": rec["k"],
                "n_connected_components": rec["n_connected_components"],
                "largest_component_size": rec["largest_component_size"],
                "largest_component_proportion": rec["largest_component_proportion"],
                "smallest_component_size": rec["smallest_component_size"],
                "is_fully_connected": rec["is_fully_connected"],
                "symmetrization_rule": cr["symmetrization_rule"],
            })
    pd.DataFrame(rows2).to_csv(
        metrics_dir / "diag2_knn_connectivity.csv", index=False
    )

    # CSV 3: Laplacian eigenvalues
    rows3 = []
    for er in eigen_results:
        for i, (ev, _) in enumerate(
            zip(er["eigenvalues"],
                er["eigengaps"] + [None])
        ):
            rows3.append({
                "matrix": er["matrix_name"],
                "diagnostic_k": er["diagnostic_k_used"],
                "eigenvalue_index": i + 1,
                "eigenvalue": ev,
                "eigengap_to_next": er["eigengaps"][i]
                                   if i < len(er["eigengaps"]) else None,
                "k_selection_rule": er["k_selection_rule"],
            })
    pd.DataFrame(rows3).to_csv(
        metrics_dir / "diag3_laplacian_eigenvalues.csv", index=False
    )

    print("  [REPORTS] All 3 CSV diagnostic tables saved.")


# =========================================================================== #
# MAIN PIPELINE
# =========================================================================== #

def run_affinity_diagnostics(base_dir: Path) -> None:
    diag_dir = base_dir / "clustering_sandbox" / "spectral_diagnostics"
    metrics_dir = diag_dir / "metrics"
    plots_dir = diag_dir / "plots"
    metadata_dir = diag_dir / "metadata"
    for d in [metrics_dir, plots_dir, metadata_dir]:
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

    dist_results = []
    conn_results = []
    eigen_results = []

    print("=" * 70)
    print("  SPECTRAL AFFINITY CALIBRATION DIAGNOSTIC")
    print("  Read-only. No clustering. No assignments. No final decisions.")
    print("=" * 70)

    t_global = time.time()

    for name, (scaled_file, _) in matrices.items():
        df_scaled = pd.read_csv(scaled_file)
        X = df_scaled.values.astype(np.float64)
        n, p = X.shape
        print(f"\n{'─'*60}")
        print(f"  Matrix: {name} | N={n:,} | p={p}")
        print(f"{'─'*60}")

        # ---- Diagnostic 1: Distance distribution ---- #
        print("  [D1] Pairwise distance distribution ...")
        dr = compute_distance_distribution(X, name)
        dist_results.append(dr)
        print(f"    Median d²  = {dr['median_sq_dist']:.4f}")
        print(f"    Theor. E[d²] = {dr['theoretical_mean_sq_dist_iid']:.1f}")
        print(f"    gamma_ref  = {dr['gamma_ref']:.6f}")
        print(f"    RBF affinities under gamma_ref:")
        for pct_label, aff in dr["rbf_affinities_at_percentiles"].items():
            print(f"      {pct_label}: {aff:.4f}")

        # ---- Diagnostic 2: kNN connectivity ---- #
        print(f"\n  [D2] kNN connectivity sweep k={DIAG_K_VALUES} ...")
        cr = compute_knn_connectivity(X, name, DIAG_K_VALUES)
        conn_results.append(cr)

        # ---- Diagnostic 3: Laplacian eigenvalues ---- #
        diag_k = select_diagnostic_k(cr)
        if diag_k == cr["k_sweep"][-1]["k"] and not cr["k_sweep"][-1]["is_fully_connected"]:
            print(f"\n  [D3] WARNING: No tested k achieved full connectivity for {name}.")
            print(f"       Using largest k={diag_k} for Laplacian diagnostic.")
        else:
            print(f"\n  [D3] Laplacian eigenvalues at k={diag_k} (smallest connected k) ...")
        er = compute_laplacian_eigenvalues(X, name, k=diag_k, n_eigenvalues=N_EIGENVALUES)
        eigen_results.append(er)
        print(f"    Eigenvalues: {[round(v, 4) for v in er['eigenvalues']]}")
        print(f"    Eigengaps:   {er['eigengaps']}")
        print(f"    Fiedler (λ₂): {er['fiedler_value_lambda2']}")

    # ---- Generate plots ---- #
    print(f"\n{'─'*60}")
    print("  Generating diagnostic plots ...")
    create_diagnostic_plots(dist_results, conn_results, eigen_results, plots_dir)

    # ---- Save CSV reports ---- #
    print("  Writing CSV reports ...")
    write_csv_reports(dist_results, conn_results, eigen_results, metrics_dir)

    # ---- Save metadata JSON ---- #
    metadata = {
        "diagnostic_name": "SPECTRAL_AFFINITY_CALIBRATION_DIAGNOSTIC",
        "status": "READ_ONLY_DIAGNOSTIC",
        "description": (
            "Pre-Experiment 4 affinity calibration. "
            "No clustering performed. No assignments created. "
            "No final affinity method or hyperparameter selected."
        ),
        "sampling_strategy": (
            "Full exact pairwise distance computation via scipy.spatial.distance.pdist "
            "(condensed form). No random sampling used. "
            "For N=4,482: ~10.04M pairs, ~80MB condensed float64."
        ),
        "symmetrization_rule": "W = (A + A^T) / 2",
        "gamma_note": (
            "gamma_ref = 1 / (2 * median_sq_dist). "
            "Under this gamma, pairs at the MEDIAN distance receive affinity "
            "exp(-0.5) ≈ 0.6065. This does NOT set the median-affinity to 0.5."
        ),
        "eigengap_note": (
            "Eigengaps are evidence about graph Laplacian structure, "
            "NOT proof of the biologically correct number of clusters."
        ),
        "k_selection_rule": (
            "Smallest k in [5,7,10,12,15,20] yielding fully connected graph. "
            "If no k achieves connectivity, the largest k is used."
        ),
        "results_summary": {
            name: {
                "gamma_ref": next(d for d in dist_results if d["matrix_name"] == name)["gamma_ref"],
                "median_sq_dist": next(d for d in dist_results if d["matrix_name"] == name)["median_sq_dist"],
                "diagnostic_k_for_eigenvalues": next(e for e in eigen_results if e["matrix_name"] == name)["diagnostic_k_used"],
                "fiedler_value": next(e for e in eigen_results if e["matrix_name"] == name)["fiedler_value_lambda2"],
                "connectivity_results": {
                    f"k={r['k']}": r["n_connected_components"]
                    for r in next(c for c in conn_results if c["matrix_name"] == name)["k_sweep"]
                }
            }
            for name in ["A_RAW", "A_LOG", "B_RAW", "B_LOG"]
        }
    }

    meta_file = metadata_dir / "affinity_diagnostics_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    total_elapsed = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC COMPLETE")
    print(f"  Total runtime: {total_elapsed:.1f} seconds")
    print(f"  Outputs in: clustering_sandbox/spectral_diagnostics/")
    print(f"    metrics/diag1_distance_distribution.csv")
    print(f"    metrics/diag2_knn_connectivity.csv")
    print(f"    metrics/diag3_laplacian_eigenvalues.csv")
    print(f"    plots/diag1_distance_distribution.png")
    print(f"    plots/diag2_knn_connectivity.png")
    print(f"    plots/diag3_laplacian_eigenvalues.png")
    print(f"    metadata/affinity_diagnostics_metadata.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    run_affinity_diagnostics(base_dir)
