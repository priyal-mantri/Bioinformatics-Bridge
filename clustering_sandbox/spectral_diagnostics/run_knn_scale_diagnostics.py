"""
clustering_sandbox/spectral_diagnostics/run_knn_scale_diagnostics.py
======================================================================
READ-ONLY Spectral kNN Neighborhood Scale Sensitivity Diagnostic.

PURPOSE:
  Evaluate the stability of the normalized graph Laplacian eigenvalue spectrum
  and eigengaps across varying neighborhood scales (k = 5, 10, 15, 20)
  for all four standardized feature matrices.

THIS SCRIPT DOES NOT:
  - Perform Spectral Clustering sweeps (K=2..7).
  - Produce cluster assignments.
  - Choose a final k or final affinity method.
  - Modify the master methodology decision log.
  - Commit anything.

METHODOLOGY:
  - Graph construction: kNN with k in {5, 10, 15, 20}.
  - Symmetrization rule: W = (A + A^T) / 2.
  - Normalized symmetric Laplacian: L_sym = I - D^{-1/2} W D^{-1/2}.
  - Compute first 10 eigenvalues via scipy.sparse.linalg.eigsh.
  - Compute eigengaps: Delta_i = lambda_{i+1} - lambda_i.
  - Identify location and magnitude of top sub-Fiedler eigengaps.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

K_SCALE_VALUES = [5, 10, 15, 20]
N_EIGENVALUES = 10

plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#444444"
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["figure.dpi"] = 300


def compute_laplacian_spectrum_for_k(
    X: np.ndarray,
    matrix_name: str,
    k: int,
    n_eigenvalues: int = 10
) -> dict:
    t0 = time.time()
    n_samples = X.shape[0]

    # kNN graph construction
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=1)
    nbrs.fit(X)
    A = nbrs.kneighbors_graph(mode="connectivity")
    W = (A + A.T) / 2.0

    # Normalized symmetric Laplacian
    L_sym = laplacian(W, normed=True)

    # Compute smallest n_eigenvalues + 1 eigenvalues
    n_to_compute = min(n_eigenvalues + 1, n_samples - 2)
    eigenvalues, _ = eigsh(L_sym, k=n_to_compute, which="SM", tol=1e-8)
    eigenvalues = np.sort(np.real(eigenvalues))

    eigenvalues_list = eigenvalues[:n_eigenvalues].tolist()

    eigengaps = []
    for i in range(len(eigenvalues_list) - 1):
        gap = eigenvalues_list[i + 1] - eigenvalues_list[i]
        eigengaps.append(round(float(gap), 6))

    # Identify largest sub-Fiedler eigengap (from position 2->3 onwards)
    sub_fiedler_gaps = eigengaps[1:] # gap 2->3 is index 0 in sub_fiedler_gaps (position i=2)
    # Position index i: gap i->i+1. eigengaps[0] is gap 1->2 (Fiedler gap).
    # eigengaps[1] is gap 2->3, etc.
    max_sub_fiedler_gap = max(sub_fiedler_gaps)
    max_sub_fiedler_pos = sub_fiedler_gaps.index(max_sub_fiedler_gap) + 2 # position i

    elapsed = time.time() - t0

    return {
        "matrix_name": matrix_name,
        "k": k,
        "n_samples": n_samples,
        "eigenvalues": [round(float(v), 6) for v in eigenvalues_list],
        "eigengaps": eigengaps,
        "fiedler_lambda2": round(float(eigenvalues_list[1]), 6),
        "fiedler_gap_1_to_2": eigengaps[0],
        "max_sub_fiedler_gap_mag": round(float(max_sub_fiedler_gap), 6),
        "max_sub_fiedler_gap_pos": f"{max_sub_fiedler_pos}->{max_sub_fiedler_pos+1}",
        "elapsed_seconds": round(elapsed, 2)
    }


def create_scale_plots(results_by_matrix: dict, plots_dir: Path) -> None:
    matrix_order = ["A_RAW", "A_LOG", "B_RAW", "B_LOG"]
    k_colors = {5: "#1f77b4", 10: "#ff7f0e", 15: "#2ca02c", 20: "#d62728"}

    # 4-panel plot: Eigengap profiles across k for each matrix
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, name in enumerate(matrix_order):
        ax = axes[idx]
        mat_results = results_by_matrix[name]

        gap_labels = [f"{i}→{i+1}" for i in range(1, 10)]
        x = np.arange(len(gap_labels))
        width = 0.2

        for i_k, k in enumerate(K_SCALE_VALUES):
            res = next(r for r in mat_results if r["k"] == k)
            gaps = res["eigengaps"]
            offset = (i_k - 1.5) * width
            ax.bar(x + offset, gaps, width=width*0.9, color=k_colors[k],
                   label=f"k={k}", alpha=0.85, edgecolor="black", linewidth=0.5)

        ax.set_title(f"Matrix: {name} (N={mat_results[0]['n_samples']})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Eigengap Position (i → i+1)")
        ax.set_ylabel("Eigengap Magnitude (Δλᵢ)")
        ax.set_xticks(x)
        ax.set_xticklabels(gap_labels, rotation=45)
        ax.legend(title="k_neighbors", fontsize=9)

    plt.suptitle(
        "Normalized Graph Laplacian Eigengap Profiles Across Neighborhood Scales (k ∈ {5, 10, 15, 20})\n"
        "[DIAGNOSTIC ONLY — Evidence of spectral stability across scale, NOT proof of cluster count]",
        fontsize=11, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plots_dir / "diag4_knn_scale_eigengaps.png", dpi=300)
    plt.close()

    # 4-panel plot: Eigenvalue spectra across k for each matrix
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, name in enumerate(matrix_order):
        ax = axes[idx]
        mat_results = results_by_matrix[name]
        ev_indices = list(range(1, N_EIGENVALUES + 1))

        for k in K_SCALE_VALUES:
            res = next(r for r in mat_results if r["k"] == k)
            evals = res["eigenvalues"]
            ax.plot(ev_indices, evals, "o-", color=k_colors[k], linewidth=1.8,
                    markersize=5, label=f"k={k} (λ₂={res['fiedler_lambda2']:.4f})")

        ax.set_title(f"Matrix: {name} — Eigenvalue Spectrum", fontsize=11, fontweight="bold")
        ax.set_xlabel("Eigenvalue Index (i)")
        ax.set_ylabel("Eigenvalue (λᵢ)")
        ax.set_xticks(ev_indices)
        ax.legend(fontsize=8)

    plt.suptitle(
        "Normalized Graph Laplacian Eigenvalues (λ₁..λ₁₀) Across Neighborhood Scales\n"
        "[Higher k increases graph density, raising Fiedler eigenvalue λ₂]",
        fontsize=11, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plots_dir / "diag4_knn_scale_eigenvalues.png", dpi=300)
    plt.close()

    print("  [PLOTS] Scale diagnostic plots created.")


def run_knn_scale_diagnostics(base_dir: Path) -> None:
    diag_dir = base_dir / "clustering_sandbox" / "spectral_diagnostics"
    metrics_dir = diag_dir / "metrics"
    plots_dir = diag_dir / "plots"
    metadata_dir = diag_dir / "metadata"
    for d in [metrics_dir, plots_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    scaled_dir = base_dir / "output" / "scaled_matrices"
    matrices = {
        "A_RAW": scaled_dir / "A_RAW_scaled.csv",
        "A_LOG": scaled_dir / "A_LOG_scaled.csv",
        "B_RAW": scaled_dir / "B_RAW_scaled.csv",
        "B_LOG": scaled_dir / "B_LOG_scaled.csv",
    }

    results_by_matrix = {}
    flat_results = []

    print("=" * 70)
    print("  SPECTRAL kNN NEIGHBORHOOD SCALE SENSITIVITY DIAGNOSTIC")
    print("  Read-only. Evaluating k ∈ {5, 10, 15, 20}. No clustering.")
    print("=" * 70)

    t_global = time.time()

    for name, filepath in matrices.items():
        df_scaled = pd.read_csv(filepath)
        X = df_scaled.values.astype(np.float64)
        n, p = X.shape
        print(f"\n{'─'*60}")
        print(f"  Matrix: {name} | N={n:,} | p={p}")
        print(f"{'─'*60}")

        mat_results = []
        for k in K_SCALE_VALUES:
            res = compute_laplacian_spectrum_for_k(X, name, k, N_EIGENVALUES)
            mat_results.append(res)
            flat_results.append(res)
            print(
                f"    k={k:2d} | λ₂={res['fiedler_lambda2']:.6f} | "
                f"Fiedler Gap (1→2)={res['fiedler_gap_1_to_2']:.6f} | "
                f"Max Sub-Fiedler Gap={res['max_sub_fiedler_gap_mag']:.6f} at {res['max_sub_fiedler_gap_pos']}"
            )
        results_by_matrix[name] = mat_results

    # Generate plots
    create_scale_plots(results_by_matrix, plots_dir)

    # Save CSV tables
    # 1. Summary of Fiedler and Max Sub-Fiedler Gaps per (Matrix, k)
    summary_rows = []
    for r in flat_results:
        summary_rows.append({
            "matrix": r["matrix_name"],
            "N": r["n_samples"],
            "k": r["k"],
            "fiedler_lambda2": r["fiedler_lambda2"],
            "gap_1_to_2_fiedler": r["fiedler_gap_1_to_2"],
            "gap_2_to_3": r["eigengaps"][1],
            "gap_3_to_4": r["eigengaps"][2],
            "gap_4_to_5": r["eigengaps"][3],
            "gap_5_to_6": r["eigengaps"][4],
            "gap_6_to_7": r["eigengaps"][5],
            "gap_7_to_8": r["eigengaps"][6],
            "gap_8_to_9": r["eigengaps"][7],
            "gap_9_to_10": r["eigengaps"][8],
            "max_sub_fiedler_gap_pos": r["max_sub_fiedler_gap_pos"],
            "max_sub_fiedler_gap_mag": r["max_sub_fiedler_gap_mag"],
        })
    pd.DataFrame(summary_rows).to_csv(
        metrics_dir / "diag4_knn_scale_summary.csv", index=False
    )

    # 2. Detailed Eigenvalues Table
    ev_rows = []
    for r in flat_results:
        for idx_ev, ev in enumerate(r["eigenvalues"]):
            gap = r["eigengaps"][idx_ev] if idx_ev < len(r["eigengaps"]) else None
            ev_rows.append({
                "matrix": r["matrix_name"],
                "k": r["k"],
                "eigenvalue_index": idx_ev + 1,
                "eigenvalue": ev,
                "eigengap_to_next": gap
            })
    pd.DataFrame(ev_rows).to_csv(
        metrics_dir / "diag4_knn_scale_eigenvalues.csv", index=False
    )

    # Update metadata JSON
    meta_file = metadata_dir / "affinity_diagnostics_metadata.json"
    if meta_file.exists():
        with open(meta_file, "r") as f:
            meta = json.load(f)
    else:
        meta = {}

    meta["knn_scale_diagnostic"] = {
        "status": "COMPLETED",
        "k_values_tested": K_SCALE_VALUES,
        "symmetrization_rule": "W = (A + A^T) / 2",
        "summary": {
            r["matrix_name"] + f"_k{r['k']}": {
                "fiedler_lambda2": r["fiedler_lambda2"],
                "gap_3_to_4": r["eigengaps"][2],
                "max_sub_fiedler_gap_pos": r["max_sub_fiedler_gap_pos"],
                "max_sub_fiedler_gap_mag": r["max_sub_fiedler_gap_mag"]
            } for r in flat_results
        }
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    total_elapsed = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"  SCALE DIAGNOSTIC COMPLETE")
    print(f"  Total runtime: {total_elapsed:.1f} seconds")
    print(f"  Outputs in: clustering_sandbox/spectral_diagnostics/")
    print(f"    metrics/diag4_knn_scale_summary.csv")
    print(f"    metrics/diag4_knn_scale_eigenvalues.csv")
    print(f"    plots/diag4_knn_scale_eigengaps.png")
    print(f"    plots/diag4_knn_scale_eigenvalues.png")
    print(f"{'='*70}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    run_knn_scale_diagnostics(base_dir)
