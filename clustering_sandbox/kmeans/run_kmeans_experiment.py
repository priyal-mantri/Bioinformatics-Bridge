"""
clustering_sandbox/kmeans/run_kmeans_experiment.py
===================================================
Exploratory K-Means Sandbox Experiment.

Evaluates K-Means clustering across K = 2..10 for four candidate standardized feature matrices:
  1. A_RAW_scaled (4,482 x 19)
  2. A_LOG_scaled (4,482 x 19)
  3. B_RAW_scaled (967 x 24)
  4. B_LOG_scaled (967 x 24)

STRICT CONSTRAINTS:
  - Clustering performed on FULL 19D / 24D standardized feature space.
  - PCA space is NOT used as clustering input.
  - 2D PCA is used ONLY for visualization scatter plots, explicitly labeled as visualization-only.
  - Results are exploratory only and NOT part of formal research decision log yet.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
import matplotlib.pyplot as plt
import seaborn as sns


# Styling settings for publication-quality exploratory plots
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


RANDOM_SEED = 42
N_INIT = 25
K_RANGE = list(range(2, 11))  # K = 2..10


def run_kmeans_sweep_for_matrix(
    name: str,
    df_scaled: pd.DataFrame,
    df_cohort_source: pd.DataFrame,
    output_base: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run K-Means sweep (K=2..10) on a single scaled matrix.

    Returns:
      (metrics_df, assignments_df, summary_dict)
    """
    n_samples, n_features = df_scaled.shape
    X = df_scaled.values

    metrics_records = []
    assignments_dict = {"SEQN": df_cohort_source["SEQN"].values}

    for k in K_RANGE:
        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_SEED,
            n_init=N_INIT
        )
        labels = kmeans.fit_predict(X)

        inertia = float(kmeans.inertia_)
        sil = float(silhouette_score(X, labels, random_state=RANDOM_SEED))
        ch = float(calinski_harabasz_score(X, labels))
        db = float(davies_bouldin_score(X, labels))

        # Count cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(lbl): int(cnt) for lbl, cnt in zip(unique_labels, counts)}

        metrics_records.append({
            "matrix_name": name,
            "K": k,
            "n_samples": n_samples,
            "n_features": n_features,
            "inertia": inertia,
            "silhouette_score": sil,
            "calinski_harabasz_score": ch,
            "davies_bouldin_score": db,
            "cluster_sizes": json.dumps(cluster_sizes),
            "min_cluster_size": int(min(counts)),
            "max_cluster_size": int(max(counts)),
            "random_seed": RANDOM_SEED,
            "n_init": N_INIT,
            "sklearn_version": sklearn.__version__
        })

        assignments_dict[f"K{k}_cluster"] = labels

    metrics_df = pd.DataFrame(metrics_records)
    assignments_df = pd.DataFrame(assignments_dict)

    summary_dict = {
        "matrix_name": name,
        "n_samples": n_samples,
        "n_features": n_features,
        "runs": metrics_records
    }

    return metrics_df, assignments_df, summary_dict


def generate_kmeans_plots(
    name: str,
    metrics_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    df_scaled: pd.DataFrame,
    plots_dir: Path
) -> None:
    """
    Generate diagnostics plots:
      1. 2x2 grid of evaluation metrics (Inertia, Silhouette, Calinski-Harabasz, Davies-Bouldin)
      2. Cluster size distribution bar chart across K=2..10
      3. 2x2 grid of 2D PCA score scatter plots colored by assignments for K=2,3,4,5 (visualization only)
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    ks = metrics_df["K"].values

    # ── 1. Metrics Summary Plot (2x2 Grid) ───────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

    # (A) Inertia / Elbow
    axes[0, 0].plot(ks, metrics_df["inertia"], "o-", color="#1f77b4", linewidth=2, markersize=6)
    axes[0, 0].set_title(f"Inertia (Elbow Curve) — {name}", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Number of Clusters (K)", fontsize=10)
    axes[0, 0].set_ylabel("Inertia (WCSS)", fontsize=10)
    axes[0, 0].set_xticks(ks)

    # (B) Silhouette Score
    axes[0, 1].plot(ks, metrics_df["silhouette_score"], "s-", color="#2ca02c", linewidth=2, markersize=6)
    axes[0, 1].set_title(f"Silhouette Score (Higher = Better) — {name}", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Number of Clusters (K)", fontsize=10)
    axes[0, 1].set_ylabel("Silhouette Score", fontsize=10)
    axes[0, 1].set_xticks(ks)

    # (C) Calinski-Harabasz Score
    axes[1, 0].plot(ks, metrics_df["calinski_harabasz_score"], "^-", color="#ff7f0e", linewidth=2, markersize=6)
    axes[1, 0].set_title(f"Calinski-Harabasz Score (Higher = Better) — {name}", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Number of Clusters (K)", fontsize=10)
    axes[1, 0].set_ylabel("Calinski-Harabasz Score", fontsize=10)
    axes[1, 0].set_xticks(ks)

    # (D) Davies-Bouldin Score
    axes[1, 1].plot(ks, metrics_df["davies_bouldin_score"], "d-", color="#d62728", linewidth=2, markersize=6)
    axes[1, 1].set_title(f"Davies-Bouldin Score (Lower = Better) — {name}", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Number of Clusters (K)", fontsize=10)
    axes[1, 1].set_ylabel("Davies-Bouldin Score", fontsize=10)
    axes[1, 1].set_xticks(ks)

    fig.tight_layout()
    metrics_plot_path = plots_dir / f"{name}_kmeans_metrics_summary.png"
    plt.savefig(metrics_plot_path)
    plt.close()

    # ── 2. Cluster Size Distribution Plot ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Prepare data for grouped bar chart
    size_data = []
    for _, row in metrics_df.iterrows():
        k_val = int(row["K"])
        sizes = json.loads(row["cluster_sizes"])
        for cluster_id, count in sizes.items():
            pct = (count / row["n_samples"]) * 100.0
            size_data.append({"K": f"K={k_val}", "Cluster": f"C{cluster_id}", "Count": count, "Pct": pct})
    
    size_df = pd.DataFrame(size_data)
    
    # Plot stacked bar chart of percentage sizes
    pivot_df = size_df.pivot(index="K", columns="Cluster", values="Pct").fillna(0)
    pivot_df.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", edgecolor="white", width=0.7)
    
    ax.set_title(f"Cluster Size Distribution (% of Total N) — {name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Clusters (K)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Percentage of Participants (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(title="Cluster ID", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)

    fig.tight_layout()
    sizes_plot_path = plots_dir / f"{name}_kmeans_cluster_sizes.png"
    plt.savefig(sizes_plot_path)
    plt.close()

    # ── 3. 2D Projection Plots (Visualization ONLY) ─────────────────────────
    # Compute 2D PCA projection of full standardized space strictly for scatter plot rendering
    pca_2d = PCA(n_components=2, random_state=RANDOM_SEED)
    coords_2d = pca_2d.fit_transform(df_scaled)
    evr1 = pca_2d.explained_variance_ratio_[0] * 100.0
    evr2 = pca_2d.explained_variance_ratio_[1] * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(11, 10), dpi=300)
    fig.suptitle(
        f"2D Projection of Full-Space K-Means Clusters — {name}\n"
        f"(*Visualization-Only projection via PCA; K-Means was computed on full {df_scaled.shape[1]}D standardized space*)",
        fontsize=11, fontweight="bold", y=0.98
    )

    k_to_plot = [2, 3, 4, 5]
    for idx, k_val in enumerate(k_to_plot):
        row_idx = idx // 2
        col_idx = idx % 2
        ax = axes[row_idx, col_idx]

        labels = assignments_df[f"K{k_val}_cluster"].values
        scatter = ax.scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c=labels, cmap="Set1", alpha=0.5, s=12, edgecolors="none"
        )
        ax.set_title(f"K = {k_val} Clusters", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"PC1 ({evr1:.1f}% var)", fontsize=9)
        ax.set_ylabel(f"PC2 ({evr2:.1f}% var)", fontsize=9)
        ax.axhline(0, color="#aaaaaa", linestyle="--", linewidth=0.7)
        ax.axvline(0, color="#aaaaaa", linestyle="--", linewidth=0.7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    proj_plot_path = plots_dir / f"{name}_kmeans_2d_projections.png"
    plt.savefig(proj_plot_path)
    plt.close()


def run_kmeans_sandbox_pipeline(base_dir: Path) -> dict:
    """
    Execute full exploratory K-Means sandbox experiment across all 4 matrices.
    """
    scaled_dir = base_dir / "output" / "scaled_matrices"
    cohort_a_path = base_dir / "output" / "analysis_cohort_a_broad.csv"
    cohort_b_path = base_dir / "output" / "analysis_cohort_b_fasting.csv"

    sandbox_dir = base_dir / "clustering_sandbox" / "kmeans"
    metrics_dir = sandbox_dir / "metrics"
    assignments_dir = sandbox_dir / "cluster_assignments"
    plots_dir = sandbox_dir / "plots"
    metadata_dir = sandbox_dir / "metadata"

    for d in [metrics_dir, assignments_dir, plots_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    matrices = {
        "A_RAW": (scaled_dir / "A_RAW_scaled.csv", cohort_a_path),
        "A_LOG": (scaled_dir / "A_LOG_scaled.csv", cohort_a_path),
        "B_RAW": (scaled_dir / "B_RAW_scaled.csv", cohort_b_path),
        "B_LOG": (scaled_dir / "B_LOG_scaled.csv", cohort_b_path),
    }

    all_summaries = {}
    total_runs = 0

    print("=================================================================")
    print("  EXPLORATORY CLUSTERING SANDBOX — K-MEANS EXPERIMENT")
    print("=================================================================")

    for name, (scaled_file, cohort_file) in matrices.items():
        df_scaled = pd.read_csv(scaled_file)
        df_cohort = pd.read_csv(cohort_file)

        print(f"\n  [Running K-Means Sweep K=2..10] -> {name} ({df_scaled.shape[0]:,} rows x {df_scaled.shape[1]} features)")

        metrics_df, assignments_df, summary = run_kmeans_sweep_for_matrix(
            name, df_scaled, df_cohort, sandbox_dir
        )

        # Export CSVs
        metrics_csv = metrics_dir / f"{name}_kmeans_metrics.csv"
        assignments_csv = assignments_dir / f"{name}_kmeans_assignments.csv"

        metrics_df.to_csv(metrics_csv, index=False)
        assignments_df.to_csv(assignments_csv, index=False)

        # Generate plots
        generate_kmeans_plots(name, metrics_df, assignments_df, df_scaled, plots_dir)

        all_summaries[name] = summary
        total_runs += len(metrics_df)

        print(f"    - Completed {len(metrics_df)} runs (K=2..10)")
        print(f"    - Metrics exported to: {metrics_csv.relative_to(base_dir)}")
        print(f"    - Assignments exported to: {assignments_csv.relative_to(base_dir)}")

    # Create metadata JSON
    metadata = {
        "experiment": "EXPLORATORY_CLUSTERING_SANDBOX_KMEANS",
        "status": "EXPERIMENTAL_SANDBOX_ONLY",
        "description": "Exploratory K-Means clustering sweep across K=2..10 for candidate standardized feature matrices.",
        "formal_pipeline_status": "NOT_MERGED_TO_FORMAL_PIPELINE",
        "methodological_constraints_confirmed": {
            "clustering_input_space": "FULL_STANDARDIZED_FEATURE_SPACE",
            "pca_used_as_clustering_input": False,
            "matrices_modified": False,
            "cohorts_combined": False,
            "raw_and_log_combined": False,
            "tridosha_mapping_performed": False,
            "final_k_selected": False,
            "formal_decision_log_modified": False
        },
        "configuration": {
            "k_range": K_RANGE,
            "random_seed": RANDOM_SEED,
            "n_init": N_INIT,
            "sklearn_version": sklearn.__version__
        },
        "total_runs_completed": total_runs,
        "matrices": list(matrices.keys()),
        "results_summary": {
            name: {
                "n_samples": summ["n_samples"],
                "n_features": summ["n_features"],
                "best_silhouette_k": int(max(summ["runs"], key=lambda x: x["silhouette_score"])["K"]),
                "best_silhouette_val": float(max(summ["runs"], key=lambda x: x["silhouette_score"])["silhouette_score"]),
                "best_ch_k": int(max(summ["runs"], key=lambda x: x["calinski_harabasz_score"])["K"]),
                "best_ch_val": float(max(summ["runs"], key=lambda x: x["calinski_harabasz_score"])["calinski_harabasz_score"]),
                "best_db_k": int(min(summ["runs"], key=lambda x: x["davies_bouldin_score"])["K"]),
                "best_db_val": float(min(summ["runs"], key=lambda x: x["davies_bouldin_score"])["davies_bouldin_score"]),
            } for name, summ in all_summaries.items()
        }
    }

    meta_file = metadata_dir / "kmeans_sandbox_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Write README.md inside clustering_sandbox/kmeans/
    readme_content = f"""# K-Means Exploratory Clustering Sandbox

## Overview

This directory contains the exploratory outputs from the initial **K-Means clustering sweep** ($K = 2 \dots 10$) performed on the four candidate standardized feature matrices:

1. `A_RAW_scaled.csv` ($N = 4,482$, $p = 19$)
2. `A_LOG_scaled.csv` ($N = 4,482$, $p = 19$)
3. `B_RAW_scaled.csv` ($N = 967$, $p = 24$)
4. `B_LOG_scaled.csv` ($N = 967$, $p = 24$)

> ⚠️ **EXPERIMENTAL STATUS**: These outputs are part of the **`experiment/clustering-sandbox`** branch. They are strictly exploratory diagnostics and have **NOT** been selected as formal methodological decisions for the main research pipeline.

---

## Methodological Constraints Confirmed

- **Input Space**: Clustering was executed directly on the **full standardized feature space** (19 dimensions for A; 24 dimensions for B).
- **PCA Role**: PCA was **NOT** used as a clustering input space. 2D PCA coordinate projections were computed **strictly for rendering 2D visualization scatter plots**.
- **Cohort Separation**: Cohort A and Cohort B, as well as RAW and LOG representations, were kept 100% separate.
- **Reproducibility**: `random_state = {RANDOM_SEED}`, `n_init = {N_INIT}`, Scikit-Learn `{sklearn.__version__}`.

---

## Execution Summary

- **Total Runs Completed**: {total_runs} runs (4 matrices $\\times$ 9 $K$ values: $K=2..10$)
- **Outputs Directory**:
  - `metrics/`: CSV files containing $K$, inertia, silhouette score, Calinski-Harabasz score, Davies-Bouldin score, and cluster size counts.
  - `cluster_assignments/`: Participant-level cluster assignment tables (preserving `SEQN` row alignment).
  - `plots/`: Metric evaluation curves, cluster size distribution bar charts, and 2D visualization scatter plots.
  - `metadata/`: Full JSON metadata.

---

*Generated by `clustering_sandbox/kmeans/run_kmeans_experiment.py`*
"""

    readme_file = sandbox_dir / "README.md"
    with open(readme_file, "w") as f:
        f.write(readme_content)

    print(f"\n  [SAVED METADATA] -> {meta_file.relative_to(base_dir)}")
    print(f"  [SAVED README]   -> {readme_file.relative_to(base_dir)}")
    print("=================================================================")

    return metadata


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    run_kmeans_sandbox_pipeline(base_dir)
