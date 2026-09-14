#!/usr/bin/env python3
"""
Experiment 5 Sensitivity Extension: Distance-Metric Sensitivity Analysis
Bioinfo Bridge — Research 2 (NHANES 2017–2018)

Evaluates whether changing from Euclidean (L2) to Manhattan (L1 / Cityblock) distance 
substantially improves cluster structure, size balance, or subsampling stability for 
Hierarchical Agglomerative Clustering using Complete and Average linkages.

Evaluated Grids:
- Matrices: A_RAW, A_LOG, B_RAW, B_LOG
- Distance Metrics: Euclidean (L2), Manhattan (L1)
- Linkages: Complete ('complete'), Average ('average')  [Ward is excluded for Manhattan]
- Cluster Counts: K = 2..7
- Subsampling Stability: Participant-overlap ARI across B=100 iterations (80% subsample size)
"""

import os
import sys
import json
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from concurrent.futures import ProcessPoolExecutor

# Set seed for reproducible subsampling
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALED_MATRICES_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../output/scaled_matrices"))
ASSIGNMENTS_DIR = os.path.join(BASE_DIR, "cluster_assignments")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")

for d in [ASSIGNMENTS_DIR, METRICS_DIR, PLOTS_DIR, METADATA_DIR]:
    os.makedirs(d, exist_ok=True)

COHORT_FILES = {
    "A": os.path.abspath(os.path.join(BASE_DIR, "../../../output/analysis_cohort_a_broad.csv")),
    "B": os.path.abspath(os.path.join(BASE_DIR, "../../../output/analysis_cohort_b_fasting.csv"))
}

MATRICES_CONFIG = {
    "A_RAW": {
        "file": "A_RAW_scaled.csv",
        "cohort_key": "A",
        "cohort": "Broad (A)",
        "scale": "Raw",
        "expected_N": 4482,
        "expected_p": 19
    },
    "A_LOG": {
        "file": "A_LOG_scaled.csv",
        "cohort_key": "A",
        "cohort": "Broad (A)",
        "scale": "Log1p",
        "expected_N": 4482,
        "expected_p": 19
    },
    "B_RAW": {
        "file": "B_RAW_scaled.csv",
        "cohort_key": "B",
        "cohort": "Fasting (B)",
        "scale": "Raw",
        "expected_N": 967,
        "expected_p": 24
    },
    "B_LOG": {
        "file": "B_LOG_scaled.csv",
        "cohort_key": "B",
        "cohort": "Fasting (B)",
        "scale": "Log1p",
        "expected_N": 967,
        "expected_p": 24
    }
}

DISTANCE_METRICS = {
    "euclidean": {
        "sklearn_metric": "euclidean",
        "scipy_metric": "euclidean",
        "dispersion_name": "WSS_L2"
    },
    "manhattan": {
        "sklearn_metric": "manhattan",
        "scipy_metric": "cityblock",
        "dispersion_name": "WAD_L1"
    }
}

LINKAGE_METHODS = ["complete", "average"]
K_RANGE = list(range(2, 8))
N_SUBSAMPLES = 100
SUBSAMPLE_RATIO = 0.80

def compute_within_dispersion(X, labels, dist_metric_key):
    """Compute within-cluster dispersion: WSS (L2 squared error) for Euclidean, WAD (L1 absolute deviation) for Manhattan."""
    unique_labels = np.unique(labels)
    dispersion = 0.0
    
    for k in unique_labels:
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            if dist_metric_key == "euclidean":
                center = np.mean(cluster_points, axis=0)
                dispersion += np.sum((cluster_points - center) ** 2)
            else:  # manhattan
                center = np.median(cluster_points, axis=0)
                dispersion += np.sum(np.abs(cluster_points - center))
                
    return float(dispersion)

def _subsample_worker(X, sub_indices, linkage_name, scipy_metric, k_range, baseline_labels_dict):
    X_sub = X[sub_indices]
    Z_sub = linkage(X_sub, method=linkage_name, metric=scipy_metric)
    res = {}
    for K in k_range:
        sub_labels = fcluster(Z_sub, t=K, criterion='maxclust') - 1
        baseline_sub = baseline_labels_dict[K][sub_indices]
        res[K] = adjusted_rand_score(baseline_sub, sub_labels)
    return res

def compute_stability_for_combination(X, linkage_name, dist_metric_key, k_range, n_iterations=100, ratio=0.80):
    scipy_metric = DISTANCE_METRICS[dist_metric_key]["scipy_metric"]
    N = len(X)
    n_sub = int(np.floor(ratio * N))
    
    # Baseline full-dataset linkage
    Z_full = linkage(X, method=linkage_name, metric=scipy_metric)
    baseline_labels_dict = {K: fcluster(Z_full, t=K, criterion='maxclust') - 1 for K in k_range}
    
    rng = np.random.RandomState(RANDOM_SEED)
    sub_indices_list = [rng.choice(N, size=n_sub, replace=False) for _ in range(n_iterations)]
    
    tasks = [
        (X, sub_indices, linkage_name, scipy_metric, k_range, baseline_labels_dict)
        for sub_indices in sub_indices_list
    ]
    
    ari_dict = {K: [] for K in k_range}
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_subsample_worker, *t) for t in tasks]
        for f in futures:
            res = f.result()
            for K in k_range:
                ari_dict[K].append(res[K])
                
    res = {}
    for K in k_range:
        scores = np.array(ari_dict[K])
        res[K] = {
            "mean_ari": float(np.mean(scores)),
            "median_ari": float(np.median(scores)),
            "std_ari": float(np.std(scores)),
            "p05_ari": float(np.percentile(scores, 5)),
            "p95_ari": float(np.percentile(scores, 95)),
            "valid_iterations": int(len(scores))
        }
    return res

def run_experiment():
    print("=" * 80)
    print("EXPERIMENT 5 SENSITIVITY EXTENSION: DISTANCE METRIC SENSITIVITY")
    print("=" * 80)
    
    all_metrics = []
    all_stability = []
    matrix_data = {}
    start_time = time.time()
    
    for mat_key, cfg in MATRICES_CONFIG.items():
        file_path = os.path.join(SCALED_MATRICES_DIR, cfg["file"])
        cohort_file = COHORT_FILES[cfg["cohort_key"]]
        print(f"\n--- Loading {mat_key} ({cfg['file']}) ---")
        
        df_scaled = pd.read_csv(file_path)
        df_cohort = pd.read_csv(cohort_file)
        
        seqn = df_cohort["SEQN"].values
        X = df_scaled.values
        N, p = X.shape
        
        matrix_data[mat_key] = {"seqn": seqn, "X": X}
        assignments_df = pd.DataFrame({"SEQN": seqn})
        
        for dist_key, dist_info in DISTANCE_METRICS.items():
            skl_metric = dist_info["sklearn_metric"]
            scipy_metric = dist_info["scipy_metric"]
            
            for linkage_name in LINKAGE_METHODS:
                print(f"  > Dist: {dist_key.upper()} | Linkage: {linkage_name.upper()}")
                print(f"    - Computing participant-overlap stability (B={N_SUBSAMPLES}, 80%)...", end="", flush=True)
                stab_dict = compute_stability_for_combination(X, linkage_name, dist_key, K_RANGE, N_SUBSAMPLES, SUBSAMPLE_RATIO)
                print(" Done.")
                
                # Compute baseline full dataset linkage via scipy for consistency
                Z_full = linkage(X, method=linkage_name, metric=scipy_metric)
                
                for K in K_RANGE:
                    labels = fcluster(Z_full, t=K, criterion='maxclust') - 1
                    
                    # Store assignment
                    col_name = f"{dist_key}_{linkage_name}_K{K}"
                    assignments_df[col_name] = labels
                    
                    # Compute quality metrics with matching distance metric
                    sil = float(silhouette_score(X, labels, metric=skl_metric))
                    ch = float(calinski_harabasz_score(X, labels))
                    db = float(davies_bouldin_score(X, labels))
                    dispersion = compute_within_dispersion(X, labels, dist_key)
                    
                    counts = pd.Series(labels).value_counts().sort_index().tolist()
                    min_size = min(counts)
                    max_size = max(counts)
                    min_prop = min_size / N
                    max_prop = max_size / N
                    max_min_ratio = max_size / min_size
                    small_cluster_flag = bool(min_prop < 0.01)
                    
                    stab = stab_dict[K]
                    
                    metric_record = {
                        "matrix": mat_key,
                        "cohort": cfg["cohort"],
                        "scale": cfg["scale"],
                        "distance_metric": dist_key,
                        "linkage": linkage_name,
                        "N": N,
                        "p": p,
                        "K": K,
                        "silhouette_score": round(sil, 6),
                        "calinski_harabasz_score": round(ch, 4),
                        "davies_bouldin_score": round(db, 4),
                        "within_dispersion": round(dispersion, 4),
                        "dispersion_type": dist_info["dispersion_name"],
                        "min_cluster_size": min_size,
                        "max_cluster_size": max_size,
                        "min_cluster_prop": round(min_prop, 6),
                        "max_cluster_prop": round(max_prop, 6),
                        "max_min_ratio": round(max_min_ratio, 4),
                        "small_cluster_flag": small_cluster_flag,
                        "cluster_sizes_str": str(counts),
                        "mean_ari": round(stab["mean_ari"], 6),
                        "median_ari": round(stab["median_ari"], 6),
                        "std_ari": round(stab["std_ari"], 6),
                        "p05_ari": round(stab["p05_ari"], 6),
                        "p95_ari": round(stab["p95_ari"], 6)
                    }
                    all_metrics.append(metric_record)
                    
                    stab_record = {
                        "matrix": mat_key,
                        "distance_metric": dist_key,
                        "linkage": linkage_name,
                        "K": K,
                        "mean_ari": round(stab["mean_ari"], 6),
                        "median_ari": round(stab["median_ari"], 6),
                        "std_ari": round(stab["std_ari"], 6),
                        "p05_ari": round(stab["p05_ari"], 6),
                        "p95_ari": round(stab["p95_ari"], 6),
                        "valid_iterations": stab["valid_iterations"]
                    }
                    all_stability.append(stab_record)
                    
                    print(f"    K={K}: Sil={sil:.4f}, CH={ch:.1f}, DB={db:.3f}, MinProp={min_prop*100:.2f}%, Mean ARI={stab['mean_ari']:.4f}")

        # Save cluster assignments for this matrix
        assign_path = os.path.join(ASSIGNMENTS_DIR, f"{mat_key}_distance_sensitivity_assignments.csv")
        assignments_df.to_csv(assign_path, index=False)
        print(f"Saved cluster assignments to {assign_path}")
        
        # Save per-matrix metrics
        mat_metrics_df = pd.DataFrame([m for m in all_metrics if m["matrix"] == mat_key])
        mat_metrics_path = os.path.join(METRICS_DIR, f"{mat_key}_distance_sensitivity_metrics.csv")
        mat_metrics_df.to_csv(mat_metrics_path, index=False)

    # Save summary tables
    summary_df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(METRICS_DIR, "distance_sensitivity_cross_matrix_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    stab_df = pd.DataFrame(all_stability)
    stab_path = os.path.join(METRICS_DIR, "distance_sensitivity_stability_summary.csv")
    stab_df.to_csv(stab_path, index=False)

    # Generate Visual Plots
    print("\n--- GENERATING SENSITIVITY COMPARISON PLOTS ---")
    generate_silhouette_comparison_plot(summary_df)
    generate_stability_comparison_plot(summary_df)
    generate_cluster_size_comparison_plot(summary_df)

    elapsed_time = round(time.time() - start_time, 2)
    
    # Save Metadata
    metadata = {
        "experiment_name": "Experiment 5 Sensitivity Extension: Distance-Metric Sensitivity",
        "project": "Bioinfo Bridge — Research 2",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_branch": "experiment/clustering-sandbox",
        "matrices_evaluated": list(MATRICES_CONFIG.keys()),
        "distance_metrics_evaluated": list(DISTANCE_METRICS.keys()),
        "linkages_evaluated": LINKAGE_METHODS,
        "k_range": K_RANGE,
        "subsampling_iterations": N_SUBSAMPLES,
        "subsampling_ratio": SUBSAMPLE_RATIO,
        "total_runs": len(all_metrics),
        "execution_time_seconds": elapsed_time,
        "random_seed": RANDOM_SEED
    }
    metadata_path = os.path.join(METADATA_DIR, "distance_sensitivity_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    print("=" * 80)
    print(f"DISTANCE SENSITIVITY EXPERIMENT COMPLETED IN {elapsed_time}s!")
    print("=" * 80)

def generate_silhouette_comparison_plot(summary_df):
    """Generates 2x2 comparison curves of Silhouette score across Euclidean vs Manhattan for Complete and Average linkage."""
    matrices = summary_df["matrix"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    styles = {
        ("euclidean", "complete"): {"color": "#1f77b4", "marker": "o", "ls": "-"},
        ("manhattan", "complete"): {"color": "#1f77b4", "marker": "s", "ls": "--"},
        ("euclidean", "average"): {"color": "#ff7f0e", "marker": "^", "ls": "-"},
        ("manhattan", "average"): {"color": "#ff7f0e", "marker": "d", "ls": "--"}
    }
    
    for i, mat_key in enumerate(matrices):
        ax = axes[i]
        mat_df = summary_df[summary_df["matrix"] == mat_key]
        
        for (dist_key, linkage_name), style_cfg in styles.items():
            sub = mat_df[(mat_df["distance_metric"] == dist_key) & (mat_df["linkage"] == linkage_name)].sort_values("K")
            ax.plot(sub["K"], sub["silhouette_score"],
                    label=f"{dist_key.capitalize()} | {linkage_name.capitalize()}",
                    color=style_cfg["color"], marker=style_cfg["marker"], linestyle=style_cfg["ls"],
                    linewidth=2, markersize=6)
            
        ax.set_title(f"{mat_key} — Silhouette Score (Euclidean vs. Manhattan)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Silhouette Score (Matching Metric)")
        ax.set_xticks(K_RANGE)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")
        
    plt.suptitle("Distance Metric Sensitivity: Silhouette Score Across K=2..7", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "distance_sensitivity_silhouette_vs_k.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved silhouette plot: {plot_path}")

def generate_stability_comparison_plot(summary_df):
    """Generates 2x2 comparison curves of Subsampling Mean ARI across Euclidean vs Manhattan."""
    matrices = summary_df["matrix"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    styles = {
        ("euclidean", "complete"): {"color": "#1f77b4", "marker": "o", "ls": "-"},
        ("manhattan", "complete"): {"color": "#1f77b4", "marker": "s", "ls": "--"},
        ("euclidean", "average"): {"color": "#ff7f0e", "marker": "^", "ls": "-"},
        ("manhattan", "average"): {"color": "#ff7f0e", "marker": "d", "ls": "--"}
    }
    
    for i, mat_key in enumerate(matrices):
        ax = axes[i]
        mat_df = summary_df[summary_df["matrix"] == mat_key]
        
        for (dist_key, linkage_name), style_cfg in styles.items():
            sub = mat_df[(mat_df["distance_metric"] == dist_key) & (mat_df["linkage"] == linkage_name)].sort_values("K")
            ax.plot(sub["K"], sub["mean_ari"],
                    label=f"{dist_key.capitalize()} | {linkage_name.capitalize()}",
                    color=style_cfg["color"], marker=style_cfg["marker"], linestyle=style_cfg["ls"],
                    linewidth=2, markersize=6)
            
        ax.axhline(0.60, color="gray", linestyle="--", alpha=0.7, label="Stability Benchmark (ARI=0.60)")
        ax.set_title(f"{mat_key} — Subsampling Mean ARI Stability", fontsize=11, fontweight='bold')
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Mean Adjusted Rand Index (ARI)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(K_RANGE)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")
        
    plt.suptitle("Distance Metric Sensitivity: Subsampling ARI Stability Across K=2..7", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "distance_sensitivity_stability_vs_k.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved stability plot: {plot_path}")

def generate_cluster_size_comparison_plot(summary_df):
    """Generates comparison plot of Minimum Cluster Proportion (%) across Euclidean vs Manhattan."""
    matrices = summary_df["matrix"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    styles = {
        ("euclidean", "complete"): {"color": "#1f77b4", "marker": "o", "ls": "-"},
        ("manhattan", "complete"): {"color": "#1f77b4", "marker": "s", "ls": "--"},
        ("euclidean", "average"): {"color": "#ff7f0e", "marker": "^", "ls": "-"},
        ("manhattan", "average"): {"color": "#ff7f0e", "marker": "d", "ls": "--"}
    }
    
    for i, mat_key in enumerate(matrices):
        ax = axes[i]
        mat_df = summary_df[summary_df["matrix"] == mat_key]
        
        for (dist_key, linkage_name), style_cfg in styles.items():
            sub = mat_df[(mat_df["distance_metric"] == dist_key) & (mat_df["linkage"] == linkage_name)].sort_values("K")
            min_props_pct = sub["min_cluster_prop"] * 100.0
            ax.plot(sub["K"], min_props_pct,
                    label=f"{dist_key.capitalize()} | {linkage_name.capitalize()}",
                    color=style_cfg["color"], marker=style_cfg["marker"], linestyle=style_cfg["ls"],
                    linewidth=2, markersize=6)
            
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="Diagnostic Flag Threshold (1%)")
        ax.set_title(f"{mat_key} — Minimum Cluster Size Proportion (%)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Minimum Cluster Proportion (%)")
        ax.set_xticks(K_RANGE)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8, loc="upper right")
        
    plt.suptitle("Distance Metric Sensitivity: Minimum Cluster Size Proportion (%) Across K=2..7", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "distance_sensitivity_cluster_sizes.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved cluster size plot: {plot_path}")

if __name__ == "__main__":
    run_experiment()
