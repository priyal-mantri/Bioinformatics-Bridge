#!/usr/bin/env python3
"""
Experiment 5: Hierarchical / Agglomerative Clustering Sandbox
Bioinfo Bridge — Research 2 (NHANES 2017–2018)

Evaluates agglomerative hierarchical clustering across four standardized candidate feature matrices:
- A_RAW_scaled.csv (N=4,482, p=19)
- A_LOG_scaled.csv (N=4,482, p=19)
- B_RAW_scaled.csv (N=967, p=24)
- B_LOG_scaled.csv (N=967, p=24)

Evaluated Linkage Methods:
- Primary: Ward ('ward')
- Sensitivity Analyses: Complete ('complete'), Average ('average')

Distance Metric: Euclidean distance (L2)

Cluster Range: K = 2..7

Stability Assessment: Participant-overlap Adjusted Rand Index (ARI) across B=100 subsampling iterations (80% subsample size without replacement).
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
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Set seed for reproducible subsampling
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALED_MATRICES_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../output/scaled_matrices"))
ASSIGNMENTS_DIR = os.path.join(BASE_DIR, "cluster_assignments")
METRICS_DIR = os.path.join(BASE_DIR, "metrics")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")

for d in [ASSIGNMENTS_DIR, METRICS_DIR, PLOTS_DIR, METADATA_DIR]:
    os.makedirs(d, exist_ok=True)

COHORT_FILES = {
    "A": os.path.abspath(os.path.join(BASE_DIR, "../../output/analysis_cohort_a_broad.csv")),
    "B": os.path.abspath(os.path.join(BASE_DIR, "../../output/analysis_cohort_b_fasting.csv"))
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

LINKAGE_METHODS = {
    "ward": {"type": "Primary", "metric": "euclidean"},
    "complete": {"type": "Sensitivity", "metric": "euclidean"},
    "average": {"type": "Sensitivity", "metric": "euclidean"}
}

K_RANGE = list(range(2, 8))  # K = 2..7
N_SUBSAMPLES = 100
SUBSAMPLE_RATIO = 0.80

def compute_wss(X, labels):
    """Compute Total Within-Cluster Sum of Squares (WSS) / Dispersion."""
    unique_labels = np.unique(labels)
    wss = 0.0
    for k in unique_labels:
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            wss += np.sum((cluster_points - centroid) ** 2)
    return float(wss)

from concurrent.futures import ProcessPoolExecutor

def _single_subsample_worker(X, sub_indices, linkage_name, k_range, baseline_labels_dict):
    X_sub = X[sub_indices]
    Z_sub = linkage(X_sub, method=linkage_name, metric='euclidean')
    res = {}
    for K in k_range:
        sub_labels = fcluster(Z_sub, t=K, criterion='maxclust') - 1
        baseline_sub = baseline_labels_dict[K][sub_indices]
        res[K] = adjusted_rand_score(baseline_sub, sub_labels)
    return res

def compute_all_stability_for_linkage(X, linkage_name, k_range, n_iterations=100, ratio=0.80):
    """
    Evaluates participant-overlap ARI stability over subsamples.
    For each subsample (80% of data without replacement), builds linkage dendrogram,
    cuts for all K in k_range, and computes ARI between baseline full-dataset assignments 
    and subsample assignments restricted strictly to the overlapping participants.
    """
    N = len(X)
    n_sub = int(np.floor(ratio * N))
    
    # Baseline assignment on full dataset via scipy linkage & fcluster
    Z_full = linkage(X, method=linkage_name, metric='euclidean')
    baseline_labels_dict = {K: fcluster(Z_full, t=K, criterion='maxclust') - 1 for K in k_range}
    
    rng = np.random.RandomState(RANDOM_SEED)
    sub_indices_list = [rng.choice(N, size=n_sub, replace=False) for _ in range(n_iterations)]
    
    tasks = [
        (X, sub_indices, linkage_name, k_range, baseline_labels_dict)
        for sub_indices in sub_indices_list
    ]
    
    ari_dict = {K: [] for K in k_range}
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_single_subsample_worker, *t) for t in tasks]
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
    print("STARTING EXPERIMENT 5: HIERARCHICAL / AGGLOMERATIVE CLUSTERING SWEEP")
    print("=" * 80)
    
    all_metrics = []
    all_stability = []
    matrix_data = {}
    
    start_time = time.time()
    
    for mat_key, cfg in MATRICES_CONFIG.items():
        file_path = os.path.join(SCALED_MATRICES_DIR, cfg["file"])
        cohort_file = COHORT_FILES[cfg["cohort_key"]]
        print(f"\n--- Loading {mat_key} from {cfg['file']} ---")
        df_scaled = pd.read_csv(file_path)
        df_cohort = pd.read_csv(cohort_file)
        
        # Extract SEQN and features
        seqn = df_cohort["SEQN"].values
        X = df_scaled.values
        N, p = X.shape
        
        assert N == cfg["expected_N"], f"N mismatch for {mat_key}: expected {cfg['expected_N']}, got {N}"
        assert p == cfg["expected_p"], f"p mismatch for {mat_key}: expected {cfg['expected_p']}, got {p}"
        print(f"Matrix {mat_key} loaded successfully: N={N}, p={p}")
        
        matrix_data[mat_key] = {"seqn": seqn, "X": X, "df": df_scaled}
        
        # DataFrame to collect cluster assignments for this matrix
        assignments_df = pd.DataFrame({"SEQN": seqn})
        
        for linkage_name, l_info in LINKAGE_METHODS.items():
            print(f"  > Evaluating Linkage: {linkage_name.upper()} ({l_info['type']})")
            print(f"    - Computing participant-overlap subsampling stability (B={N_SUBSAMPLES}, 80%)...", end="", flush=True)
            stab_dict = compute_all_stability_for_linkage(X, linkage_name, K_RANGE, N_SUBSAMPLES, SUBSAMPLE_RATIO)
            print(" Done.")
            
            for K in K_RANGE:
                print(f"    - Fitting K={K}...", end="", flush=True)
                
                # Fit model on full dataset
                model = AgglomerativeClustering(n_clusters=K, linkage=linkage_name, metric='euclidean')
                labels = model.fit_predict(X)
                
                # Store assignment
                col_name = f"{linkage_name}_K{K}"
                assignments_df[col_name] = labels
                
                # Metrics
                sil = float(silhouette_score(X, labels, metric='euclidean'))
                ch = float(calinski_harabasz_score(X, labels))
                db = float(davies_bouldin_score(X, labels))
                wss = compute_wss(X, labels)
                
                # Cluster size statistics
                counts = pd.Series(labels).value_counts().sort_index().tolist()
                min_size = min(counts)
                max_size = max(counts)
                min_prop = min_size / N
                max_prop = max_size / N
                max_min_ratio = max_size / min_size
                small_cluster_flag = bool(min_prop < 0.01)
                
                # Subsampling stability ARI
                stab = stab_dict[K]
                
                metric_record = {
                    "matrix": mat_key,
                    "cohort": cfg["cohort"],
                    "scale": cfg["scale"],
                    "linkage": linkage_name,
                    "linkage_type": l_info["type"],
                    "metric": "euclidean",
                    "N": N,
                    "p": p,
                    "K": K,
                    "silhouette_score": round(sil, 6),
                    "calinski_harabasz_score": round(ch, 4),
                    "davies_bouldin_score": round(db, 4),
                    "within_cluster_wss": round(wss, 4),
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
                
                print(f" Done. Sil={sil:.4f}, CH={ch:.1f}, DB={db:.3f}, Mean ARI={stab['mean_ari']:.4f}")
        
        # Save cluster assignments for this matrix
        assign_path = os.path.join(ASSIGNMENTS_DIR, f"{mat_key}_hierarchical_assignments.csv")
        assignments_df.to_csv(assign_path, index=False)
        print(f"Saved cluster assignments to {assign_path}")
        
        # Save per-matrix metrics
        mat_metrics_df = pd.DataFrame([m for m in all_metrics if m["matrix"] == mat_key])
        mat_metrics_path = os.path.join(METRICS_DIR, f"{mat_key}_hierarchical_metrics.csv")
        mat_metrics_df.to_csv(mat_metrics_path, index=False)
        print(f"Saved metrics to {mat_metrics_path}")

    # Save summary tables
    summary_df = pd.DataFrame(all_metrics)
    summary_path = os.path.join(METRICS_DIR, "hierarchical_cross_matrix_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved cross-matrix summary to {summary_path}")
    
    stab_df = pd.DataFrame(all_stability)
    stab_path = os.path.join(METRICS_DIR, "hierarchical_stability_summary.csv")
    stab_df.to_csv(stab_path, index=False)
    print(f"Saved stability summary to {stab_path}")

    # Generate Visualizations
    print("\n--- GENERATING PLOTS ---")
    generate_dendrogram_plots(matrix_data)
    generate_metrics_plots(summary_df)
    generate_stability_plots(summary_df)
    generate_cluster_size_plots(summary_df)
    generate_pca_projection_plots(matrix_data)

    elapsed_time = round(time.time() - start_time, 2)
    
    # Save Metadata
    metadata = {
        "experiment_name": "Experiment 5: Hierarchical / Agglomerative Clustering Sandbox",
        "project": "Bioinfo Bridge — Research 2",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_branch": "experiment/clustering-sandbox",
        "matrices_evaluated": list(MATRICES_CONFIG.keys()),
        "linkages_evaluated": list(LINKAGE_METHODS.keys()),
        "distance_metric": "euclidean",
        "k_range": K_RANGE,
        "subsampling_iterations": N_SUBSAMPLES,
        "subsampling_ratio": SUBSAMPLE_RATIO,
        "total_runs": len(all_metrics),
        "execution_time_seconds": elapsed_time,
        "random_seed": RANDOM_SEED
    }
    metadata_path = os.path.join(METADATA_DIR, "hierarchical_experiment_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    print("=" * 80)
    print(f"EXPERIMENT 5 COMPLETED SUCCESSFULLY IN {elapsed_time}s!")
    print("=" * 80)

def generate_dendrogram_plots(matrix_data):
    """Generates dual-panel dendrogram plots (Truncated full-data & Legible N=250 sample)."""
    for mat_key, data in matrix_data.items():
        X = data["X"]
        N = len(X)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Panel A: Truncated full-dataset dendrogram (Ward)
        Z_full = linkage(X, method='ward', metric='euclidean')
        dendrogram(Z_full, p=30, truncate_mode='lastp', ax=axes[0], show_leaf_counts=True, color_threshold=0.7*max(Z_full[:,2]))
        axes[0].set_title(f"{mat_key} — Full Dataset Truncated Dendrogram (Ward, N={N})", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Merged Cluster / Participant Count")
        axes[0].set_ylabel("Ward Euclidean Merge Distance")
        axes[0].grid(True, linestyle="--", alpha=0.4)
        
        # Panel B: Legible stratified sample dendrogram (N=250)
        rng = np.random.RandomState(RANDOM_SEED)
        sample_idx = rng.choice(N, size=min(250, N), replace=False)
        X_sample = X[sample_idx]
        Z_sample = linkage(X_sample, method='ward', metric='euclidean')
        dendrogram(Z_sample, p=250, ax=axes[1], leaf_rotation=90, leaf_font_size=6, color_threshold=0.7*max(Z_sample[:,2]))
        axes[1].set_title(f"{mat_key} — Representative Sample Dendrogram (Ward, Sample N=250)", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Participant Sample Index")
        axes[1].set_ylabel("Ward Euclidean Merge Distance")
        axes[1].grid(True, linestyle="--", alpha=0.4)
        
        plt.suptitle(f"Hierarchical Clustering Dendrogram Analysis — {mat_key}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f"dendrogram_{mat_key}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved dendrogram plot: {plot_path}")

def generate_metrics_plots(summary_df):
    """Generates quality metric evaluation curves across K for Ward, Complete, and Average linkages."""
    matrices = summary_df["matrix"].unique()
    
    fig, axes = plt.subplots(len(matrices), 4, figsize=(22, 4 * len(matrices)))
    if len(matrices) == 1:
        axes = np.expand_dims(axes, 0)
        
    metrics_to_plot = [
        ("silhouette_score", "Silhouette Score (↑ Higher = Better)"),
        ("calinski_harabasz_score", "Calinski-Harabasz Index (↑ Higher = Better)"),
        ("davies_bouldin_score", "Davies-Bouldin Index (↓ Lower = Better)"),
        ("within_cluster_wss", "Within-Cluster WSS Dispersion (↓ Lower = Better)")
    ]
    
    linkage_colors = {"ward": "#1f77b4", "complete": "#ff7f0e", "average": "#2ca02c"}
    linkage_markers = {"ward": "o", "complete": "s", "average": "^"}
    
    for i, mat_key in enumerate(matrices):
        mat_df = summary_df[summary_df["matrix"] == mat_key]
        
        for j, (metric_col, title_label) in enumerate(metrics_to_plot):
            ax = axes[i, j]
            
            for linkage_name in ["ward", "complete", "average"]:
                sub = mat_df[mat_df["linkage"] == linkage_name].sort_values("K")
                ax.plot(sub["K"], sub[metric_col], marker=linkage_markers[linkage_name],
                        color=linkage_colors[linkage_name], label=f"{linkage_name.capitalize()}",
                        linewidth=2, markersize=7)
                
            ax.set_title(f"{mat_key}: {title_label}", fontsize=10, fontweight='bold')
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel(metric_col.replace("_", " ").title())
            ax.set_xticks(K_RANGE)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=8)
            
    plt.suptitle("Hierarchical Clustering Diagnostic Quality Metrics Across K=2..7", fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "hierarchical_metrics_vs_k.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved metrics plot: {plot_path}")

def generate_stability_plots(summary_df):
    """Generates subsampling stability ARI curves with 5th-95th percentile confidence bands across K."""
    matrices = summary_df["matrix"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    linkage_colors = {"ward": "#1f77b4", "complete": "#ff7f0e", "average": "#2ca02c"}
    
    for i, mat_key in enumerate(matrices):
        ax = axes[i]
        mat_df = summary_df[summary_df["matrix"] == mat_key]
        
        for linkage_name in ["ward", "complete", "average"]:
            sub = mat_df[mat_df["linkage"] == linkage_name].sort_values("K")
            
            ax.plot(sub["K"], sub["mean_ari"], marker="o", color=linkage_colors[linkage_name],
                    label=f"{linkage_name.capitalize()} (Mean ARI)", linewidth=2.5)
            ax.fill_between(sub["K"], sub["p05_ari"], sub["p95_ari"], color=linkage_colors[linkage_name],
                            alpha=0.15)
            
        ax.axhline(0.60, color="gray", linestyle="--", alpha=0.7, label="Stability Benchmark (ARI=0.60)")
        ax.set_title(f"{mat_key} — Participant-Overlap Subsampling ARI Stability (B=100, 80%)", fontsize=11, fontweight='bold')
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Adjusted Rand Index (ARI)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(K_RANGE)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, loc="lower left")
        
    plt.suptitle("Hierarchical Clustering Subsampling Stability (Participant Overlap ARI) Across K=2..7", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "hierarchical_stability_vs_k.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved stability plot: {plot_path}")

def generate_cluster_size_plots(summary_df):
    """Generates cluster size proportion distributions across K for Ward linkage."""
    ward_df = summary_df[summary_df["linkage"] == "ward"]
    matrices = ward_df["matrix"].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, mat_key in enumerate(matrices):
        ax = axes[i]
        mat_df = ward_df[ward_df["matrix"] == mat_key].sort_values("K")
        
        x_positions = np.arange(len(K_RANGE))
        
        # Build proportion stack data
        for k_idx, K in enumerate(K_RANGE):
            row = mat_df[mat_df["K"] == K].iloc[0]
            sizes = eval(row["cluster_sizes_str"])
            props = [s / row["N"] for s in sizes]
            
            bottom = 0
            for cluster_i, prop in enumerate(props):
                color = plt.cm.tab10(cluster_i % 10)
                ax.bar(k_idx, prop, bottom=bottom, color=color, edgecolor="white", width=0.6)
                if prop > 0.05:
                    ax.text(k_idx, bottom + prop / 2, f"{prop*100:.1f}%", ha='center', va='center',
                            color="white", fontweight='bold', fontsize=8)
                bottom += prop
                
        # Flag small clusters (<1%)
        flagged_k = mat_df[mat_df["small_cluster_flag"]]["K"].tolist()
        if flagged_k:
            ax.set_title(f"{mat_key} (Ward) — Cluster Size Distribution (*Small cluster <1% at K={flagged_k})", fontsize=11, fontweight='bold')
        else:
            ax.set_title(f"{mat_key} (Ward) — Cluster Size Proportion Distribution Across K", fontsize=11, fontweight='bold')
            
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Proportion of Cohort")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"K={k}" for k in K_RANGE])
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis='y', linestyle="--", alpha=0.5)
        
    plt.suptitle("Hierarchical Ward Linkage Cluster Size Proportions Across K=2..7", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "hierarchical_cluster_sizes.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved cluster size plot: {plot_path}")

def generate_pca_projection_plots(matrix_data):
    """Generates 2D PCA post-hoc visual projection scatter plots for Primary Ward linkage solutions."""
    for mat_key, data in matrix_data.items():
        X = data["X"]
        
        # Fit 2D PCA strictly for visual coordinate projection
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        X_pca = pca.fit_transform(X)
        var1, var2 = pca.explained_variance_ratio_ * 100
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        axes = axes.flatten()
        
        for idx, K in enumerate(K_RANGE):
            ax = axes[idx]
            
            # Fit Ward model
            model = AgglomerativeClustering(n_clusters=K, linkage='ward', metric='euclidean')
            labels = model.fit_predict(X)
            
            sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10",
                            ax=ax, s=25, alpha=0.7, legend="full")
            
            ax.set_title(f"Ward Linkage — K={K}", fontsize=11, fontweight='bold')
            ax.set_xlabel(f"PC1 ({var1:.1f}% var)")
            ax.set_ylabel(f"PC2 ({var2:.1f}% var)")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(title="Cluster", fontsize=8, title_fontsize=9)
            
        plt.suptitle(f"{mat_key} — Post-Hoc 2D PCA Projections of Ward Hierarchical Clustering (Visualization Only)",
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f"hierarchical_pca_projections_{mat_key}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved PCA projection plot: {plot_path}")

if __name__ == "__main__":
    run_experiment()
