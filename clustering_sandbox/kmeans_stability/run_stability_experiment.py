"""
clustering_sandbox/kmeans_stability/run_stability_experiment.py
================================================================
Exploratory K-Means Stability Analysis (Experiment 2).

Evaluates partition-level stability (ARI), cluster-level stability (Jaccard),
and co-clustering consensus across K = 2..10 for four candidate standardized feature matrices:
  1. A_RAW_scaled (4,482 x 19)
  2. A_LOG_scaled (4,482 x 19)
  3. B_RAW_scaled (967 x 24)
  4. B_LOG_scaled (967 x 24)

RESAMPLING DESIGN:
  - Subsampling fraction: 80% without replacement
  - Repetitions: 200 per K per matrix (7,200 resampled fits total)
  - Reference fit: Full-data K-Means (K=2..10, n_init=25, seed=42)
  - Label-invariant comparison: Adjusted Rand Index (ARI) on sampled subset
  - Cluster matching: Max Jaccard overlap per reference cluster

STRICT METHODOLOGICAL BOUNDARIES:
  - Quantitative evidence only.
  - NO optimal K selected.
  - NO subjective labels ('good', 'weak', 'best') assigned.
  - NO Tridosha / Prakriti mapping performed.
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
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns


# Set plot styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


REF_SEED = 42
REF_N_INIT = 25
SUB_SEED_BASE = 10000
SUB_FRACTION = 0.80
N_REPETITIONS = 200
K_RANGE = list(range(2, 11))


_SHARED_X = None
_SHARED_REF_LABELS = None


def _init_worker(X, ref_labels):
    global _SHARED_X, _SHARED_REF_LABELS
    _SHARED_X = X
    _SHARED_REF_LABELS = ref_labels


def _compute_single_resample(args):
    """
    Worker function for parallel resampling execution.
    args: (sub_size, rep_idx, k)
    """
    sub_size, rep_idx, k = args
    X = _SHARED_X
    ref_labels = _SHARED_REF_LABELS
    n_samples = len(X)
    
    # Subsample 80% without replacement
    rng = np.random.RandomState(SUB_SEED_BASE + rep_idx)
    sub_idx = rng.choice(n_samples, size=sub_size, replace=False)
    
    sub_X = X[sub_idx]
    sub_ref_labels = ref_labels[sub_idx]
    
    # Fit resample K-Means
    sub_km = KMeans(
        n_clusters=k,
        random_state=SUB_SEED_BASE + 50000 + rep_idx,
        n_init=10
    ).fit(sub_X)
    
    sub_res_labels = sub_km.labels_
    
    # 1. Adjusted Rand Index (label-permutation invariant)
    ari = float(adjusted_rand_score(sub_ref_labels, sub_res_labels))
    
    # 2. Cluster-level Jaccard overlap
    jaccards = {}
    for c in range(k):
        ref_c_mask = (sub_ref_labels == c)
        ref_c_count = np.sum(ref_c_mask)
        
        if ref_c_count == 0:
            jaccards[c] = 0.0
            continue
            
        max_jaccard = 0.0
        for c_prime in range(k):
            res_c_mask = (sub_res_labels == c_prime)
            intersection = np.sum(ref_c_mask & res_c_mask)
            union = np.sum(ref_c_mask | res_c_mask)
            jaccard = float(intersection / union) if union > 0 else 0.0
            if jaccard > max_jaccard:
                max_jaccard = jaccard
        jaccards[c] = max_jaccard
        
    return ari, jaccards, sub_idx, sub_res_labels


def run_stability_sweep_for_matrix(
    name: str,
    df_scaled: pd.DataFrame,
    df_cohort_source: pd.DataFrame,
    max_workers: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Perform 200 stability resamples for each K=2..10 on a single matrix.
    """
    n_samples, n_features = df_scaled.shape
    X = df_scaled.values
    sub_size = int(np.floor(SUB_FRACTION * n_samples))

    metrics_records = []
    cluster_jaccard_records = []
    consensus_records = []

    for k in K_RANGE:
        t_k0 = time.time()
        print(f"    -> K={k:2d}... ", end="", flush=True)

        # Reference fit on full dataset
        ref_km = KMeans(
            n_clusters=k,
            random_state=REF_SEED,
            n_init=REF_N_INIT
        ).fit(X)
        ref_labels = ref_km.labels_

        # Reference cluster size diagnostics
        ref_counts = np.bincount(ref_labels, minlength=k)
        ref_props = ref_counts / n_samples
        min_prop = float(np.min(ref_props))
        max_prop = float(np.max(ref_props))
        ratio_max_min = float(max_prop / min_prop) if min_prop > 0 else np.nan
        std_props = float(np.std(ref_props))
        
        # Normalized Entropy: H_norm = -sum(p * ln(p)) / ln(K)
        non_zero_p = ref_props[ref_props > 0]
        norm_entropy = float(-np.sum(non_zero_p * np.log(non_zero_p)) / np.log(k))

        # Parallel resampling execution with worker memory initialization
        worker_args = [(sub_size, rep, k) for rep in range(N_REPETITIONS)]
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(X, ref_labels)
        ) as executor:
            resample_results = list(executor.map(_compute_single_resample, worker_args))

        aris = [r[0] for r in resample_results]
        jaccard_dicts = [r[1] for r in resample_results]

        # Consolidate partition-level ARI metrics
        mean_ari = float(np.mean(aris))
        median_ari = float(np.median(aris))
        std_ari = float(np.std(aris))
        min_ari = float(np.min(aris))
        max_ari = float(np.max(aris))
        p5_ari = float(np.percentile(aris, 5))
        p25_ari = float(np.percentile(aris, 25))
        p75_ari = float(np.percentile(aris, 75))
        p95_ari = float(np.percentile(aris, 95))

        # Consolidate cluster-level Jaccard stability
        cluster_jaccards_overall = []
        for c in range(k):
            c_jaccards = [j_dict[c] for j_dict in jaccard_dicts]
            cluster_jaccards_overall.extend(c_jaccards)
            
            c_mean = float(np.mean(c_jaccards))
            c_median = float(np.median(c_jaccards))
            c_std = float(np.std(c_jaccards))
            c_min = float(np.min(c_jaccards))
            c_max = float(np.max(c_jaccards))
            c_rec_75 = float(np.mean([j >= 0.75 for j in c_jaccards]))
            c_rec_60 = float(np.mean([j >= 0.60 for j in c_jaccards]))

            cluster_jaccard_records.append({
                "matrix_name": name,
                "K": k,
                "cluster_id": c,
                "ref_cluster_size": int(ref_counts[c]),
                "ref_cluster_proportion": float(ref_props[c]),
                "mean_jaccard": c_mean,
                "median_jaccard": c_median,
                "std_jaccard": c_std,
                "min_jaccard": c_min,
                "max_jaccard": c_max,
                "recovery_rate_ge_075": c_rec_75,
                "recovery_rate_ge_060": c_rec_60
            })

        mean_jaccard_overall = float(np.mean(cluster_jaccards_overall))
        median_jaccard_overall = float(np.median(cluster_jaccards_overall))

        # Consensus Co-Clustering Analysis
        # Track pairwise co-memberships and co-occurrences using sub-sampling index
        if n_samples <= 1000:
            # Full co-occurrence matrix for N=967
            co_occur = np.zeros((n_samples, n_samples), dtype=np.int32)
            co_member = np.zeros((n_samples, n_samples), dtype=np.int32)

            for _, _, sub_idx, sub_labels in resample_results:
                # Add co-occurrences
                ix_grid = np.ix_(sub_idx, sub_idx)
                co_occur[ix_grid] += 1
                
                # Add co-memberships
                same_label = (sub_labels[:, None] == sub_labels[None, :])
                co_member[ix_grid] += same_label.astype(np.int32)

            # Extract upper triangle (i < j) where co_occur > 0
            tri_u = np.triu_indices(n_samples, k=1)
            valid = co_occur[tri_u] > 0
            consensus_vals = co_member[tri_u][valid] / co_occur[tri_u][valid]
        else:
            # Efficient sampling of 100,000 random pairs for N=4,482
            pair_rng = np.random.RandomState(4242)
            pair_i = pair_rng.randint(0, n_samples, size=100000)
            pair_j = pair_rng.randint(0, n_samples, size=100000)
            valid_pairs = pair_i != pair_j
            p_i = pair_i[valid_pairs]
            p_j = pair_j[valid_pairs]

            co_occur_counts = np.zeros(len(p_i), dtype=np.int32)
            co_member_counts = np.zeros(len(p_i), dtype=np.int32)

            for _, _, sub_idx, sub_labels in resample_results:
                lookup_pos = np.full(n_samples, -1, dtype=np.int32)
                lookup_pos[sub_idx] = np.arange(len(sub_idx))
                
                pos_i = lookup_pos[p_i]
                pos_j = lookup_pos[p_j]
                
                both_in = (pos_i >= 0) & (pos_j >= 0)
                co_occur_counts[both_in] += 1
                
                if np.any(both_in):
                    same_cl = (sub_labels[pos_i[both_in]] == sub_labels[pos_j[both_in]])
                    co_member_counts[both_in] += same_cl.astype(np.int32)

            valid = co_occur_counts > 0
            consensus_vals = co_member_counts[valid] / co_occur_counts[valid]

        # Proportion of Ambiguous Clustering (PAC): % of pairs with S_ij in (0.10, 0.90)
        pac_01_09 = float(np.mean((consensus_vals > 0.10) & (consensus_vals < 0.90)))
        pac_02_08 = float(np.mean((consensus_vals > 0.20) & (consensus_vals < 0.80)))

        # Consensus AUC (Empirical CDF AUC)
        sorted_cons = np.sort(consensus_vals)
        cdf_y = np.linspace(0, 1, len(sorted_cons))
        consensus_auc = float(np.trapz(cdf_y, sorted_cons))

        consensus_records.append({
            "matrix_name": name,
            "K": k,
            "pac_01_09": pac_01_09,
            "pac_02_08": pac_02_08,
            "consensus_auc": consensus_auc,
            "mean_consensus": float(np.mean(consensus_vals)),
            "median_consensus": float(np.median(consensus_vals))
        })

        metrics_records.append({
            "matrix_name": name,
            "N": n_samples,
            "number_of_features": n_features,
            "K": k,
            "resampling_fraction": SUB_FRACTION,
            "number_of_resamples": N_REPETITIONS,
            "mean_ARI": mean_ari,
            "median_ARI": median_ari,
            "SD_ARI": std_ari,
            "min_ARI": min_ari,
            "max_ARI": max_ari,
            "ARI_5th_percentile": p5_ari,
            "ARI_25th_percentile": p25_ari,
            "ARI_75th_percentile": p75_ari,
            "ARI_95th_percentile": p95_ari,
            "mean_Jaccard_overall": mean_jaccard_overall,
            "median_Jaccard_overall": median_jaccard_overall,
            "pac_01_09": pac_01_09,
            "min_cluster_proportion": min_prop,
            "max_cluster_proportion": max_prop,
            "max_min_cluster_ratio": ratio_max_min,
            "sd_cluster_proportions": std_props,
            "normalized_entropy": norm_entropy,
            "seed_reference": REF_SEED,
            "seed_base_resample": SUB_SEED_BASE,
            "sklearn_version": sklearn.__version__
        })

        print(f"done ({time.time() - t_k0:.2f}s)")

    metrics_df = pd.DataFrame(metrics_records)
    jaccard_df = pd.DataFrame(cluster_jaccard_records)
    consensus_df = pd.DataFrame(consensus_records)

    summary_dict = {
        "matrix_name": name,
        "n_samples": n_samples,
        "n_features": n_features,
        "stability_by_k": metrics_records
    }

    return metrics_df, jaccard_df, consensus_df, summary_dict


def generate_stability_plots(
    all_metrics: dict[str, pd.DataFrame],
    all_jaccards: dict[str, pd.DataFrame],
    all_consensus: dict[str, pd.DataFrame],
    plots_dir: Path
) -> None:
    """
    Generate non-evaluative stability diagnostic plots:
      1. Individual ARI Distribution vs K plots (mean/median line + 5th-95th percentile shaded bands)
      2. Combined Cross-Matrix ARI Comparison plot
      3. Cluster-level Jaccard Stability summary plot
      4. Consensus PAC / CDF diagnostic plots
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    palette = {"A_RAW": "#1f77b4", "A_LOG": "#2ca02c", "B_RAW": "#ff7f0e", "B_LOG": "#d62728"}

    # ── 1. Individual ARI plots per matrix ────────────────────────────────────
    for name, m_df in all_metrics.items():
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ks = m_df["K"].values
        means = m_df["mean_ARI"].values
        medians = m_df["median_ARI"].values
        p5 = m_df["ARI_5th_percentile"].values
        p95 = m_df["ARI_95th_percentile"].values
        p25 = m_df["ARI_25th_percentile"].values
        p75 = m_df["ARI_75th_percentile"].values

        ax.fill_between(ks, p5, p95, color=palette[name], alpha=0.15, label="5th–95th Percentile Band")
        ax.fill_between(ks, p25, p75, color=palette[name], alpha=0.30, label="25th–75th Percentile Band")
        ax.plot(ks, means, "o-", color=palette[name], linewidth=2.2, label="Mean ARI")
        ax.plot(ks, medians, "s--", color="#333333", linewidth=1.5, label="Median ARI")

        ax.set_title(f"Adjusted Rand Index (ARI) Stability across 200 Subsamples — {name}", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Number of Clusters (K)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Adjusted Rand Index (ARI)", fontsize=10, fontweight="bold")
        ax.set_xticks(ks)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower left", frameon=True)

        fig.tight_layout()
        plt.savefig(plots_dir / f"{name}_ari_vs_k.png")
        plt.close()

    # ── 2. Combined Cross-Matrix ARI Comparison ──────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    for name, m_df in all_metrics.items():
        ax.plot(m_df["K"], m_df["mean_ARI"], "o-", color=palette[name], linewidth=2.2, markersize=6, label=f"{name} (Mean ARI)")

    ax.set_title("Cross-Matrix ARI Partition Stability Comparison (80% Subsampling, 200 Reps)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Clusters (K)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Adjusted Rand Index (ARI)", fontsize=11, fontweight="bold")
    ax.set_xticks(K_RANGE)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", frameon=True)

    fig.tight_layout()
    plt.savefig(plots_dir / "cross_matrix_ari_comparison.png")
    plt.close()

    # ── 3. Cluster-Level Jaccard Stability Summary ───────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    for name, j_df in all_jaccards.items():
        # Group by K to get mean of cluster-level Jaccard
        k_means = j_df.groupby("K")["mean_jaccard"].mean()
        ax.plot(k_means.index, k_means.values, "s-", color=palette[name], linewidth=2.0, markersize=6, label=f"{name} (Mean Cluster Jaccard)")

    ax.set_title("Overall Mean Cluster-Level Jaccard Stability across K=2..10", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Clusters (K)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Cluster Jaccard Similarity", fontsize=11, fontweight="bold")
    ax.set_xticks(K_RANGE)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower left", frameon=True)

    fig.tight_layout()
    plt.savefig(plots_dir / "jaccard_stability_summary.png")
    plt.close()

    # ── 4. Consensus PAC (Proportion of Ambiguous Clustering) ────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    for name, c_df in all_consensus.items():
        ax.plot(c_df["K"], c_df["pac_01_09"], "d-", color=palette[name], linewidth=2.0, markersize=6, label=f"{name} (PAC [0.10, 0.90])")

    ax.set_title("Proportion of Ambiguous Clustering (PAC) Diagnostic across K=2..10\n(Lower PAC indicates sharper cluster boundaries)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Clusters (K)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Proportion of Ambiguous Pairs (PAC)", fontsize=11, fontweight="bold")
    ax.set_xticks(K_RANGE)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    plt.savefig(plots_dir / "consensus_cdf_plots.png")
    plt.close()


def run_kmeans_stability_pipeline(base_dir: Path) -> dict:
    """
    Execute full exploratory K-Means stability experiment across all 4 matrices.
    """
    scaled_dir = base_dir / "output" / "scaled_matrices"
    cohort_a_path = base_dir / "output" / "analysis_cohort_a_broad.csv"
    cohort_b_path = base_dir / "output" / "analysis_cohort_b_fasting.csv"

    exp2_dir = base_dir / "clustering_sandbox" / "kmeans_stability"
    metrics_dir = exp2_dir / "metrics"
    jaccard_dir = exp2_dir / "cluster_stability"
    consensus_dir = exp2_dir / "consensus"
    plots_dir = exp2_dir / "plots"
    metadata_dir = exp2_dir / "metadata"

    for d in [metrics_dir, jaccard_dir, consensus_dir, plots_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    matrices = {
        "A_RAW": (scaled_dir / "A_RAW_scaled.csv", cohort_a_path),
        "A_LOG": (scaled_dir / "A_LOG_scaled.csv", cohort_a_path),
        "B_RAW": (scaled_dir / "B_RAW_scaled.csv", cohort_b_path),
        "B_LOG": (scaled_dir / "B_LOG_scaled.csv", cohort_b_path),
    }

    all_metrics = {}
    all_jaccards = {}
    all_consensus = {}
    all_summaries = {}

    print("=================================================================")
    print("  EXPLORATORY CLUSTERING SANDBOX — EXPERIMENT 2 (STABILITY)")
    print("=================================================================")

    start_time = time.time()

    for name, (scaled_file, cohort_file) in matrices.items():
        df_scaled = pd.read_csv(scaled_file)
        df_cohort = pd.read_csv(cohort_file)

        print(f"\n  [Running Stability Subsampling: 200 reps x 9 K] -> {name} ({df_scaled.shape[0]:,} rows x {df_scaled.shape[1]} features)")

        t_mat0 = time.time()
        m_df, j_df, c_df, summary = run_stability_sweep_for_matrix(
            name, df_scaled, df_cohort, max_workers=4
        )
        t_mat1 = time.time()

        # Save individual CSVs
        m_df.to_csv(metrics_dir / f"{name}_stability_metrics.csv", index=False)
        j_df.to_csv(jaccard_dir / f"{name}_cluster_jaccard.csv", index=False)
        c_df.to_csv(consensus_dir / f"{name}_consensus_diagnostics.csv", index=False)

        all_metrics[name] = m_df
        all_jaccards[name] = j_df
        all_consensus[name] = c_df
        all_summaries[name] = summary

        print(f"    - Completed 1,800 resampled fits in {t_mat1 - t_mat0:.1f}s")
        print(f"    - K=2 Mean ARI: {m_df.loc[m_df['K']==2, 'mean_ARI'].values[0]:.4f}, K=3 Mean ARI: {m_df.loc[m_df['K']==3, 'mean_ARI'].values[0]:.4f}")

    # Create Cross-Matrix Summary CSV
    cross_matrix_df = pd.concat(list(all_metrics.values()), ignore_index=True)
    cross_matrix_csv = metrics_dir / "cross_matrix_stability_summary.csv"
    cross_matrix_df.to_csv(cross_matrix_csv, index=False)

    # Generate Plots
    generate_stability_plots(all_metrics, all_jaccards, all_consensus, plots_dir)

    elapsed = time.time() - start_time

    # Create Metadata JSON
    metadata = {
        "experiment": "EXPLORATORY_CLUSTERING_SANDBOX_KMEANS_STABILITY",
        "status": "EXPERIMENTAL_SANDBOX_ONLY",
        "description": "Exploratory partition stability (ARI), cluster Jaccard stability, and consensus diagnostics across 80% participant subsampling (200 reps).",
        "formal_pipeline_status": "NOT_MERGED_TO_FORMAL_PIPELINE",
        "methodological_constraints_confirmed": {
            "clustering_input_space": "FULL_STANDARDIZED_FEATURE_SPACE",
            "pca_used_as_clustering_input": False,
            "optimal_k_selected": False,
            "subjective_quality_labels_assigned": False,
            "tridosha_mapping_performed": False,
            "formal_decision_log_modified": False
        },
        "resampling_design": {
            "subsampling_fraction": SUB_FRACTION,
            "number_of_repetitions_per_k": N_REPETITIONS,
            "total_resampled_fits_completed": len(matrices) * len(K_RANGE) * N_REPETITIONS,
            "k_range": K_RANGE,
            "reference_seed": REF_SEED,
            "reference_n_init": REF_N_INIT,
            "subsample_seed_base": SUB_SEED_BASE,
            "sklearn_version": sklearn.__version__,
            "total_runtime_seconds": float(elapsed)
        },
        "matrices_analyzed": list(matrices.keys()),
        "quantitative_summary_by_matrix": {
            name: m_df[["K", "mean_ARI", "median_ARI", "SD_ARI", "mean_Jaccard_overall", "pac_01_09"]].to_dict(orient="records")
            for name, m_df in all_metrics.items()
        }
    }

    meta_file = metadata_dir / "kmeans_stability_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Create README.md inside clustering_sandbox/kmeans_stability/
    readme_content = f"""# K-Means Stability Analysis Sandbox (Experiment 2)

## Research Question & Purpose

This directory contains quantitative stability evidence for K-Means partitions ($K = 2 \dots 10$) under participant-level sample perturbations.

The core methodological question addressed is:
> *Are the cluster partitions reproducible when the participant sample is perturbed via 80% subsampling without replacement?*

---

## Resampling & Experimental Design

- **Subsampling Fraction**: 80% of participants ($0.80 \\times N$) without replacement per resample.
- **Repetitions**: 200 repetitions per $K$ per matrix (**7,200 total resampled fits** across 4 matrices $\\times$ 9 $K$ values).
- **Reference Fit**: Full-data K-Means fit ($N$ rows) with `random_state = {REF_SEED}`, `n_init = {REF_N_INIT}`.
- **Partition Stability Metric**: **Adjusted Rand Index (ARI)** computed between the reference partition restricted to sampled participants and the resampled partition. Label-permutation invariant.
- **Cluster-Level Metric**: **Jaccard Similarity** calculated via maximal overlap matching per reference cluster.
- **Consensus Metric**: **Proportion of Ambiguous Clustering (PAC)** measuring the fraction of sample pairs with co-clustering probability in $(0.10, 0.90)$.
- **Software**: Scikit-Learn `{sklearn.__version__}`.

---

## Methodological Boundaries Confirmed

- ⚠️ **No Optimal K Selected**: No threshold or composite score was used to pick a "winning" $K$.
- ⚠️ **No Subjective Labels**: Qualitative descriptors (e.g. "strong", "weak", "good", "poor") were **not** assigned.
- ⚠️ **Full Standardized Space**: Clustering executed strictly on full 19D / 24D standardized feature space (PCA was **not** used as input).
- ⚠️ **Preserved Cohort Separation**: Cohort A ($N=4,482$) and Cohort B ($N=967$), as well as RAW and LOG representations, were kept 100% separate.

---

*Generated by `clustering_sandbox/kmeans_stability/run_stability_experiment.py`*
"""

    readme_file = exp2_dir / "README.md"
    with open(readme_file, "w") as f:
        f.write(readme_content)

    print(f"\n  [SAVED METADATA] -> {meta_file.relative_to(base_dir)}")
    print(f"  [SAVED README]   -> {readme_file.relative_to(base_dir)}")
    print(f"  [TOTAL RUNTIME]  -> {elapsed:.1f} seconds")
    print("=================================================================")

    return metadata


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    run_kmeans_stability_pipeline(base_dir)
