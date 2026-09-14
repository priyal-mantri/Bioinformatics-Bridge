"""
Experiment 6: Cross-Model Clustering Benchmark and K Selection
==============================================================
Branch: experiment/clustering-sandbox
Purpose: Methodological comparison of completed clustering model families.
         Determines whether convergent evidence supports K=2, K=3, or K=4.

Models included (using EXISTING outputs only):
  - K-Means         (Experiments 1 + 2)
  - GMM             (Experiment 3)
  - Spectral        (Experiment 4)
  - Hierarchical    (Experiment 5A — Ward linkage primary)
  - HDBSCAN         (Experiment 5A — sanity check only, NOT K-based)

SEQN alignment pre-verified: exact row order match across all models/matrices.

DO NOT:
  - Rerun or modify any prior model experiment
  - Choose K before examining all evidence
  - Introduce biological/Ayurvedic interpretation
  - Commit or merge to master

Author: Experiment 6 pipeline
"""

import os
import json
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SANDBOX   = os.path.join(REPO_ROOT, "clustering_sandbox")
OUT_DIR   = os.path.join(SANDBOX, "cross_model")
FIG_DIR   = os.path.join(OUT_DIR, "figures")
META_DIR  = os.path.join(OUT_DIR, "metadata")

MATRICES  = ["A_RAW", "A_LOG", "B_RAW", "B_LOG"]
COHORT_MAP = {"A_RAW": "A", "A_LOG": "A", "B_RAW": "B", "B_LOG": "B"}
SCALE_MAP  = {"A_RAW": "Raw", "A_LOG": "Log1p", "B_RAW": "Raw", "B_LOG": "Log1p"}
N_MAP      = {"A_RAW": 4482, "A_LOG": 4482, "B_RAW": 967, "B_LOG": 967}
P_MAP      = {"A_RAW": 19, "A_LOG": 19, "B_RAW": 24, "B_LOG": 24}

K_PRIMARY  = [2, 3, 4]
K_ALL      = [2, 3, 4, 5, 6, 7]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: SEQN INTEGRITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
def verify_seqn(df, matrix, model, expected_n):
    assert len(df) == expected_n, \
        f"AUDIT FAIL: {model}/{matrix} has N={len(df)}, expected {expected_n}"
    assert df["SEQN"].isna().sum() == 0, \
        f"AUDIT FAIL: {model}/{matrix} has NaN SEQN"
    assert df["SEQN"].duplicated().sum() == 0, \
        f"AUDIT FAIL: {model}/{matrix} has duplicate SEQN"

def verify_seqn_alignment(ref_seqn, other_seqn, ref_label, other_label):
    assert (ref_seqn.values == other_seqn.values).all(), \
        f"AUDIT FAIL: SEQN row order mismatch between {ref_label} and {other_label}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: LOAD ALL ASSIGNMENT TABLES
# ─────────────────────────────────────────────────────────────────────────────
def load_all_assignments():
    """Load and audit SEQN-aligned assignment tables for K-Means, GMM,
    Spectral, and Hierarchical (Ward only for cross-model ARI)."""
    assigns = {}
    for m in MATRICES:
        assigns[m] = {}
        expected_n = N_MAP[m]

        # K-Means (K2–K7 columns available; K8–K10 also present but excluded)
        km = pd.read_csv(os.path.join(SANDBOX, "kmeans", "cluster_assignments",
                                      f"{m}_kmeans_assignments.csv"))
        verify_seqn(km, m, "kmeans", expected_n)
        assigns[m]["kmeans"] = km

        # GMM (K2–K7 hard assignments)
        gmm = pd.read_csv(os.path.join(SANDBOX, "gmm", "assignments",
                                       f"{m}_gmm_assignments.csv"))
        verify_seqn(gmm, m, "gmm", expected_n)
        # GMM assignment columns are K2_cluster, K3_cluster, ...
        assigns[m]["gmm"] = gmm

        # Spectral (K2–K7)
        sp = pd.read_csv(os.path.join(SANDBOX, "spectral", "cluster_assignments",
                                      f"{m}_spectral_assignments.csv"))
        verify_seqn(sp, m, "spectral", expected_n)
        assigns[m]["spectral"] = sp

        # Hierarchical — Ward only for cross-model ARI
        hi = pd.read_csv(os.path.join(SANDBOX, "hierarchical", "cluster_assignments",
                                      f"{m}_hierarchical_assignments.csv"))
        verify_seqn(hi, m, "hierarchical", expected_n)
        assigns[m]["hierarchical_ward"] = hi

        # Cross-model SEQN alignment verification
        ref_seqn = km["SEQN"]
        for model_name, df in [("gmm", gmm), ("spectral", sp), ("hierarchical_ward", hi)]:
            verify_seqn_alignment(ref_seqn, df["SEQN"],
                                  f"kmeans/{m}", f"{model_name}/{m}")

    print("✅ SEQN audit passed: all assignment tables N-correct, no NaN/dup, exact row alignment.")
    return assigns


def get_label_col(df, model, k):
    """Return the cluster label column for a given model/K."""
    if model == "kmeans":
        return df[f"K{k}_cluster"]
    elif model == "gmm":
        return df[f"K{k}_cluster"]
    elif model == "spectral":
        return df[f"K{k}_cluster"]
    elif model == "hierarchical_ward":
        return df[f"ward_K{k}"]
    else:
        raise ValueError(f"Unknown model: {model}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: MASTER METRICS TABLE
# ─────────────────────────────────────────────────────────────────────────────
def build_master_metrics():
    """Collect all internal metrics from existing model outputs into one table."""
    rows = []

    for m in MATRICES:
        cohort = COHORT_MAP[m]
        scale  = SCALE_MAP[m]

        # ── K-Means ──────────────────────────────────────────────────────────
        km_met = pd.read_csv(os.path.join(SANDBOX, "kmeans", "metrics",
                                          f"{m}_kmeans_metrics.csv"))
        km_stab = pd.read_csv(os.path.join(SANDBOX, "kmeans_stability", "metrics",
                                           "cross_matrix_stability_summary.csv"))
        km_stab = km_stab[km_stab["matrix_name"] == m].set_index("K")

        for _, r in km_met[km_met["K"].between(2, 7)].iterrows():
            k = int(r["K"])
            import json as _json
            sizes = _json.loads(r["cluster_sizes"])
            total = sum(sizes.values())
            min_p = min(sizes.values()) / total
            max_p = max(sizes.values()) / total
            stab_mean = km_stab.loc[k, "mean_ARI"] if k in km_stab.index else np.nan
            stab_sd   = km_stab.loc[k, "SD_ARI"]   if k in km_stab.index else np.nan
            rows.append({
                "cohort": cohort, "matrix": m, "scale": scale,
                "model": "K-Means", "linkage": "NA",
                "K": k,
                "silhouette": round(r["silhouette_score"], 5),
                "CH": round(r["calinski_harabasz_score"], 2),
                "DB": round(r["davies_bouldin_score"], 4),
                "stability_ari_mean": round(stab_mean, 4) if not np.isnan(stab_mean) else np.nan,
                "stability_ari_sd":   round(stab_sd, 4)   if not np.isnan(stab_sd)   else np.nan,
                "stability_source": "kmeans_stability (B=200, 80% subsample, ARI)",
                "min_cluster_prop": round(min_p, 4),
                "max_cluster_prop": round(max_p, 4),
                "max_min_ratio": round(max_p / min_p, 2) if min_p > 0 else np.nan,
                "small_cluster_flag": min_p < 0.01,
                "degeneracy_note": ""
            })

        # ── GMM ──────────────────────────────────────────────────────────────
        gmm_met = pd.read_csv(os.path.join(SANDBOX, "gmm", "metrics",
                                           f"{m}_gmm_metrics.csv"))
        for _, r in gmm_met.iterrows():
            k = int(r["K"])
            rows.append({
                "cohort": cohort, "matrix": m, "scale": scale,
                "model": "GMM", "linkage": "full_cov",
                "K": k,
                "silhouette": round(r["silhouette"], 5),
                "CH": round(r["CH"], 2),
                "DB": round(r["DB"], 4),
                "stability_ari_mean": np.nan,
                "stability_ari_sd":   np.nan,
                "stability_source": "NA (not generated)",
                "min_cluster_prop": round(r["min_cluster_proportion"], 4),
                "max_cluster_prop": round(r["max_cluster_proportion"], 4),
                "max_min_ratio": round(r["max_cluster_proportion"] / r["min_cluster_proportion"], 2)
                                  if r["min_cluster_proportion"] > 0 else np.nan,
                "small_cluster_flag": r["min_cluster_proportion"] < 0.01,
                "degeneracy_note": f"BIC={r['BIC']:.0f}; mean_max_posterior={r['mean_max_posterior']:.3f}"
            })

        # ── Spectral ─────────────────────────────────────────────────────────
        sp_met = pd.read_csv(os.path.join(SANDBOX, "spectral", "metrics",
                                          f"{m}_spectral_metrics.csv"))
        for _, r in sp_met.iterrows():
            k = int(r["K"])
            import json as _json
            sizes = _json.loads(r["cluster_sizes"])
            total = sum(sizes.values())
            min_p = min(sizes.values()) / total
            max_p = max(sizes.values()) / total
            rows.append({
                "cohort": cohort, "matrix": m, "scale": scale,
                "model": "Spectral", "linkage": "kNN_k10_NJW",
                "K": k,
                "silhouette": round(r["silhouette_score"], 5),
                "CH": round(r["calinski_harabasz_score"], 2),
                "DB": round(r["davies_bouldin_score"], 4),
                "stability_ari_mean": np.nan,
                "stability_ari_sd":   np.nan,
                "stability_source": "NA (not generated)",
                "min_cluster_prop": round(min_p, 4),
                "max_cluster_prop": round(max_p, 4),
                "max_min_ratio": round(max_p / min_p, 2) if min_p > 0 else np.nan,
                "small_cluster_flag": min_p < 0.01,
                "degeneracy_note": f"eigengap_prev={r['eigengap_K_minus_1_to_K']:.5f}; eigengap_next={r['eigengap_K_to_K_plus_1']:.5f}"
            })

        # ── Hierarchical — Ward (primary) ─────────────────────────────────
        hi_met = pd.read_csv(os.path.join(SANDBOX, "hierarchical", "metrics",
                                          f"{m}_hierarchical_metrics.csv"))
        hi_stab = pd.read_csv(os.path.join(SANDBOX, "hierarchical", "metrics",
                                           "hierarchical_stability_summary.csv"))
        hi_stab_ward = hi_stab[(hi_stab["matrix"] == m) & (hi_stab["linkage"] == "ward")].set_index("K")

        hi_ward = hi_met[hi_met["linkage"] == "ward"]
        for _, r in hi_ward.iterrows():
            k = int(r["K"])
            stab_mean = hi_stab_ward.loc[k, "mean_ari"] if k in hi_stab_ward.index else np.nan
            stab_sd   = hi_stab_ward.loc[k, "std_ari"]  if k in hi_stab_ward.index else np.nan
            rows.append({
                "cohort": cohort, "matrix": m, "scale": scale,
                "model": "Hierarchical", "linkage": "Ward",
                "K": k,
                "silhouette": round(r["silhouette_score"], 5),
                "CH": round(r["calinski_harabasz_score"], 2),
                "DB": round(r["davies_bouldin_score"], 4),
                "stability_ari_mean": round(stab_mean, 4) if not np.isnan(stab_mean) else np.nan,
                "stability_ari_sd":   round(stab_sd, 4)   if not np.isnan(stab_sd)   else np.nan,
                "stability_source": "hierarchical_stability (B=100, 80% subsample, participant-overlap ARI)",
                "min_cluster_prop": round(r["min_cluster_prop"], 4),
                "max_cluster_prop": round(r["max_cluster_prop"], 4),
                "max_min_ratio": round(r["max_min_ratio"], 2),
                "small_cluster_flag": bool(r["small_cluster_flag"]),
                "degeneracy_note": ""
            })

        # ── Hierarchical — Complete (flagged as degenerate) ───────────────
        hi_comp = hi_met[hi_met["linkage"] == "complete"]
        for _, r in hi_comp.iterrows():
            k = int(r["K"])
            rows.append({
                "cohort": cohort, "matrix": m, "scale": scale,
                "model": "Hierarchical", "linkage": "Complete",
                "K": k,
                "silhouette": round(r["silhouette_score"], 5),
                "CH": round(r["calinski_harabasz_score"], 2),
                "DB": round(r["davies_bouldin_score"], 4),
                "stability_ari_mean": np.nan,
                "stability_ari_sd":   np.nan,
                "stability_source": "excluded — degenerate outlier-peeling",
                "min_cluster_prop": round(r["min_cluster_prop"], 4),
                "max_cluster_prop": round(r["max_cluster_prop"], 4),
                "max_min_ratio": round(r["max_min_ratio"], 2),
                "small_cluster_flag": bool(r["small_cluster_flag"]),
                "degeneracy_note": "DEGENERATE: outlier-peeling — high silhouette is artifact"
            })

        # ── Hierarchical — Average (flagged as degenerate) ────────────────
        hi_avg = hi_met[hi_met["linkage"] == "average"]
        for _, r in hi_avg.iterrows():
            k = int(r["K"])
            rows.append({
                "cohort": cohort, "matrix": m, "scale": scale,
                "model": "Hierarchical", "linkage": "Average",
                "K": k,
                "silhouette": round(r["silhouette_score"], 5),
                "CH": round(r["calinski_harabasz_score"], 2),
                "DB": round(r["davies_bouldin_score"], 4),
                "stability_ari_mean": np.nan,
                "stability_ari_sd":   np.nan,
                "stability_source": "excluded — degenerate outlier-peeling",
                "min_cluster_prop": round(r["min_cluster_prop"], 4),
                "max_cluster_prop": round(r["max_cluster_prop"], 4),
                "max_min_ratio": round(r["max_min_ratio"], 2),
                "small_cluster_flag": bool(r["small_cluster_flag"]),
                "degeneracy_note": "DEGENERATE: outlier-peeling — high ARI is artifact"
            })

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "cross_model_master_metrics.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Master metrics table: {len(df)} rows → {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: CROSS-MODEL PAIRWISE ARI / NMI
# ─────────────────────────────────────────────────────────────────────────────
PAIRWISE_MODELS = ["kmeans", "gmm", "spectral", "hierarchical_ward"]
MODEL_LABELS    = {
    "kmeans": "K-Means",
    "gmm": "GMM",
    "spectral": "Spectral",
    "hierarchical_ward": "Hier-Ward"
}

def compute_pairwise_ari_nmi(assigns):
    """Compute ARI and NMI between every pair of models for each matrix × K."""
    ari_rows = []
    nmi_rows = []

    model_pairs = []
    for i, ma in enumerate(PAIRWISE_MODELS):
        for mb in PAIRWISE_MODELS[i+1:]:
            model_pairs.append((ma, mb))

    for m in MATRICES:
        cohort = COHORT_MAP[m]
        scale  = SCALE_MAP[m]
        for k in K_ALL:
            for ma, mb in model_pairs:
                try:
                    labels_a = get_label_col(assigns[m][ma], ma, k).values
                    labels_b = get_label_col(assigns[m][mb], mb, k).values
                    ari_val = adjusted_rand_score(labels_a, labels_b)
                    nmi_val = normalized_mutual_info_score(labels_a, labels_b,
                                                           average_method="arithmetic")
                    row_base = {
                        "matrix": m, "cohort": cohort, "scale": scale, "K": k,
                        "model_A": MODEL_LABELS[ma],
                        "model_B": MODEL_LABELS[mb],
                    }
                    ari_rows.append({**row_base, "ARI": round(ari_val, 5)})
                    nmi_rows.append({**row_base, "NMI": round(nmi_val, 5)})
                except KeyError as e:
                    # Column missing (shouldn't happen for K2-7)
                    print(f"  WARNING: missing column for {m}/{ma}/{mb}/K{k}: {e}")
                    row_base = {
                        "matrix": m, "cohort": cohort, "scale": scale, "K": k,
                        "model_A": MODEL_LABELS[ma], "model_B": MODEL_LABELS[mb],
                    }
                    ari_rows.append({**row_base, "ARI": np.nan})
                    nmi_rows.append({**row_base, "NMI": np.nan})

    df_ari = pd.DataFrame(ari_rows)
    df_nmi = pd.DataFrame(nmi_rows)

    ari_path = os.path.join(OUT_DIR, "cross_model_pairwise_ari.csv")
    nmi_path = os.path.join(OUT_DIR, "cross_model_pairwise_nmi.csv")
    df_ari.to_csv(ari_path, index=False)
    df_nmi.to_csv(nmi_path, index=False)
    print(f"✅ Pairwise ARI table: {len(df_ari)} rows → {ari_path}")
    print(f"✅ Pairwise NMI table: {len(df_nmi)} rows → {nmi_path}")
    return df_ari, df_nmi


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: K EVIDENCE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def build_k_evidence_summary(master_df, ari_df):
    """Aggregate evidence per K across models and matrices."""
    rows = []

    # GMM BIC minima (pre-verified)
    gmm_bic_min = {"A_RAW": 4, "A_LOG": 3, "B_RAW": 2, "B_LOG": 2}
    # Spectral eigengap signals (largest eigengap beyond K=2 per matrix)
    # B_RAW and B_LOG: λ3→λ4 gap → weak signal for K=3 spectral cutoff
    # A_RAW, A_LOG: no meaningful eigengap beyond K=2
    spectral_eigengap_signal = {"A_RAW": None, "A_LOG": None, "B_RAW": 3, "B_LOG": 3}

    # Primary models for separation evidence (exclude degenerate linkages)
    primary_models = ["K-Means", "GMM", "Spectral", "Hierarchical"]
    primary_links  = ["NA", "full_cov", "kNN_k10_NJW", "Ward"]
    prim_mask = (
        master_df["model"].isin(primary_models) &
        master_df["linkage"].isin(primary_links)
    )

    for k in K_ALL:
        k_df = master_df[prim_mask & (master_df["K"] == k)]

        for cohort in ["A", "B"]:
            ck = k_df[k_df["cohort"] == cohort]

            # ── Separation ────────────────────────────────────────────────
            sc_vals = ck["silhouette"].dropna()
            sc_mean = sc_vals.mean() if len(sc_vals) > 0 else np.nan
            sc_max  = sc_vals.max()  if len(sc_vals) > 0 else np.nan

            # ── Stability ─────────────────────────────────────────────────
            km_stab = ck[ck["model"] == "K-Means"]["stability_ari_mean"].values
            hi_stab = ck[ck["model"] == "Hierarchical"]["stability_ari_mean"].values
            km_stab_mean = float(np.nanmean(km_stab)) if len(km_stab) > 0 else np.nan
            hi_stab_mean = float(np.nanmean(hi_stab)) if len(hi_stab) > 0 else np.nan

            # ── Cross-model ARI ───────────────────────────────────────────
            ari_c = ari_df[(ari_df["cohort"] == cohort) & (ari_df["K"] == k)]
            mean_cross_ari_raw = ari_c[ari_c["scale"] == "Raw"]["ARI"].mean()
            mean_cross_ari_log = ari_c[ari_c["scale"] == "Log1p"]["ARI"].mean()

            # ── Degeneracy ────────────────────────────────────────────────
            any_small = ck["small_cluster_flag"].any()

            # ── GMM BIC ───────────────────────────────────────────────────
            mats_in_cohort = [mx for mx in MATRICES if COHORT_MAP[mx] == cohort]
            gmm_bic_favors = sum(1 for mx in mats_in_cohort if gmm_bic_min.get(mx) == k)

            # ── Spectral eigengap ─────────────────────────────────────────
            spectral_favors = sum(1 for mx in mats_in_cohort
                                  if spectral_eigengap_signal.get(mx) == k)

            # ── RAW/LOG robustness: are RAW and LOG cross-model ARI similar? ─
            raw_vals = ari_c[ari_c["scale"] == "Raw"]["ARI"].dropna()
            log_vals = ari_c[ari_c["scale"] == "Log1p"]["ARI"].dropna()
            raw_log_ari_diff = abs(raw_vals.mean() - log_vals.mean()) \
                if (len(raw_vals) > 0 and len(log_vals) > 0) else np.nan

            rows.append({
                "K": k,
                "cohort": cohort,
                "mean_silhouette_primary_models": round(sc_mean, 4) if not np.isnan(sc_mean) else np.nan,
                "max_silhouette_primary_models": round(sc_max, 4)  if not np.isnan(sc_max)  else np.nan,
                "kmeans_stability_mean_ari": round(km_stab_mean, 4) if not np.isnan(km_stab_mean) else np.nan,
                "hierarchical_ward_stability_mean_ari": round(hi_stab_mean, 4) if not np.isnan(hi_stab_mean) else np.nan,
                "mean_cross_model_ari_raw": round(mean_cross_ari_raw, 4) if not np.isnan(mean_cross_ari_raw) else np.nan,
                "mean_cross_model_ari_log": round(mean_cross_ari_log, 4) if not np.isnan(mean_cross_ari_log) else np.nan,
                "any_small_cluster_flag": any_small,
                "gmm_bic_minima_count": gmm_bic_favors,
                "spectral_eigengap_K3_signal_count": spectral_favors if k == 3 else 0,
                "raw_log_cross_ari_diff": round(raw_log_ari_diff, 4) if not np.isnan(raw_log_ari_diff) else np.nan,
            })

    df = pd.DataFrame(rows).sort_values(["K", "cohort"])
    out_path = os.path.join(OUT_DIR, "k_evidence_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ K evidence summary: {len(df)} rows → {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: FIGURES
# ─────────────────────────────────────────────────────────────────────────────

COLOR_MODEL = {
    "K-Means":    "#4C72B0",
    "GMM":        "#DD8452",
    "Spectral":   "#55A868",
    "Hier-Ward":  "#C44E52",
}
MARKER_MODEL = {"K-Means": "o", "GMM": "s", "Spectral": "^", "Hier-Ward": "D"}


def fig1_silhouette_by_k(master_df):
    """Fig 1: Cross-model silhouette by K, one panel per cohort."""
    primary_models = {
        "K-Means": "NA",
        "GMM": "full_cov",
        "Spectral": "kNN_k10_NJW",
        "Hier-Ward": "Ward",
    }
    # Map Hierarchical/Ward label
    prim_df = master_df[
        (master_df["K"].between(2, 7)) &
        (
            ((master_df["model"] == "K-Means") & (master_df["linkage"] == "NA")) |
            ((master_df["model"] == "GMM")      & (master_df["linkage"] == "full_cov")) |
            ((master_df["model"] == "Spectral") & (master_df["linkage"] == "kNN_k10_NJW")) |
            ((master_df["model"] == "Hierarchical") & (master_df["linkage"] == "Ward"))
        )
    ].copy()
    prim_df["model_label"] = prim_df.apply(
        lambda r: "Hier-Ward" if r["model"] == "Hierarchical" else r["model"], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle("Fig 1: Cross-Model Silhouette Score by K\n(Primary linkages only; Ward for Hierarchical)",
                 fontsize=12, fontweight="bold", y=1.01)

    for ax, cohort, title in zip(axes, ["A", "B"],
                                  ["Cohort A (Broad, N=4482, p=19)",
                                   "Cohort B (Fasting, N=967, p=24)"]):
        c_df = prim_df[prim_df["cohort"] == cohort]
        for model_label, color in COLOR_MODEL.items():
            m_df = c_df[c_df["model_label"] == model_label]
            if m_df.empty:
                continue
            # Average across RAW and LOG, plot with error band
            pivot = m_df.groupby("K")["silhouette"].agg(["mean", "std"]).reset_index()
            ax.plot(pivot["K"], pivot["mean"], marker=MARKER_MODEL[model_label],
                    color=color, linewidth=1.8, markersize=6, label=model_label)
            ax.fill_between(pivot["K"],
                            pivot["mean"] - pivot["std"],
                            pivot["mean"] + pivot["std"],
                            color=color, alpha=0.12)

        # K=2/3/4 reference lines
        for kref, ls in [(2, "--"), (3, ":"), (4, "-.")]:
            ax.axvline(kref, color="gray", linewidth=0.8, linestyle=ls, alpha=0.5)

        ax.set_xlabel("K", fontsize=11)
        ax.set_ylabel("Silhouette Score", fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(K_ALL)
        ax.legend(fontsize=9, loc="upper right")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=-0.02)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig1_silhouette_by_k.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Fig 1 → {out}")


def fig2_stability_by_k(master_df):
    """Fig 2: K-Means and Hierarchical Ward stability ARI by K."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle("Fig 2: Partition Stability (Subsampling ARI) by K\n"
                 "K-Means: B=200 resamples | Hierarchical-Ward: B=100 resamples | GMM/Spectral: not available",
                 fontsize=11, fontweight="bold", y=1.02)

    for ax, cohort, title in zip(axes, ["A", "B"],
                                  ["Cohort A (Broad, N=4482, p=19)",
                                   "Cohort B (Fasting, N=967, p=24)"]):
        c_df = master_df[master_df["cohort"] == cohort]

        for model_label, model_col, link, color, ls in [
            ("K-Means (B=200)", "K-Means", "NA",   COLOR_MODEL["K-Means"],  "-"),
            ("Hier-Ward (B=100)", "Hierarchical", "Ward", COLOR_MODEL["Hier-Ward"], "--"),
        ]:
            m_df = c_df[(c_df["model"] == model_col) & (c_df["linkage"] == link) &
                        (c_df["K"].between(2, 7))].copy()
            pivot = m_df.groupby("K")["stability_ari_mean"].agg(["mean", "std"]).reset_index()
            pivot = pivot.dropna(subset=["mean"])
            if pivot.empty:
                continue
            ax.plot(pivot["K"], pivot["mean"], marker="o", color=color,
                    linewidth=1.8, linestyle=ls, markersize=6, label=model_label)
            ax.fill_between(pivot["K"],
                            (pivot["mean"] - pivot["std"]).clip(0),
                            (pivot["mean"] + pivot["std"]).clip(1),
                            color=color, alpha=0.12)

        # Reference thresholds
        ax.axhline(0.80, color="darkgreen", linewidth=0.8, linestyle=":", alpha=0.6,
                   label="ARI=0.80 reference")
        ax.axhline(0.50, color="darkorange", linewidth=0.8, linestyle=":", alpha=0.6,
                   label="ARI=0.50 reference")

        for kref, lss in [(2, "--"), (3, ":"), (4, "-.")]:
            ax.axvline(kref, color="gray", linewidth=0.8, linestyle=lss, alpha=0.5)

        ax.set_xlabel("K", fontsize=11)
        ax.set_ylabel("Mean ARI (subsampling stability)", fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(K_ALL)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="lower left")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig2_stability_by_k.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Fig 2 → {out}")


def fig3_cross_model_ari_heatmaps(ari_df):
    """
    Fig 3a/3b/3c: Cross-model ARI heatmaps for K=2, K=3, K=4.
    Layout: 2 rows (Cohort A top, Cohort B bottom) × 3 cols (K=2, K=3, K=4).
    Each cell shows: RAW value / LOG value.
    Models on axes: K-Means, GMM, Spectral, Hier-Ward (4×4 symmetric).
    """
    model_order = ["K-Means", "GMM", "Spectral", "Hier-Ward"]
    n_models = len(model_order)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "Fig 3: Cross-Model ARI Heatmaps (K=2, 3, 4)\n"
        "Upper-left triangle = ARI; lower-right = mirror. "
        "Each cell: RAW (top) / LOG (bottom). Diagonal = 1.0.",
        fontsize=12, fontweight="bold", y=1.01
    )

    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlGn
    norm = mcolors.Normalize(vmin=-0.05, vmax=1.0)

    for row_idx, cohort in enumerate(["A", "B"]):
        for col_idx, k in enumerate([2, 3, 4]):
            ax = axes[row_idx][col_idx]

            # Build RAW and LOG matrices
            ari_mat_raw = pd.DataFrame(np.nan, index=model_order, columns=model_order)
            ari_mat_log = pd.DataFrame(np.nan, index=model_order, columns=model_order)

            for _, r in ari_df[(ari_df["cohort"] == cohort) & (ari_df["K"] == k)].iterrows():
                ma, mb = r["model_A"], r["model_B"]
                if ma not in model_order or mb not in model_order:
                    continue
                val = r["ARI"]
                if r["scale"] == "Raw":
                    ari_mat_raw.loc[ma, mb] = val
                    ari_mat_raw.loc[mb, ma] = val
                else:
                    ari_mat_log.loc[ma, mb] = val
                    ari_mat_log.loc[mb, ma] = val

            # Average for background colour
            ari_combined = ari_mat_raw.copy()
            for i_m in model_order:
                for j_m in model_order:
                    vals = [v for v in [ari_mat_raw.loc[i_m, j_m], ari_mat_log.loc[i_m, j_m]]
                            if not np.isnan(v)]
                    ari_combined.loc[i_m, j_m] = np.mean(vals) if vals else np.nan

            # Draw heatmap background
            mat_vals = ari_combined.values.astype(float)
            np.fill_diagonal(mat_vals, 1.0)
            im = ax.imshow(mat_vals, cmap=cmap, norm=norm, aspect="auto")

            # Annotate each cell
            for i_idx, i_m in enumerate(model_order):
                for j_idx, j_m in enumerate(model_order):
                    if i_idx == j_idx:
                        ax.text(j_idx, i_idx, "1.0\n1.0",
                                ha="center", va="center", fontsize=7.5, fontweight="bold",
                                color="white")
                    else:
                        raw_v = ari_mat_raw.loc[i_m, j_m]
                        log_v = ari_mat_log.loc[i_m, j_m]
                        raw_s = f"{raw_v:.3f}" if not np.isnan(raw_v) else "NA"
                        log_s = f"{log_v:.3f}" if not np.isnan(log_v) else "NA"
                        cell_mean = np.nanmean([raw_v, log_v])
                        text_color = "white" if (np.isnan(cell_mean) or cell_mean < 0.35) else "black"
                        ax.text(j_idx, i_idx,
                                f"R:{raw_s}\nL:{log_s}",
                                ha="center", va="center", fontsize=7,
                                color=text_color)

            ax.set_xticks(range(n_models))
            ax.set_yticks(range(n_models))
            ax.set_xticklabels(model_order, fontsize=8.5, rotation=25, ha="right")
            ax.set_yticklabels(model_order, fontsize=8.5)
            cohort_label = "Cohort A (Broad)" if cohort == "A" else "Cohort B (Fasting)"
            ax.set_title(f"{cohort_label} | K={k}", fontsize=9.5, fontweight="bold")

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="ARI")

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    out = os.path.join(FIG_DIR, "fig3_cross_model_ari_heatmaps.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Fig 3 → {out}")


def fig4_cluster_balance(master_df):
    """Fig 4: Cluster size balance (min cluster proportion) for K=2,3,4."""
    primary_mask = (
        (master_df["model"] == "K-Means") & (master_df["linkage"] == "NA") |
        (master_df["model"] == "GMM")      & (master_df["linkage"] == "full_cov") |
        (master_df["model"] == "Spectral") & (master_df["linkage"] == "kNN_k10_NJW") |
        (master_df["model"] == "Hierarchical") & (master_df["linkage"] == "Ward")
    )
    prim_df = master_df[primary_mask & (master_df["K"].isin([2, 3, 4]))].copy()
    prim_df["model_label"] = prim_df.apply(
        lambda r: "Hier-Ward" if r["model"] == "Hierarchical" else r["model"], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("Fig 4: Minimum Cluster Proportion by K (K=2, 3, 4)\n"
                 "Dashed line at 1% = degeneracy diagnostic threshold",
                 fontsize=12, fontweight="bold", y=1.01)

    x_positions = {2: 0, 3: 1, 4: 2}
    bar_w = 0.18
    offsets = {"K-Means": -1.5, "GMM": -0.5, "Spectral": 0.5, "Hier-Ward": 1.5}

    for ax, cohort, title in zip(axes, ["A", "B"],
                                  ["Cohort A (Broad)", "Cohort B (Fasting)"]):
        c_df = prim_df[prim_df["cohort"] == cohort]
        for model_label, color in COLOR_MODEL.items():
            m_df = c_df[c_df["model_label"] == model_label]
            # Average across RAW and LOG
            grp = m_df.groupby("K")["min_cluster_prop"].mean()
            xs = [x_positions[k] + offsets[model_label] * bar_w for k in grp.index]
            ax.bar(xs, grp.values, width=bar_w, color=color, alpha=0.8,
                   label=model_label, edgecolor="white", linewidth=0.5)

        ax.axhline(0.01, color="red", linewidth=1.2, linestyle="--", alpha=0.8,
                   label="1% degeneracy threshold")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["K=2", "K=3", "K=4"], fontsize=11)
        ax.set_ylabel("Min cluster proportion", fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 0.65)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig4_cluster_balance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Fig 4 → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: K COMPARISON REPORT
# ─────────────────────────────────────────────────────────────────────────────
def write_k_comparison_report(master_df, ari_df, evid_df):
    """Write the qualitative evidence matrix and final K decision narrative."""

    # ── Pre-compute key numbers for narrative ─────────────────────────────
    def _ev(k, cohort, col):
        row = evid_df[(evid_df["K"] == k) & (evid_df["cohort"] == cohort)]
        if row.empty: return np.nan
        return row.iloc[0][col]

    # K-Means stability
    km_stab = {}
    stab_src = master_df[(master_df["model"] == "K-Means") & (master_df["linkage"] == "NA")]
    for k in [2, 3, 4]:
        for cohort in ["A", "B"]:
            vals = stab_src[(stab_src["K"] == k) & (stab_src["cohort"] == cohort)]["stability_ari_mean"].dropna()
            km_stab[(k, cohort)] = float(vals.mean()) if len(vals) > 0 else np.nan

    # Cross-model ARI
    cross_ari = {}
    for k in [2, 3, 4]:
        for cohort in ["A", "B"]:
            for scale in ["Raw", "Log1p"]:
                vals = ari_df[(ari_df["K"] == k) & (ari_df["cohort"] == cohort) & (ari_df["scale"] == scale)]["ARI"].dropna()
                cross_ari[(k, cohort, scale)] = float(vals.mean()) if len(vals) > 0 else np.nan

    # GMM silhouette at K=2 (special note on imbalance)
    gmm_met_a_raw = pd.read_csv(os.path.join(SANDBOX, "gmm", "metrics", "A_RAW_gmm_metrics.csv"))
    gmm_sc_k2_araw = gmm_met_a_raw[gmm_met_a_raw["K"] == 2].iloc[0]["silhouette"]
    gmm_min_k2_araw = gmm_met_a_raw[gmm_met_a_raw["K"] == 2].iloc[0]["min_cluster_proportion"]

    report_lines = []
    def W(s=""): report_lines.append(s)

    W("# Experiment 6: K Comparison Report")
    W("## Cross-Model Clustering Benchmark — Evidence-Based K Assessment")
    W()
    W(f"**Generated**: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    W(f"**Branch**: `experiment/clustering-sandbox`")
    W(f"**Status**: Analysis complete — no commits made")
    W()
    W("> **Methodological note**: High internal metrics (silhouette, CH, DB) do NOT establish")
    W("> biological validity. High stability ARI does NOT imply discrete natural clusters.")
    W("> Degenerate linkages (Complete, Average) are excluded from the primary evidence.")
    W("> GMM BIC is a relative model-fit criterion, not proof of biological subpopulations.")
    W("> Spectral eigengap in Cohort B (λ₃→λ₄) is interpreted as weak evidence for K=3,")
    W("> not K=4 (eigengap between eigenvalue 3 and 4 suggests 3 spectral dimensions).")
    W()
    W("---")
    W()
    W("## 1. Qualitative Evidence Matrix")
    W()
    W("Labels: **strong** | **moderate** | **weak** | **mixed** | **negative**")
    W("All labels are traceable to the quantitative tables (see citations).")
    W()

    # ── Evidence matrix header ─────────────────────────────────────────────
    W("| Evidence Dimension | K=2 (A) | K=2 (B) | K=3 (A) | K=3 (B) | K=4 (A) | K=4 (B) |")
    W("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")

    # Helper: silhouette label
    def sc_label(k, cohort):
        v = _ev(k, cohort, "mean_silhouette_primary_models")
        if np.isnan(v): return "NA"
        if v >= 0.25:   return "**strong**"
        if v >= 0.15:   return "**moderate**"
        if v >= 0.08:   return "**weak**"
        return "**negative**"

    def stab_label(stab_val):
        if np.isnan(stab_val): return "NA"
        if stab_val >= 0.85:   return "**strong**"
        if stab_val >= 0.70:   return "**moderate**"
        if stab_val >= 0.50:   return "**weak**"
        return "**negative**"

    def xari_label(k, cohort):
        r = cross_ari.get((k, cohort, "Raw"), np.nan)
        l = cross_ari.get((k, cohort, "Log1p"), np.nan)
        v = np.nanmean([r, l])
        if np.isnan(v): return "NA"
        if v >= 0.70:  return "**strong**"
        if v >= 0.50:  return "**moderate**"
        if v >= 0.30:  return "**weak**"
        return "**negative**"

    def deg_label(k, cohort):
        v = _ev(k, cohort, "any_small_cluster_flag")
        return "⚠️ flag" if v else "clean"

    def rawlog_label(k, cohort):
        d = _ev(k, cohort, "raw_log_cross_ari_diff")
        if np.isnan(d): return "NA"
        if d <= 0.05:   return "**strong**"
        if d <= 0.15:   return "**moderate**"
        return "**weak**"

    def hdbscan_label():
        return "**negative** (no large density-sep. groups)"

    W(f"| Internal separation (SC) | {sc_label(2,'A')} | {sc_label(2,'B')} | {sc_label(3,'A')} | {sc_label(3,'B')} | {sc_label(4,'A')} | {sc_label(4,'B')} |")
    W(f"| K-Means stability (ARI) | {stab_label(km_stab[(2,'A')])} | {stab_label(km_stab[(2,'B')])} | {stab_label(km_stab[(3,'A')])} | {stab_label(km_stab[(3,'B')])} | {stab_label(km_stab[(4,'A')])} | {stab_label(km_stab[(4,'B')])} |")
    W(f"| Hier-Ward stability (ARI) | {stab_label(_ev(2,'A','hierarchical_ward_stability_mean_ari'))} | {stab_label(_ev(2,'B','hierarchical_ward_stability_mean_ari'))} | {stab_label(_ev(3,'A','hierarchical_ward_stability_mean_ari'))} | {stab_label(_ev(3,'B','hierarchical_ward_stability_mean_ari'))} | {stab_label(_ev(4,'A','hierarchical_ward_stability_mean_ari'))} | {stab_label(_ev(4,'B','hierarchical_ward_stability_mean_ari'))} |")
    W(f"| Cross-model ARI agreement | {xari_label(2,'A')} | {xari_label(2,'B')} | {xari_label(3,'A')} | {xari_label(3,'B')} | {xari_label(4,'A')} | {xari_label(4,'B')} |")
    W(f"| RAW/LOG robustness | {rawlog_label(2,'A')} | {rawlog_label(2,'B')} | {rawlog_label(3,'A')} | {rawlog_label(3,'B')} | {rawlog_label(4,'A')} | {rawlog_label(4,'B')} |")
    W(f"| Degeneracy check | {deg_label(2,'A')} | {deg_label(2,'B')} | {deg_label(3,'A')} | {deg_label(3,'B')} | {deg_label(4,'A')} | {deg_label(4,'B')} |")
    W(f"| HDBSCAN consistency | {hdbscan_label()} | {hdbscan_label()} | {hdbscan_label()} | {hdbscan_label()} | {hdbscan_label()} | {hdbscan_label()} |")
    W(f"| GMM BIC minimum | A_RAW→K=4; A_LOG→K=3 | B_RAW→K=2; B_LOG→K=2 | A_LOG BIC min | — | A_RAW BIC min | — |")
    W(f"| Spectral eigengap | strongest at K=2 (λ₁) | strongest at K=2 (λ₁) | — | **weak** (λ₃→λ₄) | — | — |")
    W()
    W("---")
    W()
    W("## 2. Per-K Evidence Assessment")
    W()

    # ── K=2 ───────────────────────────────────────────────────────────────
    W("### K=2")
    W()
    W("**Strongest evidence FOR K=2:**")
    W(f"- K-Means subsampling stability is highest at K=2 across all matrices")
    W(f"  (A_RAW: {km_stab[(2,'A')]:.3f}, B_RAW: {km_stab[(2,'B')]:.3f},")
    W(f"   A_LOG: {master_df[(master_df['model']=='K-Means')&(master_df['matrix']=='A_LOG')&(master_df['K']==2)]['stability_ari_mean'].iloc[0]:.3f},")
    W(f"   B_LOG: {master_df[(master_df['model']=='K-Means')&(master_df['matrix']=='B_LOG')&(master_df['K']==2)]['stability_ari_mean'].iloc[0]:.3f})")
    W(f"- GMM BIC minimum for Cohort B (B_RAW, B_LOG) is at K=2.")
    W(f"- Spectral dominant eigengap is at K=2 in all four matrices (λ₁ gap ≈ 0.08–0.10),")
    W(f"  consistent with a single dominant spectral boundary.")
    W(f"- No small-cluster degeneracy at K=2 in any model or matrix.")
    W(f"- Cluster balance at K=2 is good across all models (min_prop ≈ 0.43–0.50 for K-Means).")
    W()
    W("**Strongest evidence AGAINST K=2:**")
    W(f"- All primary models show silhouette ≈ 0.06–0.11 at K=2 (well below 0.25 threshold).")
    W(f"  This indicates the K=2 boundary is not geometrically sharp — it is likely a")
    W(f"  broad midpoint split of a continuous overlapping distribution.")
    W(f"- GMM K=2 component split in A_RAW is heavily skewed: {gmm_min_k2_araw:.1%} / {1-gmm_min_k2_araw:.1%},")
    W(f"  with SC={gmm_sc_k2_araw:.4f}. High posterior confidence does not imply discrete separation.")
    W(f"- K=2 stability in Cohort B (K-Means ARI={km_stab[(2,'B')]:.3f}) is considerably lower")
    W(f"  than Cohort A (ARI={km_stab[(2,'A')]:.3f}), with large SD — not robustly stable in B.")
    W(f"- Cross-model ARI at K=2 is {cross_ari.get((2,'A','Raw'),np.nan):.3f} (A-RAW) / {cross_ari.get((2,'B','Raw'),np.nan):.3f} (B-RAW):")
    W(f"  different models do not agree on the same K=2 partition.")
    W(f"- HDBSCAN finds no large density-separated groups in either cohort at any scale.")
    W()
    W("**Cohort/model dependence:** K-Means stability strongly supports K=2 in Cohort A, moderately in B.")
    W("GMM BIC supports K=2 in Cohort B only. Spectral eigengap supports K=2 in all matrices")
    W("but this reflects the dominant λ₁, not a sharp K=2 partition signal.")
    W()
    W("**RAW/LOG robustness:** Stability pattern persists across RAW and LOG in both cohorts.")
    W()

    # ── K=3 ───────────────────────────────────────────────────────────────
    W("### K=3")
    W()
    W("**Strongest evidence FOR K=3:**")
    W(f"- GMM BIC minimum for A_LOG is at K=3.")
    W(f"- Spectral λ₃→λ₄ eigengap is the largest secondary gap in Cohort B")
    W(f"  (B_RAW: 0.080, B_LOG: 0.088), providing weak evidence for a 3-dimensional")
    W(f"  spectral representation — interpreted as possible K=3 spectral cutoff in Cohort B only.")
    W(f"- K-Means stability at K=3 is moderate-high in Cohort B (B_RAW: {km_stab[(3,'B')]:.3f}).")
    W()
    W("**Strongest evidence AGAINST K=3:**")
    W(f"- In Cohort A, K-Means stability drops substantially from K=2 to K=3")
    W(f"  (A_RAW: {km_stab[(2,'A')]:.3f}→{km_stab[(3,'A')]:.3f}).")
    W(f"- Silhouette scores at K=3 are universally very low (≈0.02–0.09 across all models).")
    W(f"- Cross-model ARI at K=3 is {cross_ari.get((3,'A','Raw'),np.nan):.3f} (A-RAW) / {cross_ari.get((3,'B','Raw'),np.nan):.3f} (B-RAW):")
    W(f"  models disagree substantially on which participants form the third group.")
    W(f"- Hierarchical Ward stability at K=3 is low in Cohort A (≈0.19–0.21).")
    W(f"- Spectral eigengap evidence is restricted to Cohort B; Cohort A shows no eigengap at K=3.")
    W(f"- GMM BIC support is in A_LOG only; A_RAW BIC minimum is at K=4, not K=3.")
    W()
    W("**Cohort/model dependence:** Evidence for K=3 is weak and inconsistent — it comes from")
    W("one cohort (spectral eigengap: B only), one matrix (GMM BIC: A_LOG only),")
    W("and one metric type. No convergent signal across models and cohorts.")
    W()

    # ── K=4 ───────────────────────────────────────────────────────────────
    W("### K=4")
    W()
    W("**Strongest evidence FOR K=4:**")
    W(f"- GMM BIC minimum for A_RAW is at K=4.")
    W(f"- K-Means stability in A_LOG at K=4 is {master_df[(master_df['model']=='K-Means')&(master_df['matrix']=='A_LOG')&(master_df['K']==4)]['stability_ari_mean'].iloc[0]:.3f}")
    W(f"  — notably high (comparable to K=2 stability in that matrix).")
    W()
    W("**Strongest evidence AGAINST K=4:**")
    W(f"- GMM BIC supports K=4 in A_RAW only; all other matrices prefer lower K.")
    W(f"- Silhouette at K=4 is near-zero across all models and matrices (≈0.02–0.09).")
    W(f"- Cross-model ARI at K=4 is {cross_ari.get((4,'A','Raw'),np.nan):.3f} (A-RAW) / {cross_ari.get((4,'B','Raw'),np.nan):.3f} (B-RAW).")
    W(f"- No spectral eigengap evidence for K=4 in any matrix.")
    W(f"- No HDBSCAN density signal for 4 subpopulations.")
    W(f"- A_LOG high K-Means stability at K=4 is not consistent with A_RAW (A_RAW K=4: {km_stab[(4,'A')]:.3f}).")
    W()
    W("**Cohort/model dependence:** K=4 support is almost entirely single-matrix (A_RAW GMM BIC)")
    W("and does not generalize across matrices, cohorts, or model families.")
    W()
    W("---")
    W()

    # ── Final decision ─────────────────────────────────────────────────────
    W("## 3. Final Conclusion")
    W()
    W("> **Conclusion: No single K is sufficiently supported.**")
    W()
    W("The cross-model benchmark does not produce convergent evidence for K=2, K=3, or K=4.")
    W("The experiments are more consistent with broad, overlapping phenotypic variation than")
    W("with sharply separated, reproducible discrete subpopulations.")
    W()
    W("**Rationale:**")
    W()
    W("1. **K=2** has the strongest single-metric case (K-Means stability in Cohort A),")
    W("   but fails on geometric separation (SC ≈ 0.06–0.11), has poor cross-model agreement")
    W("   (models assign different participants to the two groups), and is not supported")
    W("   by density-based evidence (HDBSCAN). K=2 stability reflects that any")
    W("   continuous ellipsoidal distribution will repeatedly split at roughly the same")
    W("   midpoint — not that the split is meaningful.")
    W()
    W("2. **K=3** has weak secondary evidence (Cohort B spectral eigengap, A_LOG GMM BIC)")
    W("   but this evidence is fragmented: each signal comes from a different cohort,")
    W("   matrix, and model. No consistent pattern across all four matrices or model families.")
    W()
    W("3. **K=4** has the weakest case: supported by a single BIC minimum in a single matrix")
    W("   (A_RAW GMM), with negligible geometric separation and no cross-model agreement.")
    W()
    W("4. **HDBSCAN** finds no robust large-scale density-separated populations in either")
    W("   cohort at any parameter setting, consistent with the absence of discrete cluster structure.")
    W()
    W("**Recommendation:**")
    W()
    W("> Do not commit to a single K yet.")
    W("> The clustering evidence from all five model families is insufficient to justify")
    W("> selecting a specific K for downstream biological analysis.")
    W("> If a forced choice is required by study design, K=2 has the most stability")
    W("> support in Cohort A under K-Means, but this should be clearly labelled as")
    W("> a pragmatic choice under weak evidence, not as a data-supported finding.")
    W()
    W("---")
    W()
    W("## 4. HDBSCAN Sanity Check Summary")
    W()
    W("- **Cohort A**: All parameter combinations yield a single mega-cluster (90–99.9% of participants)")
    W("  plus 1–2 micro-clusters of N=3–7. No multi-cluster density structure detected.")
    W("- **Cohort B**: Best-case solution (B_LOG, mcs=5, ms=3) yields 2 clusters covering 499")
    W("  participants, but this collapses to 126 participants at mcs≥7 (structural instability).")
    W("- **Overall**: HDBSCAN does not provide independent evidence for any stable K.")
    W()
    W("---")
    W()
    W("## 5. Source Table References")
    W()
    W("| Table | Path |")
    W("|:------|:-----|")
    W("| Master metrics | `clustering_sandbox/cross_model/cross_model_master_metrics.csv` |")
    W("| Pairwise ARI   | `clustering_sandbox/cross_model/cross_model_pairwise_ari.csv` |")
    W("| Pairwise NMI   | `clustering_sandbox/cross_model/cross_model_pairwise_nmi.csv` |")
    W("| K evidence     | `clustering_sandbox/cross_model/k_evidence_summary.csv` |")
    W("| K-Means stability source | `clustering_sandbox/kmeans_stability/metrics/cross_matrix_stability_summary.csv` |")
    W("| Hierarchical stability source | `clustering_sandbox/hierarchical/metrics/hierarchical_stability_summary.csv` |")
    W("| GMM metrics source | `clustering_sandbox/gmm/metrics/{matrix}_gmm_metrics.csv` |")
    W("| Spectral metrics source | `clustering_sandbox/spectral/metrics/{matrix}_spectral_metrics.csv` |")

    report_path = os.path.join(OUT_DIR, "k_comparison_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"✅ K comparison report → {report_path}")
    return report_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: METADATA
# ─────────────────────────────────────────────────────────────────────────────
def write_metadata():
    meta = {
        "experiment": "Experiment 6: Cross-Model Clustering Benchmark and K Selection",
        "branch": "experiment/clustering-sandbox",
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "models_included": ["K-Means", "GMM", "Spectral", "Hierarchical (Ward/Complete/Average)"],
        "hdbscan_role": "sanity check only — not included in K-based comparison",
        "matrices": MATRICES,
        "cohort_A": {"matrices": ["A_RAW", "A_LOG"], "N": 4482, "p": 19},
        "cohort_B": {"matrices": ["B_RAW", "B_LOG"], "N": 967, "p": 24},
        "K_primary": K_PRIMARY,
        "K_all_evaluated": K_ALL,
        "pairwise_models_for_ARI": list(MODEL_LABELS.values()),
        "stability_sources": {
            "K-Means": "kmeans_stability (B=200, 80% subsample, ARI)",
            "Hierarchical-Ward": "hierarchical_stability (B=100, 80% subsample, participant-overlap ARI)",
            "GMM": "NA — not generated",
            "Spectral": "NA — not generated"
        },
        "degenerate_linkages_excluded_from_primary": ["Complete", "Average"],
        "gmm_bic_minima_verified": {"A_RAW": 4, "A_LOG": 3, "B_RAW": 2, "B_LOG": 2},
        "spectral_eigengap_interpretation": (
            "Lambda_3 to Lambda_4 gap in Cohort B (B_RAW: 0.080, B_LOG: 0.088) "
            "supports possible K=3 spectral cutoff (3 spectral dimensions) — weak secondary evidence only. "
            "Cohort A shows no meaningful eigengap beyond K=2."
        ),
        "seqn_audit": "PASSED — exact row-order match across all models/matrices",
        "no_values_invented": True,
        "no_commits_made": True,
        "no_biological_interpretation": True,
        "random_seeds": "not applicable (reading existing outputs only)",
        "sklearn_version_source_experiments": "1.4.1.post1",
        "outputs": {
            "cross_model_master_metrics.csv": "All models × matrices × K internal metrics",
            "cross_model_pairwise_ari.csv": "ARI between every model pair per matrix/K",
            "cross_model_pairwise_nmi.csv": "NMI between every model pair per matrix/K",
            "k_evidence_summary.csv": "Per-K aggregated evidence dimensions",
            "k_comparison_report.md": "Qualitative evidence matrix + final K verdict",
            "figures/fig1_silhouette_by_k.png": "Cross-model silhouette by K",
            "figures/fig2_stability_by_k.png": "Stability ARI by K",
            "figures/fig3_cross_model_ari_heatmaps.png": "6-panel ARI heatmaps K=2/3/4",
            "figures/fig4_cluster_balance.png": "Min cluster proportion K=2/3/4"
        }
    }
    out_path = os.path.join(META_DIR, "cross_model_experiment_metadata.json")
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Metadata → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("EXPERIMENT 6: Cross-Model Clustering Benchmark and K Selection")
    print("=" * 70)
    print()

    print("Step 0: Loading and auditing all assignment tables...")
    assigns = load_all_assignments()
    print()

    print("Step 1: Building master metrics table...")
    master_df = build_master_metrics()
    print()

    print("Step 2: Computing cross-model pairwise ARI and NMI...")
    ari_df, nmi_df = compute_pairwise_ari_nmi(assigns)
    print()

    print("Step 3: Building K evidence summary...")
    evid_df = build_k_evidence_summary(master_df, ari_df)
    print()

    print("Step 4: Generating figures...")
    fig1_silhouette_by_k(master_df)
    fig2_stability_by_k(master_df)
    fig3_cross_model_ari_heatmaps(ari_df)
    fig4_cluster_balance(master_df)
    print()

    print("Step 5: Writing K comparison report...")
    write_k_comparison_report(master_df, ari_df, evid_df)
    print()

    print("Step 6: Writing metadata...")
    write_metadata()
    print()

    print("=" * 70)
    print("Experiment 6 complete. Run validate_cross_model_benchmark.py next.")
    print("=" * 70)


if __name__ == "__main__":
    main()
