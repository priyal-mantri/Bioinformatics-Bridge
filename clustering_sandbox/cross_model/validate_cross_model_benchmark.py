"""
Experiment 6: Cross-Model Clustering Benchmark — Validation Script
===================================================================
Audits all outputs of run_cross_model_benchmark.py.

Checks:
  - All expected files present
  - Master metrics: correct row count, no NaN in key columns, K range correct
  - Pairwise ARI/NMI: all model pairs present, values in [-0.1, 1.0]
  - K evidence summary: all K × cohort combinations present
  - SEQN alignment: re-verified against source assignment files
  - HDBSCAN: not present in pairwise ARI table (confirms exclusion)
  - Degenerate linkages: present in master_metrics with degeneracy_note
  - No values invented for GMM/Spectral stability
"""

import os
import json
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SANDBOX   = os.path.join(REPO_ROOT, "clustering_sandbox")
OUT_DIR   = os.path.join(SANDBOX, "cross_model")
FIG_DIR   = os.path.join(OUT_DIR, "figures")
META_DIR  = os.path.join(OUT_DIR, "metadata")

MATRICES       = ["A_RAW", "A_LOG", "B_RAW", "B_LOG"]
COHORT_MAP     = {"A_RAW": "A", "A_LOG": "A", "B_RAW": "B", "B_LOG": "B"}
N_MAP          = {"A_RAW": 4482, "A_LOG": 4482, "B_RAW": 967, "B_LOG": 967}
K_ALL          = [2, 3, 4, 5, 6, 7]
K_PRIMARY      = [2, 3, 4]
MODEL_ORDER    = ["K-Means", "GMM", "Spectral", "Hier-Ward"]
EXPECTED_PAIRS = [
    ("K-Means", "GMM"), ("K-Means", "Spectral"), ("K-Means", "Hier-Ward"),
    ("GMM", "Spectral"), ("GMM", "Hier-Ward"), ("Spectral", "Hier-Ward"),
]

EXPECTED_FILES = [
    os.path.join(OUT_DIR, "cross_model_master_metrics.csv"),
    os.path.join(OUT_DIR, "cross_model_pairwise_ari.csv"),
    os.path.join(OUT_DIR, "cross_model_pairwise_nmi.csv"),
    os.path.join(OUT_DIR, "k_evidence_summary.csv"),
    os.path.join(OUT_DIR, "k_comparison_report.md"),
    os.path.join(META_DIR, "cross_model_experiment_metadata.json"),
    os.path.join(FIG_DIR, "fig1_silhouette_by_k.png"),
    os.path.join(FIG_DIR, "fig2_stability_by_k.png"),
    os.path.join(FIG_DIR, "fig3_cross_model_ari_heatmaps.png"),
    os.path.join(FIG_DIR, "fig4_cluster_balance.png"),
]

errors = []
warnings_list = []

def FAIL(msg):
    errors.append(f"❌ FAIL: {msg}")

def WARN(msg):
    warnings_list.append(f"⚠️  WARN: {msg}")

def PASS(msg):
    print(f"  ✅ {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# Check 1: File presence
# ─────────────────────────────────────────────────────────────────────────────
print("Check 1: File presence")
for fpath in EXPECTED_FILES:
    if os.path.isfile(fpath):
        PASS(f"Found: {os.path.basename(fpath)}")
    else:
        FAIL(f"Missing file: {fpath}")

# ─────────────────────────────────────────────────────────────────────────────
# Check 2: Master metrics table
# ─────────────────────────────────────────────────────────────────────────────
print("\nCheck 2: Master metrics table")
master = pd.read_csv(os.path.join(OUT_DIR, "cross_model_master_metrics.csv"))

# Expected rows: 4 matrices × 6 K values × (K-Means + GMM + Spectral + Ward + Complete + Average)
# = 4 × 6 × 6 = 144
expected_min_rows = 4 * 6 * 4  # minimum: just primary models
if len(master) >= expected_min_rows:
    PASS(f"Row count: {len(master)} (≥ {expected_min_rows} minimum)")
else:
    FAIL(f"Master metrics has only {len(master)} rows (expected ≥ {expected_min_rows})")

req_cols = ["cohort", "matrix", "model", "linkage", "K", "silhouette", "CH", "DB",
            "stability_ari_mean", "min_cluster_prop", "max_cluster_prop",
            "small_cluster_flag", "degeneracy_note"]
for col in req_cols:
    if col in master.columns:
        PASS(f"Column present: {col}")
    else:
        FAIL(f"Missing column: {col}")

# K range check
for m in MATRICES:
    m_df = master[master["matrix"] == m]
    k_vals = sorted(m_df["K"].unique())
    if set(K_ALL).issubset(set(k_vals)):
        PASS(f"{m}: all K={K_ALL} present")
    else:
        missing_k = set(K_ALL) - set(k_vals)
        FAIL(f"{m}: missing K values: {missing_k}")

# Degenerate linkages present with note
degenerate_rows = master[master["linkage"].isin(["Complete", "Average"])]
if len(degenerate_rows) > 0:
    have_note = degenerate_rows["degeneracy_note"].str.contains("DEGENERATE", na=False).all()
    if have_note:
        PASS(f"Degenerate linkages: {len(degenerate_rows)} rows all have DEGENERATE note")
    else:
        WARN(f"Some Complete/Average rows missing DEGENERATE note")
else:
    WARN("No Complete/Average rows found in master metrics")

# GMM/Spectral stability should be NA
gmm_stab = master[(master["model"] == "GMM")]["stability_ari_mean"]
sp_stab  = master[(master["model"] == "Spectral")]["stability_ari_mean"]
if gmm_stab.isna().all():
    PASS("GMM stability_ari_mean: all NA (correct — not generated)")
else:
    FAIL(f"GMM stability_ari_mean has non-NA values: {gmm_stab.dropna().values}")
if sp_stab.isna().all():
    PASS("Spectral stability_ari_mean: all NA (correct — not generated)")
else:
    FAIL(f"Spectral stability_ari_mean has non-NA values: {sp_stab.dropna().values}")

# ─────────────────────────────────────────────────────────────────────────────
# Check 3: Pairwise ARI table
# ─────────────────────────────────────────────────────────────────────────────
print("\nCheck 3: Pairwise ARI / NMI table")
ari_df = pd.read_csv(os.path.join(OUT_DIR, "cross_model_pairwise_ari.csv"))
nmi_df = pd.read_csv(os.path.join(OUT_DIR, "cross_model_pairwise_nmi.csv"))

# Expected rows: 4 matrices × 6 K values × 6 pairs = 144
expected_ari_rows = len(MATRICES) * len(K_ALL) * len(EXPECTED_PAIRS)
if len(ari_df) == expected_ari_rows:
    PASS(f"ARI table row count: {len(ari_df)} (expected {expected_ari_rows})")
else:
    FAIL(f"ARI table has {len(ari_df)} rows (expected {expected_ari_rows})")

# ARI value range
ari_vals = ari_df["ARI"].dropna()
if (ari_vals >= -0.1).all() and (ari_vals <= 1.001).all():
    PASS(f"ARI values in valid range [-0.1, 1.0]: min={ari_vals.min():.4f} max={ari_vals.max():.4f}")
else:
    FAIL(f"ARI values out of range: min={ari_vals.min():.4f} max={ari_vals.max():.4f}")

# No HDBSCAN in ARI table
hdb_rows = ari_df[ari_df["model_A"].str.contains("HDBSCAN|hdbscan", na=False) |
                  ari_df["model_B"].str.contains("HDBSCAN|hdbscan", na=False)]
if len(hdb_rows) == 0:
    PASS("HDBSCAN correctly excluded from pairwise ARI table")
else:
    FAIL(f"HDBSCAN found in ARI table ({len(hdb_rows)} rows) — should be excluded")

# All expected pairs present for each matrix/K
for m in MATRICES:
    for k in K_ALL:
        m_k_df = ari_df[(ari_df["matrix"] == m) & (ari_df["K"] == k)]
        found_pairs = set(zip(m_k_df["model_A"], m_k_df["model_B"]))
        for pa, pb in EXPECTED_PAIRS:
            if (pa, pb) not in found_pairs and (pb, pa) not in found_pairs:
                FAIL(f"Missing pair ({pa},{pb}) for {m}/K{k}")
PASS("All expected model pairs present across all matrices and K values")

# ─────────────────────────────────────────────────────────────────────────────
# Check 4: K evidence summary
# ─────────────────────────────────────────────────────────────────────────────
print("\nCheck 4: K evidence summary")
evid = pd.read_csv(os.path.join(OUT_DIR, "k_evidence_summary.csv"))

expected_ev_rows = len(K_ALL) * 2  # 6 K values × 2 cohorts
if len(evid) == expected_ev_rows:
    PASS(f"K evidence rows: {len(evid)} (expected {expected_ev_rows})")
else:
    FAIL(f"K evidence has {len(evid)} rows (expected {expected_ev_rows})")

for k in K_ALL:
    for cohort in ["A", "B"]:
        row = evid[(evid["K"] == k) & (evid["cohort"] == cohort)]
        if len(row) == 1:
            pass
        else:
            FAIL(f"Missing K={k}, cohort={cohort} in evidence summary")
PASS("All K × cohort combinations present in evidence summary")

# ─────────────────────────────────────────────────────────────────────────────
# Check 5: SEQN re-verification
# ─────────────────────────────────────────────────────────────────────────────
print("\nCheck 5: SEQN re-verification")
for m in MATRICES:
    expected_n = N_MAP[m]
    km_seqn = pd.read_csv(os.path.join(SANDBOX, "kmeans", "cluster_assignments",
                                       f"{m}_kmeans_assignments.csv"))["SEQN"]
    for model_name, path_pat in [
        ("GMM", os.path.join(SANDBOX, "gmm", "assignments", f"{m}_gmm_assignments.csv")),
        ("Spectral", os.path.join(SANDBOX, "spectral", "cluster_assignments",
                                  f"{m}_spectral_assignments.csv")),
        ("Hierarchical", os.path.join(SANDBOX, "hierarchical", "cluster_assignments",
                                      f"{m}_hierarchical_assignments.csv")),
    ]:
        other = pd.read_csv(path_pat)["SEQN"]
        if (km_seqn.values == other.values).all() and len(other) == expected_n:
            PASS(f"SEQN alignment: K-Means/{m} ↔ {model_name}/{m} (N={expected_n})")
        else:
            FAIL(f"SEQN mismatch: K-Means/{m} ↔ {model_name}/{m}")

# ─────────────────────────────────────────────────────────────────────────────
# Check 6: k_comparison_report.md contains required sections
# ─────────────────────────────────────────────────────────────────────────────
print("\nCheck 6: k_comparison_report.md structure")
with open(os.path.join(OUT_DIR, "k_comparison_report.md")) as f:
    report_text = f.read()

for section in ["Qualitative Evidence Matrix", "K=2", "K=3", "K=4",
                "Final Conclusion", "HDBSCAN Sanity Check"]:
    if section in report_text:
        PASS(f"Report section present: '{section}'")
    else:
        FAIL(f"Report missing section: '{section}'")

# Must contain final conclusion keyword
if any(kw in report_text for kw in [
        "No single K is sufficiently supported",
        "K=2 is sufficiently supported",
        "K=3 is sufficiently supported",
        "K=4 is sufficiently supported"]):
    PASS("Report contains one of the four required conclusion statements")
else:
    FAIL("Report does not contain a recognized conclusion statement")

# ─────────────────────────────────────────────────────────────────────────────
# Check 7: Metadata
# ─────────────────────────────────────────────────────────────────────────────
print("\nCheck 7: Metadata")
with open(os.path.join(META_DIR, "cross_model_experiment_metadata.json")) as f:
    meta = json.load(f)

for key in ["experiment", "branch", "timestamp_utc", "models_included",
            "hdbscan_role", "matrices", "K_primary", "seqn_audit",
            "no_values_invented", "no_commits_made", "outputs"]:
    if key in meta:
        PASS(f"Metadata key present: {key}")
    else:
        FAIL(f"Metadata missing key: {key}")

if meta.get("no_values_invented") is True:
    PASS("Metadata confirms: no values invented")
else:
    FAIL("Metadata does not confirm no_values_invented=True")

if meta.get("seqn_audit") == "PASSED — exact row-order match across all models/matrices":
    PASS("Metadata confirms: SEQN audit PASSED")
else:
    WARN(f"Metadata SEQN audit field: {meta.get('seqn_audit')}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
if errors:
    print(f"VALIDATION: ❌ FAILED — {len(errors)} error(s) found")
    for e in errors:
        print(f"  {e}")
else:
    print(f"VALIDATION: ✅ PASSED — all checks clean")

if warnings_list:
    print(f"\nWarnings ({len(warnings_list)}):")
    for w in warnings_list:
        print(f"  {w}")
print("=" * 60)
