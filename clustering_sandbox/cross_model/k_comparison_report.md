# Experiment 6: K Comparison Report
## Cross-Model Clustering Benchmark — Evidence-Based K Assessment

**Generated**: 2026-09-14 06:29 UTC
**Branch**: `experiment/clustering-sandbox`
**Status**: Analysis complete — no commits made

> **Methodological note**: High internal metrics (silhouette, CH, DB) do NOT establish
> biological validity. High stability ARI does NOT imply discrete natural clusters.
> Degenerate linkages (Complete, Average) are excluded from the primary evidence.
> GMM BIC is a relative model-fit criterion, not proof of biological subpopulations.
> Spectral eigengap in Cohort B (λ₃→λ₄) is interpreted as weak evidence for K=3,
> not K=4 (eigengap between eigenvalue 3 and 4 suggests 3 spectral dimensions).

---

## 1. Qualitative Evidence Matrix

Labels: **strong** | **moderate** | **weak** | **mixed** | **negative**
All labels are traceable to the quantitative tables (see citations).

| Evidence Dimension | K=2 (A) | K=2 (B) | K=3 (A) | K=3 (B) | K=4 (A) | K=4 (B) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Internal separation (SC) | **weak** | **weak** | **negative** | **negative** | **negative** | **negative** |
| K-Means stability (ARI) | **strong** | **moderate** | **moderate** | **moderate** | **moderate** | **moderate** |
| Hier-Ward stability (ARI) | **negative** | **negative** | **negative** | **negative** | **negative** | **negative** |
| Cross-model ARI agreement | **negative** | **negative** | **negative** | **weak** | **negative** | **weak** |
| RAW/LOG robustness | **strong** | **strong** | **moderate** | **moderate** | **moderate** | **strong** |
| Degeneracy check | clean | clean | ⚠️ flag | clean | ⚠️ flag | clean |
| HDBSCAN consistency | **negative** (no large density-sep. groups) | **negative** (no large density-sep. groups) | **negative** (no large density-sep. groups) | **negative** (no large density-sep. groups) | **negative** (no large density-sep. groups) | **negative** (no large density-sep. groups) |
| GMM BIC minimum | A_RAW→K=4; A_LOG→K=3 | B_RAW→K=2; B_LOG→K=2 | A_LOG BIC min | — | A_RAW BIC min | — |
| Spectral eigengap | strongest at K=2 (λ₁) | strongest at K=2 (λ₁) | — | **weak** (λ₃→λ₄) | — | — |

---

## 2. Per-K Evidence Assessment

### K=2

**Strongest evidence FOR K=2:**
- K-Means subsampling stability is highest at K=2 across all matrices
  (A_RAW: 0.958, B_RAW: 0.799,
   A_LOG: 0.951,
   B_LOG: 0.842)
- GMM BIC minimum for Cohort B (B_RAW, B_LOG) is at K=2.
- Spectral dominant eigengap is at K=2 in all four matrices (λ₁ gap ≈ 0.08–0.10),
  consistent with a single dominant spectral boundary.
- No small-cluster degeneracy at K=2 in any model or matrix.
- Cluster balance at K=2 is good across all models (min_prop ≈ 0.43–0.50 for K-Means).

**Strongest evidence AGAINST K=2:**
- All primary models show silhouette ≈ 0.06–0.11 at K=2 (well below 0.25 threshold).
  This indicates the K=2 boundary is not geometrically sharp — it is likely a
  broad midpoint split of a continuous overlapping distribution.
- GMM K=2 component split in A_RAW is heavily skewed: 15.7% / 84.3%,
  with SC=0.2071. High posterior confidence does not imply discrete separation.
- K=2 stability in Cohort B (K-Means ARI=0.799) is considerably lower
  than Cohort A (ARI=0.958), with large SD — not robustly stable in B.
- Cross-model ARI at K=2 is 0.221 (A-RAW) / 0.115 (B-RAW):
  different models do not agree on the same K=2 partition.
- HDBSCAN finds no large density-separated groups in either cohort at any scale.

**Cohort/model dependence:** K-Means stability strongly supports K=2 in Cohort A, moderately in B.
GMM BIC supports K=2 in Cohort B only. Spectral eigengap supports K=2 in all matrices
but this reflects the dominant λ₁, not a sharp K=2 partition signal.

**RAW/LOG robustness:** Stability pattern persists across RAW and LOG in both cohorts.

### K=3

**Strongest evidence FOR K=3:**
- GMM BIC minimum for A_LOG is at K=3.
- Spectral λ₃→λ₄ eigengap is the largest secondary gap in Cohort B
  (B_RAW: 0.080, B_LOG: 0.088), providing weak evidence for a 3-dimensional
  spectral representation — interpreted as possible K=3 spectral cutoff in Cohort B only.
- K-Means stability at K=3 is moderate-high in Cohort B (B_RAW: 0.830).

**Strongest evidence AGAINST K=3:**
- In Cohort A, K-Means stability drops substantially from K=2 to K=3
  (A_RAW: 0.958→0.830).
- Silhouette scores at K=3 are universally very low (≈0.02–0.09 across all models).
- Cross-model ARI at K=3 is 0.168 (A-RAW) / 0.263 (B-RAW):
  models disagree substantially on which participants form the third group.
- Hierarchical Ward stability at K=3 is low in Cohort A (≈0.19–0.21).
- Spectral eigengap evidence is restricted to Cohort B; Cohort A shows no eigengap at K=3.
- GMM BIC support is in A_LOG only; A_RAW BIC minimum is at K=4, not K=3.

**Cohort/model dependence:** Evidence for K=3 is weak and inconsistent — it comes from
one cohort (spectral eigengap: B only), one matrix (GMM BIC: A_LOG only),
and one metric type. No convergent signal across models and cohorts.

### K=4

**Strongest evidence FOR K=4:**
- GMM BIC minimum for A_RAW is at K=4.
- K-Means stability in A_LOG at K=4 is 0.920
  — notably high (comparable to K=2 stability in that matrix).

**Strongest evidence AGAINST K=4:**
- GMM BIC supports K=4 in A_RAW only; all other matrices prefer lower K.
- Silhouette at K=4 is near-zero across all models and matrices (≈0.02–0.09).
- Cross-model ARI at K=4 is 0.227 (A-RAW) / 0.367 (B-RAW).
- No spectral eigengap evidence for K=4 in any matrix.
- No HDBSCAN density signal for 4 subpopulations.
- A_LOG high K-Means stability at K=4 is not consistent with A_RAW (A_RAW K=4: 0.826).

**Cohort/model dependence:** K=4 support is almost entirely single-matrix (A_RAW GMM BIC)
and does not generalize across matrices, cohorts, or model families.

---

## 3. Final Conclusion

> **Conclusion: No single K is sufficiently supported.**

The cross-model benchmark does not produce convergent evidence for K=2, K=3, or K=4.
The experiments are more consistent with broad, overlapping phenotypic variation than
with sharply separated, reproducible discrete subpopulations.

**Rationale:**

1. **K=2** has the strongest single-metric case (K-Means stability in Cohort A),
   but fails on geometric separation (SC ≈ 0.06–0.11), has poor cross-model agreement
   (models assign different participants to the two groups), and is not supported
   by density-based evidence (HDBSCAN). K=2 stability reflects that any
   continuous ellipsoidal distribution will repeatedly split at roughly the same
   midpoint — not that the split is meaningful.

2. **K=3** has weak secondary evidence (Cohort B spectral eigengap, A_LOG GMM BIC)
   but this evidence is fragmented: each signal comes from a different cohort,
   matrix, and model. No consistent pattern across all four matrices or model families.

3. **K=4** has the weakest case: supported by a single BIC minimum in a single matrix
   (A_RAW GMM), with negligible geometric separation and no cross-model agreement.

4. **HDBSCAN** finds no robust large-scale density-separated populations in either
   cohort at any parameter setting, consistent with the absence of discrete cluster structure.

**Recommendation:**

> Do not commit to a single K yet.
> The clustering evidence from all five model families is insufficient to justify
> selecting a specific K for downstream biological analysis.
> If a forced choice is required by study design, K=2 has the most stability
> support in Cohort A under K-Means, but this should be clearly labelled as
> a pragmatic choice under weak evidence, not as a data-supported finding.

---

## 4. HDBSCAN Sanity Check Summary

- **Cohort A**: All parameter combinations yield a single mega-cluster (90–99.9% of participants)
  plus 1–2 micro-clusters of N=3–7. No multi-cluster density structure detected.
- **Cohort B**: Best-case solution (B_LOG, mcs=5, ms=3) yields 2 clusters covering 499
  participants, but this collapses to 126 participants at mcs≥7 (structural instability).
- **Overall**: HDBSCAN does not provide independent evidence for any stable K.

---

## 5. Source Table References

| Table | Path |
|:------|:-----|
| Master metrics | `clustering_sandbox/cross_model/cross_model_master_metrics.csv` |
| Pairwise ARI   | `clustering_sandbox/cross_model/cross_model_pairwise_ari.csv` |
| Pairwise NMI   | `clustering_sandbox/cross_model/cross_model_pairwise_nmi.csv` |
| K evidence     | `clustering_sandbox/cross_model/k_evidence_summary.csv` |
| K-Means stability source | `clustering_sandbox/kmeans_stability/metrics/cross_matrix_stability_summary.csv` |
| Hierarchical stability source | `clustering_sandbox/hierarchical/metrics/hierarchical_stability_summary.csv` |
| GMM metrics source | `clustering_sandbox/gmm/metrics/{matrix}_gmm_metrics.csv` |
| Spectral metrics source | `clustering_sandbox/spectral/metrics/{matrix}_spectral_metrics.csv` |