# output/phenotype_characterization/

This directory contains the Stage 8 outputs: phenotypic characterization of the exploratory candidate cluster partitions identified in the cross-model benchmark (Stage 7), and a directional comparison with Ayurvedic constitutional literature.

## Candidate Partitions

Following the cross-model clustering benchmark (Decision 014), two cohort-specific exploratory candidate partitions were retained:

- **Cohort A**: K=2 (K-Means, A_RAW representation)
- **Cohort B**: K=3 (K-Means, B_RAW representation)

These are exploratory partitions. They do not represent validated biological phenotype groups, and they are not labeled as or equated with Ayurvedic Dosha categories (Vata, Pitta, Kapha). The cross-model evidence supporting these candidates is moderate and is described in full in `docs/methodology_decision_log.md` (Decision 014).

## Directory Structure

```
phenotype_characterization/
├── cohort_a/
│   ├── cohort_a_membership.csv
│   ├── cohort_a_primary_profiles.csv
│   └── cohort_a_statistical_tests.csv
├── cohort_b/
│   ├── cohort_b_membership.csv
│   ├── cohort_b_primary_profiles.csv
│   └── cohort_b_statistical_tests.csv
├── literature_concordance/
│   ├── cohort_a_literature_concordance.csv
│   ├── cohort_b_literature_concordance.csv
│   └── literature_concordance_summary.csv
├── sensitivity/
│   ├── cross_model_pairwise_ari.csv
│   └── raw_vs_log_feature_persistence.csv
├── plots/
│   ├── fig1_cohort_a_zscore_heatmap.png
│   ├── fig2_cohort_b_zscore_heatmap.png
│   ├── fig3_cohort_a_discriminating_features_boxplots.png
│   ├── fig4_cohort_b_discriminating_features_boxplots.png
│   ├── fig5_derived_metabolic_ratios.png
│   └── fig6_literature_concordance_matrix.png
└── metadata.json
```

## cohort_a/

Contains phenotypic characterization outputs for the Cohort A K=2 candidate partition.

| File | Description |
| :--- | :--- |
| `cohort_a_membership.csv` | Number of participants assigned to each cluster |
| `cohort_a_primary_profiles.csv` | Mean and standard deviation of each biomarker by cluster, on both raw and standardized scales |
| `cohort_a_statistical_tests.csv` | Between-cluster statistical comparison for each feature (test statistic, p-value, effect size, FDR-corrected q-value) |

## cohort_b/

Same structure as `cohort_a/`, for the Cohort B K=3 candidate partition.

| File | Description |
| :--- | :--- |
| `cohort_b_membership.csv` | Number of participants assigned to each cluster |
| `cohort_b_primary_profiles.csv` | Mean and standard deviation of each biomarker by cluster |
| `cohort_b_statistical_tests.csv` | Between-cluster statistical comparison for each feature |

## literature_concordance/

These tables compare the observed direction of biomarker displacement in each cluster against the physiological characteristics attributed to each Dosha constitution in independently published Ayurvedic literature.

This is a **concordance analysis**, not a classification system. Clusters are not labeled as Vata, Pitta, or Kapha. The analysis asks whether observed cluster profiles are directionally consistent with literature descriptions — it does not assert that any cluster is a Dosha type. Non-significant statistical tests do not disqualify directional consistency.

| File | Description |
| :--- | :--- |
| `cohort_a_literature_concordance.csv` | Feature-level concordance table for Cohort A clusters |
| `cohort_b_literature_concordance.csv` | Feature-level concordance table for Cohort B clusters |
| `literature_concordance_summary.csv` | Combined summary across both cohorts |

The operationalization methodology that defines which literature characteristics correspond to which physiological variables is documented in `docs/Research2_Final_Operationalization.md`.

## sensitivity/

These files provide stability and consistency evidence for the characterization findings.

| File | Description |
| :--- | :--- |
| `cross_model_pairwise_ari.csv` | Adjusted Rand Index (ARI) between K-Means, GMM, Spectral, and Hierarchical cluster assignments at each candidate K, for each matrix — describes how consistently different model families identify the same partition |
| `raw_vs_log_feature_persistence.csv` | Feature-level comparison of cluster displacement direction and magnitude between RAW and LOG representations — describes how stable the characterization findings are across transformation choices |

These are sensitivity indicators, not validation certificates. The cross-model ARI is moderate, reflecting the mixed convergence documented in Decision 014.

## plots/

Six figures summarising the Stage 8 findings.

| File | Description |
| :--- | :--- |
| `fig1_cohort_a_zscore_heatmap.png` | Z-score heatmap of cluster mean profiles across all Cohort A features (K=2) |
| `fig2_cohort_b_zscore_heatmap.png` | Z-score heatmap of cluster mean profiles across all Cohort B features (K=3) |
| `fig3_cohort_a_discriminating_features_boxplots.png` | Boxplots for the features with the largest between-cluster effect sizes in Cohort A |
| `fig4_cohort_b_discriminating_features_boxplots.png` | Boxplots for the features with the largest between-cluster effect sizes in Cohort B |
| `fig5_derived_metabolic_ratios.png` | Cluster comparison of derived metabolic ratios (HOMA-IR, TC/HDL, TG/HDL) — audit only; these were not used as clustering features |
| `fig6_literature_concordance_matrix.png` | Summary matrix of directional concordance findings across Doshas, clusters, and both cohorts |

## metadata.json

Records Stage 8 run provenance and data integrity checks. Key fields:

- Source cluster assignment files (K-Means, A_RAW for Cohort A; K-Means, B_RAW for Cohort B)
- Row counts for both cohorts at time of join
- Join integrity status: **100% participant match** confirmed for both Cohort A (N=4,482) and Cohort B (N=967)
- Anti-circularity verification: confirms no Dosha biomarker labels or Ayurvedic cluster labels were introduced during characterization
