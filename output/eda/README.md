# output/eda/

This directory contains exploratory data analysis (EDA) outputs produced before feature matrix construction. EDA was conducted as an orientation and audit step to understand the structure of the merged and preprocessed dataset prior to any clustering decisions. It does not constitute primary evidence for any research conclusion.

## Contents

```
eda/
├── EDA_SUMMARY.md                       # Top-level narrative summary of EDA findings
├── overview/
│   └── overview_summary.txt             # Dataset shape, column list, basic summary statistics
├── distributions/
│   ├── distribution_report.txt          # Skewness, kurtosis, range per variable
│   ├── distributions_summary.csv        # Per-variable distribution statistics table
│   ├── distributions_summary_grid.png   # Grid of variable histograms
│   └── individual/                      # Per-variable histogram plots
├── missingness/
│   ├── missing_values_table.csv         # Missing count and percentage per column
│   ├── missingness_report.txt           # Narrative description of missingness patterns
│   ├── missingness_heatmap.png          # Heatmap of missing value patterns across columns
│   └── missingness_correlation.png      # Co-occurrence pattern of missing values across variables
├── correlations/
│   ├── correlation_matrix.csv           # Pearson correlation matrix across all features
│   ├── high_correlations.csv            # Pairs exceeding a high-correlation threshold
│   ├── correlation_heatmap.png          # Heatmap of the correlation matrix
│   └── correlation_report.txt           # Narrative description of notable correlations
├── demographics/
│   ├── demographics_report.txt          # Age, sex, and ethnicity summaries
│   ├── age_distribution.png
│   ├── gender_distribution.png
│   ├── ethnicity_distribution.png
│   ├── biomarkers_by_age_group.csv
│   ├── biomarkers_by_age_group.png
│   ├── biomarkers_by_gender.csv
│   └── biomarkers_by_gender.png
└── outliers/
    ├── outlier_counts.csv               # Per-variable extreme value counts
    ├── outlier_report.txt               # Description of outlier patterns
    ├── outliers_summary_grid.png        # Grid of boxplots highlighting extremes
    └── individual/                      # Per-variable outlier boxplots
```

## Categories

**Overview**: Basic shape and summary statistics for the merged preprocessed dataset.

**Distributions**: Histograms and skewness/kurtosis statistics for each variable. The skewness results informed the log1p transformation decisions in Stage 3, which are documented formally in `docs/variable_selection_analysis.md`.

**Missingness**: Patterns of missing data across variables. Missingness varies substantially between non-fasting and fasting-only biomarkers, which is the primary reason for the dual cohort design (Cohort A / Cohort B).

**Correlations**: Pairwise Pearson correlations across all candidate features. Notable correlations (e.g., between derived ratios and their constituent biomarkers) were relevant to the decision to exclude derived ratios from feature matrices.

**Demographics**: Distribution of age, sex, and ethnicity in the cohort, and stratified biomarker comparisons. These are descriptive only and were not used as clustering features.

**Outliers**: Extreme values identified using interquartile range criteria. No outlier removal was applied — participants with physiologically plausible extreme values are retained. This choice is documented in the preprocessing decision log.
