# output/pca/

This directory contains the outputs of Stage 5: exploratory Principal Component Analysis (PCA) applied to the four standardized feature matrices. PCA here is an audit and orientation step, not the primary analysis. Downstream clustering was performed on the full standardized feature space.

## Files

### Score matrices (participant projections)

| File | Description |
| :--- | :--- |
| `A_RAW_pca_scores.csv` | PC scores for all 4,482 Cohort A participants (RAW representation) |
| `A_LOG_pca_scores.csv` | PC scores for all 4,482 Cohort A participants (LOG representation) |
| `B_RAW_pca_scores.csv` | PC scores for all 967 Cohort B participants (RAW representation) |
| `B_LOG_pca_scores.csv` | PC scores for all 967 Cohort B participants (LOG representation) |

### Loading matrices (feature contributions)

| File | Description |
| :--- | :--- |
| `A_RAW_pca_loadings.csv` | Feature loadings on each PC, Cohort A RAW (19 features × 19 PCs) |
| `A_LOG_pca_loadings.csv` | Feature loadings on each PC, Cohort A LOG (19 features × 19 PCs) |
| `B_RAW_pca_loadings.csv` | Feature loadings on each PC, Cohort B RAW (24 features × 24 PCs) |
| `B_LOG_pca_loadings.csv` | Feature loadings on each PC, Cohort B LOG (24 features × 24 PCs) |

### Variance tables

| File | Description |
| :--- | :--- |
| `A_RAW_pca_variance.csv` | Explained variance and cumulative variance per PC, Cohort A RAW |
| `A_LOG_pca_variance.csv` | Explained variance and cumulative variance per PC, Cohort A LOG |
| `B_RAW_pca_variance.csv` | Explained variance and cumulative variance per PC, Cohort B RAW |
| `B_LOG_pca_variance.csv` | Explained variance and cumulative variance per PC, Cohort B LOG |

### Metadata

| File | Description |
| :--- | :--- |
| `pca_exploration_metadata.json` | Full PCA run record: variance by component, Kaiser criterion results, recorded methodological decision |

### Diagnostic plots (`plots/`)

Three plot types were generated per matrix (12 plots total):

| Pattern | Description |
| :--- | :--- |
| `*_scree_plot.png` | Eigenvalue scree plot — variance explained per component |
| `*_cum_var_plot.png` | Cumulative variance curve — variance retained as components are added |
| `*_pc1_pc2_scores.png` | Scatter plot of participant projections onto PC1 and PC2 |

## Key Finding and Decision

Variance is broadly distributed across components with no dominant axis. PC1 explains approximately 14.5%–18.4% of total variance depending on the matrix. Under the Kaiser criterion (eigenvalue ≥ 1.0), 7 components are retained for Cohort A (~62% variance) and 8–9 components for Cohort B (~67%–69% variance).

**Decision 013**: Because a small number of principal components captures only a limited fraction of total variance, PCA was not used as the primary input space for downstream clustering. Clustering proceeds on the full standardized feature matrices in `output/scaled_matrices/`. The complete PCA solutions here are preserved as an exploratory audit trail and are available for post-hoc interpretation and visualization.

This decision and its supporting variance statistics are recorded in `pca_exploration_metadata.json` and `docs/methodology_decision_log.md`.
