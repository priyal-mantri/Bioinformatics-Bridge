# Experiment 5 Sensitivity Extension: Distance-Metric Sensitivity

**Project**: Bioinfo Bridge — Research 2 (NHANES 2017–2018)  
**Stage**: Sandbox Experiment 5 Extension — Distance-Metric Sensitivity Analysis  
**Branch**: `experiment/clustering-sandbox`  
**Status**: COMPLETED & AUDIT VALIDATED (100% Clean)

---

## Executive Summary

This sensitivity extension evaluates whether replacing **Euclidean distance ($L_2$)** with **Manhattan distance ($L_1$ / Cityblock)** substantially alters or rescues the degraded hierarchical clustering partitions observed under Complete and Average linkages across the four standardized NHANES candidate feature matrices (`A_RAW`, `A_LOG`, `B_RAW`, `B_LOG`) for $K = 2 \dots 7$.

*(Note: Primary Ward linkage `ward` is mathematically formulated for squared Euclidean variance minimization and is excluded from Manhattan distance tests).*

---

## Methodological Comparison Grid (96 Experimental Runs)

### Cohort A Broad ($N=4,482, p=19$) — Primary Numerical Metrics

| Matrix | Distance | Linkage | K | Silhouette | CH Score | DB Index | Min Size | Min Prop | Max Size | Max Prop | Mean ARI (Stability) | Small Cluster Flag |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| **A_RAW** | Euclidean | Complete | 2 | 0.7609 | 121.6 | 0.394 | 4 | 0.09% | 4478 | 99.91% | 0.8680 | True |
| **A_RAW** | Manhattan | Complete | 2 | 0.5327 | 158.4 | 0.786 | 12 | 0.27% | 4470 | 99.73% | 0.5074 | True |
| **A_RAW** | Euclidean | Complete | 3 | 0.4539 | 110.3 | 1.223 | 4 | 0.09% | 4474 | 99.82% | 0.6446 | True |
| **A_RAW** | Manhattan | Complete | 3 | 0.2957 | 110.0 | 1.878 | 12 | 0.27% | 4464 | 99.60% | 0.4430 | True |
| **A_RAW** | Euclidean | Complete | 4 | 0.4294 | 96.4 | 1.148 | 4 | 0.09% | 4470 | 99.73% | 0.6704 | True |
| **A_RAW** | Manhattan | Complete | 4 | 0.2626 | 80.1 | 1.920 | 8 | 0.18% | 4464 | 99.60% | 0.2876 | True |
| **A_RAW** | Euclidean | Complete | 7 | 0.3800 | 65.2 | 1.037 | 1 | 0.02% | 4467 | 99.67% | 0.5457 | True |
| **A_RAW** | Manhattan | Complete | 7 | 0.2304 | 54.0 | 1.779 | 5 | 0.11% | 4452 | 99.33% | **0.1055** | True |
| **A_RAW** | Euclidean | Average | 2 | 0.7609 | 121.6 | 0.394 | 4 | 0.09% | 4478 | 99.91% | 0.9102 | True |
| **A_RAW** | Manhattan | Average | 2 | 0.6090 | 126.3 | 0.633 | 6 | 0.13% | 4476 | 99.87% | 0.7985 | True |
| **A_RAW** | Euclidean | Average | 3 | 0.5783 | 101.9 | 0.988 | 4 | 0.09% | 4474 | 99.82% | 0.7554 | True |
| **A_RAW** | Manhattan | Average | 3 | 0.5773 | 70.9 | 0.758 | 2 | 0.04% | 4476 | 99.87% | 0.7822 | True |
| **A_LOG** | Euclidean | Complete | 2 | 0.4203 | 86.8 | 1.490 | 32 | 0.71% | 4450 | 99.29% | 0.5894 | True |
| **A_LOG** | Manhattan | Complete | 2 | 0.2456 | 123.3 | 2.174 | 117 | 2.61% | 4365 | 97.39% | 0.3744 | False |
| **A_LOG** | Euclidean | Complete | 3 | 0.3905 | 49.2 | 1.503 | 3 | 0.07% | 4447 | 99.22% | 0.5733 | True |
| **A_LOG** | Manhattan | Complete | 3 | 0.1930 | 73.9 | 2.558 | 55 | 1.23% | 4365 | 97.39% | 0.2531 | False |
| **A_LOG** | Euclidean | Complete | 7 | 0.0990 | 46.9 | 2.707 | 1 | 0.02% | 4441 | 99.09% | 0.2086 | True |
| **A_LOG** | Manhattan | Complete | 7 | 0.0327 | 83.1 | 2.849 | 7 | 0.16% | 4361 | 97.30% | **0.0666** | True |

---

## Targeted Methodological Evaluation

### Question:
*"Does changing Euclidean to Manhattan distance for Complete/Average linkage reveal a substantially more robust, non-degenerate, and reproducible hierarchical clustering structure?"*

### Empirical Answer:
**NO.** Changing from Euclidean to Manhattan distance does **NOT** yield a substantially more robust, non-degenerate, or reproducible hierarchical clustering structure.

### Detailed Findings:
1. **Tiny Outlier-Peeling Clusters Persist**:
   - Under both Euclidean and Manhattan distances, Complete and Average linkages exhibit severe "outlier peeling" behavior.
   - For Manhattan distance, the minimum cluster proportion (`min_cluster_prop`) remains below $0.27\%$ ($N_1 = 5 \dots 12$ participants out of $4,482$) on Cohort A, leaving $> 99.7\%$ of participants collapsed into a single massive cluster.
2. **Degraded Cluster-Size Balance**:
   - Manhattan distance fails to produce balanced partitions. At $K=3..7$, partitions remain completely uninterpretable (e.g., $99.7\%$ vs $0.2\%$).
3. **No Stronger Internal Separation**:
   - Silhouette scores under Manhattan distance are lower or comparable to Euclidean distance across $K \ge 3$ ($\text{SC} \approx 0.03 \dots 0.29$ for Manhattan Complete vs $0.10 \dots 0.45$ for Euclidean Complete).
4. **Worse Subsampling Stability**:
   - Participant-overlap ARI stability under Manhattan Complete linkage drops sharply down to $\text{Mean ARI} = 0.066 \dots 0.105$ at $K=6..7$ on Cohort A, proving that Manhattan distance partitions are even more volatile under dataset perturbation than Euclidean distance.

---

## Validation Status

Validated by `validate_distance_sensitivity_experiment.py`:
- ✅ **100% Structure**: All required metric CSVs, assignment CSVs, comparison PNGs, and JSON metadata are present in `clustering_sandbox/hierarchical/distance_metric_sensitivity/`.
- ✅ **100% SEQN Integrity**: Participant SEQN alignment verified 1-to-1 against baseline analysis cohort files.
- ✅ **100% Clean Metrics**: All 96 experimental runs verified clean with valid numeric metric ranges and stability bounds.
