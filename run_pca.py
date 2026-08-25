"""
run_pca.py
==========
Top-level entry point to perform complete independent exploratory PCA on the four candidate scaled feature matrices:
  - output/scaled_matrices/A_RAW_scaled.csv (4,482 x 19)
  - output/scaled_matrices/A_LOG_scaled.csv (4,482 x 19)
  - output/scaled_matrices/B_RAW_scaled.csv (967 x 24)
  - output/scaled_matrices/B_LOG_scaled.csv (967 x 24)

Outputs exported to:
  - output/pca/
  - output/pca/plots/
"""

from pathlib import Path
from pipeline.pca import run_pca_exploration_pipeline

def main():
    base_dir = Path(__file__).resolve().parent
    scaled_dir = base_dir / "output" / "scaled_matrices"
    cohort_a_path = base_dir / "output" / "analysis_cohort_a_broad.csv"
    cohort_b_path = base_dir / "output" / "analysis_cohort_b_fasting.csv"
    output_dir = base_dir / "output" / "pca"

    run_pca_exploration_pipeline(
        scaled_dir=scaled_dir,
        cohort_a_path=cohort_a_path,
        cohort_b_path=cohort_b_path,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()
