"""
run_feature_matrices.py
========================
Top-level entry point to construct and validate the four candidate feature matrices:
  - output/feature_matrices/A_RAW.csv
  - output/feature_matrices/A_LOG.csv
  - output/feature_matrices/B_RAW.csv
  - output/feature_matrices/B_LOG.csv
  - output/feature_matrices/feature_matrices_metadata.json
"""

from pathlib import Path
from pipeline.feature_matrices import run_feature_matrix_pipeline

def main():
    base_dir = Path(__file__).resolve().parent
    cohort_a_path = base_dir / "output" / "analysis_cohort_a_broad.csv"
    cohort_b_path = base_dir / "output" / "analysis_cohort_b_fasting.csv"
    output_dir = base_dir / "output" / "feature_matrices"

    run_feature_matrix_pipeline(
        cohort_a_path=cohort_a_path,
        cohort_b_path=cohort_b_path,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()
