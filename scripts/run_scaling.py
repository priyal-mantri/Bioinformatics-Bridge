"""
run_scaling.py
==============
Top-level entry point to perform Z-score standardization on the four feature matrices:
  - output/scaled_matrices/A_RAW_scaled.csv
  - output/scaled_matrices/A_LOG_scaled.csv
  - output/scaled_matrices/B_RAW_scaled.csv
  - output/scaled_matrices/B_LOG_scaled.csv
  - output/scaled_matrices/scaled_matrices_metadata.json
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pathlib import Path
from pipeline.scaling import run_scaling_pipeline

def main():
    base_dir = Path(__file__).resolve().parent
    matrix_dir = base_dir / "output" / "feature_matrices"
    cohort_a_path = base_dir / "output" / "analysis_cohort_a_broad.csv"
    cohort_b_path = base_dir / "output" / "analysis_cohort_b_fasting.csv"
    output_dir = base_dir / "output" / "scaled_matrices"

    run_scaling_pipeline(
        matrix_dir=matrix_dir,
        cohort_a_path=cohort_a_path,
        cohort_b_path=cohort_b_path,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()