"""
run_pipeline.py
===============
Entry point for the NHANES merge pipeline.

Usage:
    python run_pipeline.py

Optional flags:
    python run_pipeline.py --explore
        Prints all available columns in every CSV file.
        Useful when deciding which variables to add to config.py.
"""

import sys
import argparse
from pipeline.merge import run_merge_pipeline
from pipeline.utils import list_all_columns
from pipeline.config import DATASET_DIR, FILE_MAP


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NHANES Merge Pipeline — Research 2: The Bioinformatics Bridge"
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Print all available columns in every CSV file and exit.",
    )
    args = parser.parse_args()

    if args.explore:
        # Exploration mode: list every available column without running the pipeline
        list_all_columns(DATASET_DIR, FILE_MAP)
        sys.exit(0)

    # Normal mode: run the full merge pipeline
    merged_df = run_merge_pipeline()

    print(f"\n  Pipeline completed successfully.")
    print(f"  Merged dataframe has {len(merged_df):,} rows and "
          f"{len(merged_df.columns)} columns.\n")


if __name__ == "__main__":
    main()
