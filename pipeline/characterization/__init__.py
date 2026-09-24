"""
Stage 8 — Downstream Phenotype Characterisation and Literature Concordance package.
"""

from .data_loader import load_characterization_dataset, load_all_assignments
from .profiling import compute_cluster_membership, compute_feature_profiles
from .statistics import run_cohort_statistics
from .literature_concordance import evaluate_literature_concordance
from .sensitivity import run_sensitivity_analysis
from .visualizations import generate_all_plots

__all__ = [
    "load_characterization_dataset",
    "load_all_assignments",
    "compute_cluster_membership",
    "compute_feature_profiles",
    "run_cohort_statistics",
    "evaluate_literature_concordance",
    "run_sensitivity_analysis",
    "generate_all_plots",
]
