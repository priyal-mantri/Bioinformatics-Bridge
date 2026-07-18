# pipeline/__init__.py
# Makes 'pipeline' a proper Python package.
# Exposes the main entry point for clean imports.

from pipeline.merge import run_merge_pipeline

__all__ = ["run_merge_pipeline"]
