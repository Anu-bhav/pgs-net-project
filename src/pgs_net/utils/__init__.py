# src/pgs_net/utils/__init__.py
"""Utility functions for PGS-Net."""

from .analysis_processing import (
    GLOBAL_DATA,
    HEAD_DATA,  # Export constants
    LAYER_IDX,
    PGS_FFN_DATA,
    STEP,
    flatten_analysis_dict,
    parse_tensor_list_column,
    process_run_analysis_data,
)
from .logging_utils import get_logger
from .viz_utils import (
    plot_correlation_heatmap,
    plot_dynamics_analysis,
    plot_embedding_visualization,
    plot_geometry_analysis,
    plot_training_curves,
)

# Only import if dependencies are met
try:
    from .projection_utils import project_embeddings

    _projection_available = True
except ImportError:
    _projection_available = False

    def project_embeddings(*args, **kwargs):
        get_logger("projection_utils").error("Projection libraries (umap-learn/scikit-learn) not installed.")
        return None


__all__ = [
    "get_logger",
    # Analysis Processing
    "flatten_analysis_dict",
    "parse_tensor_list_column",
    "process_run_analysis_data",
    "STEP",
    "LAYER_IDX",
    "PGS_FFN_DATA",
    "GLOBAL_DATA",
    "HEAD_DATA",
    # Visualization
    "plot_training_curves",
    "plot_geometry_analysis",
    "plot_dynamics_analysis",
    "plot_correlation_heatmap",
    "plot_embedding_visualization",
    # Projection
    "project_embeddings",
]

if not _projection_available:
    # Remove project_embeddings from __all__ if dependencies are missing
    __all__.remove("project_embeddings")
