# src/pgs_net/modules/__init__.py
"""Imports sub-modules for PGS-Net components."""

from .interfaces import (
    NeighborSearch,
    DynamicKController,
    FormalForceCalculator,
    MetaConfigLearner,
    Normalization,
    Regularization,
)
from .placeholders import PlaceholderDynamicKController, PlaceholderFormalForce, PlaceholderMetaConfig
from .complex_utils import ComplexLinear, ComplexRMSNorm
from .adapters import InputAdapter, OutputAdapter, complex_project, complex_output
from .geometry_similarity import GeometrySimilarity
from .clustering_assignment import ClusteringAssignment
from .dynamic_k import UsageBasedDynamicKController, SplitMergeDynamicKController
from .queen_computation import QueenComputation
from .neighbor_search import NaiveNeighborSearch, FaissNeighborSearch
from .formal_force import PotentialEnergyForceV2  # Add SimplePotential too if kept
from .force_calculator import UpdateForceCalculator
from .normalization import RMSNormImpl, AdaptiveGroupNorm
from .regularization import OrthogonalRegularization, CentroidRepulsionRegularization, SpectralNormConstraint
from .integrator import UpdateIntegrator
from .meta_config import AdvancedHyperNetworkMetaConfig  # Add HyperNetwork too if kept
from .cross_layer import CrossLayerHandler
from .non_locality import NonLocalityModule

__all__ = [
    # Interfaces
    "NeighborSearch",
    "DynamicKController",
    "FormalForceCalculator",
    "MetaConfigLearner",
    "Normalization",
    "Regularization",
    # Placeholders
    "PlaceholderDynamicKController",
    "PlaceholderFormalForce",
    "PlaceholderMetaConfig",
    # Implementations
    "ComplexLinear",
    "ComplexRMSNorm",
    "InputAdapter",
    "OutputAdapter",
    "complex_project",
    "complex_output",
    "GeometrySimilarity",
    "ClusteringAssignment",
    "UsageBasedDynamicKController",
    "SplitMergeDynamicKController",
    "QueenComputation",
    "NaiveNeighborSearch",
    "FaissNeighborSearch",
    "PotentialEnergyForceV2",
    "UpdateForceCalculator",
    "RMSNormImpl",
    "AdaptiveGroupNorm",
    "OrthogonalRegularization",
    "CentroidRepulsionRegularization",
    "SpectralNormConstraint",
    "UpdateIntegrator",
    "AdvancedHyperNetworkMetaConfig",
    "CrossLayerHandler",
    "NonLocalityModule",
]
