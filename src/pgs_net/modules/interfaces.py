# src/pgs_net/modules/interfaces.py
"""Abstract Base Classes for pluggable PGS-Net components."""

import abc
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Union


class NeighborSearch(nn.Module, abc.ABC):
    """Abstract base class for neighbor search algorithms."""

    @abc.abstractmethod
    def find_neighbors(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Finds k nearest neighbors. Returns indices (B, T, k) and vecs (B, T, k, D) or None."""
        pass


class DynamicKController(nn.Module, abc.ABC):
    """Abstract base class for dynamic cluster count controllers."""

    @abc.abstractmethod
    @torch.no_grad()
    def update_k(
        self, current_assignments: Optional[torch.Tensor], centroids: Optional[torch.Tensor], x_h: Optional[torch.Tensor]
    ) -> None:
        """Update internal state and potentially the active cluster mask."""
        pass

    @abc.abstractmethod
    def get_active_indices(self) -> torch.Tensor:
        """Returns a 1D LongTensor containing the indices of the currently active clusters."""
        pass


class FormalForceCalculator(nn.Module, abc.ABC):
    """Abstract base class for force calculation derived from a potential energy."""

    @abc.abstractmethod
    def calculate_force(
        self,
        x_h: torch.Tensor,
        A: torch.Tensor,
        local_queens: torch.Tensor,
        global_queen: torch.Tensor,
        centroids_h: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates the force acting on tokens x_h = -gradient(Energy, x_h)."""
        pass


class MetaConfigLearner(nn.Module, abc.ABC):
    """Abstract base class for meta-learning hyperparameters."""

    @abc.abstractmethod
    def get_dynamic_config(self, current_state: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, float, int, bool]]:
        """Generates dynamic hyperparameter values based on the current training state."""
        pass


class Normalization(nn.Module, abc.ABC):
    """Abstract base class for normalization layers."""

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Regularization(nn.Module, abc.ABC):
    """Abstract base class for regularization loss terms or constraints."""

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Calculate and return the regularization loss term (scalar float tensor)."""
        pass

    @abc.abstractmethod
    @torch.no_grad()
    def apply_constraints(self, module: nn.Module) -> None:
        """Apply constraints directly to module parameters (e.g., weight clipping)."""
        pass
