# src/pgs_net/modules/regularization.py
"""Regularization loss terms and constraints."""

import logging
from typing import Optional

import torch
import torch.nn as nn

from .interfaces import Regularization

logger = logging.getLogger(__name__)


class OrthogonalRegularization(Regularization):
    """Penalizes non-orthogonality of rows/columns in a weight matrix (e.g., centroids)."""

    def __init__(self, strength: float = 0.01, mode: str = "rows"):
        """
        Args:
            strength (float): Weight of the regularization loss.
            mode (str): 'rows' or 'cols' to orthogonalize.

        """
        super().__init__()
        self.strength = strength
        self.mode = mode
        if self.strength > 0:
            logger.info(f"Initialized OrthogonalRegularization (Strength={strength:.2g}, Mode={mode})")

    def forward(self, W: torch.Tensor) -> torch.Tensor:  # Pass the parameter tensor (e.g., centroids)
        """Calculates the orthogonality penalty."""
        loss = torch.tensor(0.0, device=W.device, dtype=torch.float32)
        if self.strength <= 0:
            return loss

        if W.dim() != 2:
            logger.warning(f"OrthogonalRegularization expects 2D tensor, got {W.dim()}D. Skipping.")
            return loss
        num_items, dim = W.shape
        if self.mode == "cols":
            W = W.t()
            num_items, dim = dim, num_items  # Operate on rows
        if num_items < 2:
            return loss  # Need multiple items to compare

        # Calculate Gram matrix: G = W @ W.conj().T
        if W.is_complex():
            gram = torch.matmul(W, W.conj().t())
        else:
            gram = torch.matmul(W, W.t())

        # Penalize off-diagonal elements (deviation from identity * scale)
        identity = torch.eye(num_items, device=gram.device, dtype=gram.dtype)
        # L2 penalty on off-diagonal elements' magnitude squared
        off_diag_penalty = (gram * (1 - identity)).abs().pow(2).sum()
        # Normalize by number of off-diagonal pairs
        num_pairs = num_items * (num_items - 1)
        avg_off_diag_penalty = off_diag_penalty / max(1.0, num_pairs)

        loss = self.strength * avg_off_diag_penalty.float()  # Ensure float loss
        return loss

    @torch.no_grad()
    def apply_constraints(self, module: nn.Module) -> None:
        pass  # No constraints applied here

    def extra_repr(self) -> str:
        return f"strength={self.strength:.2g}, mode={self.mode}"


class CentroidRepulsionRegularization(Regularization):
    """Penalizes closeness between centroids using inverse square potential."""

    def __init__(self, strength: float = 0.001, eps: float = 1e-6):
        """
        Args:
            strength (float): Weight of the regularization loss.
            eps (float): Smoothing epsilon for inverse distance.

        """
        super().__init__()
        self.strength = strength
        self.eps = eps
        if self.strength > 0:
            logger.info(f"Initialized CentroidRepulsionRegularization (Strength={strength:.2g}, eps={eps:.1g})")

    def forward(self, C: torch.Tensor) -> torch.Tensor:  # Pass centroids tensor (K, D)
        """Calculates the repulsion energy loss."""
        loss = torch.tensor(0.0, device=C.device, dtype=torch.float32)
        if self.strength <= 0:
            return loss
        if C.dim() != 2 or C.shape[0] <= 1:
            return loss  # Need >= 2 centroids
        K = C.shape[0]

        try:
            # Pairwise squared distances (complex-aware)
            diffs = C.unsqueeze(1) - C.unsqueeze(0)  # (K, K, D)
            dist_sq = diffs.abs().pow(2).sum(dim=-1)  # (K, K) Real distances sq
            dist_sq = dist_sq.float()  # Ensure float for division

            # Inverse distance squared potential (avoid self, add eps)
            potential = 1.0 / (dist_sq + self.eps)
            potential.diagonal().fill_(0)

            # Sum upper triangle only, normalize by number of pairs
            num_pairs = K * (K - 1) / 2.0
            total_repulsion_energy = torch.triu(potential, diagonal=1).sum() / max(1.0, num_pairs)
            loss = self.strength * total_repulsion_energy.float()  # Ensure float loss
        except Exception as e:
            logger.error(f"CentroidRepulsionRegularization failed: {e}", exc_info=True)
            loss = torch.tensor(0.0, device=C.device, dtype=torch.float32)

        return loss

    @torch.no_grad()
    def apply_constraints(self, module: nn.Module) -> None:
        pass

    def extra_repr(self) -> str:
        return f"strength={self.strength:.2g}, eps={self.eps:.1g}"


class SpectralNormConstraint(Regularization):
    """Applies spectral normalization as a constraint via torch utility."""

    def __init__(self, name: str = "weight", n_power_iterations: int = 1, dim: int = 0):
        """Relies on applying torch.nn.utils.parametrizations.spectral_norm at layer init."""
        super().__init__()
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.dim = dim
        logger.info(f"Initialized SpectralNormConstraint (TargetParam={name}). Requires explicit application via torch.nn.utils.")

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return torch.tensor(0.0)  # No loss term added

    @torch.no_grad()
    def apply_constraints(self, module: nn.Module) -> None:
        logger.debug(f"SpectralNormConstraint.apply_constraints called (No-op - Assumes applied via torch utils).")
        pass

    def extra_repr(self) -> str:
        return f"target_param={self.name}, n_iter={self.n_power_iterations}"
