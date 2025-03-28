# src/pgs_net/modules/regularization.py
"""Regularization loss terms and constraints."""

import torch
import torch.nn as nn
import logging
from typing import Optional
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
        logger.info(f"Initialized OrthogonalRegularization (Strength={strength}, Mode={mode})")

    def forward(self, W: torch.Tensor) -> torch.Tensor:  # Pass the parameter tensor (e.g., centroids)
        """Calculates the orthogonality penalty."""
        loss = torch.tensor(0.0, device=W.device, dtype=torch.float32)
        if self.strength <= 0:
            return loss

        if W.dim() != 2:  # Expects (NumItems, Dim) or (Dim, NumItems)
            logger.warning(f"OrthogonalRegularization expects 2D tensor, got {W.dim()}D. Skipping.")
            return loss
        if W.shape[0] < 2 or W.shape[1] < 2:
            return loss  # Need multiple items/dims

        mat = W.t() if self.mode == "cols" else W  # Operate on rows (NumItems, Dim)
        num_items = mat.shape[0]

        # Calculate Gram matrix: G = W @ W.T
        # Handle complex numbers: G = W @ W.conj().T
        if mat.is_complex():
            gram = torch.matmul(mat, mat.conj().t())
        else:
            gram = torch.matmul(mat, mat.t())

        # Penalize off-diagonal elements (deviation from identity * scale)
        identity = torch.eye(num_items, device=gram.device, dtype=gram.dtype)  # Match dtype
        # L2 penalty on off-diagonal elements
        off_diag_penalty = (gram * (1 - identity)).abs().pow(2).sum() / max(
            1, num_items * (num_items - 1)
        )  # Average off-diagonal squared mag

        # Optionally penalize diagonal deviation from 1? No, allow varying norms.
        loss = self.strength * off_diag_penalty.float()  # Ensure float loss
        return loss

    @torch.no_grad()
    def apply_constraints(self, module: nn.Module) -> None:
        pass  # No constraints applied here


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
        logger.info(f"Initialized CentroidRepulsionRegularization (Strength={strength})")

    def forward(self, C: torch.Tensor) -> torch.Tensor:  # Pass centroids tensor (K, D)
        """Calculates the repulsion energy loss."""
        loss = torch.tensor(0.0, device=C.device, dtype=torch.float32)
        if self.strength <= 0:
            return loss
        if C.dim() != 2 or C.shape[0] <= 1:
            return loss  # Need >= 2 centroids
        K = C.shape[0]

        # Pairwise squared distances (complex-aware)
        diffs = C.unsqueeze(1) - C.unsqueeze(0)  # (K, K, D)
        dist_sq = diffs.abs().pow(2).sum(dim=-1)  # (K, K) Real distances sq

        # Inverse distance squared potential (avoid self, add eps)
        potential = 1.0 / (dist_sq + self.eps)
        potential.diagonal().fill_(0)

        # Sum upper triangle only
        total_repulsion_energy = torch.triu(potential, diagonal=1).sum()
        loss = self.strength * total_repulsion_energy.float()  # Ensure float loss
        return loss

    @torch.no_grad()
    def apply_constraints(self, module: nn.Module) -> None:
        pass


class SpectralNormConstraint(Regularization):
    """Applies spectral normalization as a constraint via torch utility."""

    def __init__(self, name: str = "weight", n_power_iterations: int = 1, dim: int = 0):
        """
        Relies on applying torch.nn.utils.parametrizations.spectral_norm
        to the target layer/parameter during model initialization.

        Args:
            name (str): Name of the weight parameter in the module (e.g., 'weight').
            n_power_iterations (int): Number of power iterations for estimation.
            dim (int): Dimension corresponding to the input channel for Conv layers.
        """
        super().__init__()
        self.name = name
        self.n_power_iterations = n_power_iterations
        self.dim = dim
        logger.info(f"Initialized SpectralNormConstraint (TargetParam={name}). Requires explicit application via torch.nn.utils.")

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # This regularization adds no loss term itself; constraint applied elsewhere.
        return torch.tensor(0.0)

    @torch.no_grad()
    def apply_constraints(self, module: nn.Module) -> None:
        # Constraint is typically applied via torch.nn.utils.parametrizations.spectral_norm
        # during model initialization or via hooks during forward pass.
        # No operation needed here usually after optimizer step.
        logger.debug(f"SpectralNormConstraint.apply_constraints called (No-op - Assumes applied via torch utils).")
        pass
