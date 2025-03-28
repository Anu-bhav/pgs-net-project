# src/pgs_net/modules/formal_force.py
"""Formal Force calculation using Potential Energy gradients."""

import torch
import torch.nn as nn
import logging
import time
from typing import Dict, Optional
from .interfaces import FormalForceCalculator

logger = logging.getLogger(__name__)


class PlaceholderFormalForce(FormalForceCalculator):
    """Placeholder: Returns zero force."""

    def __init__(self, config: Dict, is_complex: bool, dtype: torch.dtype):
        super().__init__()
        logger.warning("Using PlaceholderFormalForce.")

    def calculate_force(self, x_h, A, local_queens, global_queen, centroids_h):
        return torch.zeros_like(x_h)


class PotentialEnergyForceV2(FormalForceCalculator):
    """Calculates force from potential including token repulsion."""

    def __init__(self, config: Dict, is_complex: bool, dtype: torch.dtype):
        """
        Initializes the PotentialEnergyForceV2 calculator.

        Args:
            config (Dict): Configuration dictionary (expects 'update_forces.formal_force_params').
            is_complex (bool): Whether inputs are complex.
            dtype (torch.dtype): Data type (real or complex).
        """
        super().__init__()
        self.params = config.get("update_forces", {}).get("formal_force_params", {})
        self.attr_centroid_w = self.params.get("attraction_centroid_weight", 1.0)
        self.attr_global_w = self.params.get("attraction_global_weight", 0.1)
        self.repel_token_w = self.params.get("token_repulsion_weight", 0.0001)
        self.repel_radius_sq = self.params.get("token_repulsion_radius", 1.0) ** 2
        self.repel_potential_type = self.params.get("repulsion_potential_type", "smoothed_inverse_sq")
        self.repel_eps = self.params.get("repulsion_smooth_eps", 1e-3)
        self.align_w = self.params.get("alignment_weight", 0.0)  # Alignment weight
        self.is_complex = is_complex
        self.dtype = dtype
        logger.info(
            f"Initialized PotentialEnergyForceV2 (AttrCen={self.attr_centroid_w}, AttrGlob={self.attr_global_w}, RepelTok={self.repel_token_w}, Align={self.align_w})."
        )

    def calculate_force(
        self,
        x_h: torch.Tensor,
        A: torch.Tensor,
        local_queens: Optional[torch.Tensor],
        global_queen: Optional[torch.Tensor],
        centroids_h: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates force = -gradient(Energy, x_h)."""
        # Create a new tensor that requires grad for the gradient calculation
        x_h_input = x_h.detach().clone().requires_grad_(True)
        B, T, D = x_h_input.shape
        K = centroids_h.shape[0]

        total_energy = torch.tensor(0.0, device=x_h.device, dtype=torch.float32)  # Energy must be scalar float

        # --- Define Potential Energy E w.r.t x_h_input ---
        energy_terms = {}  # For analysis

        # 1. Attraction to Centroids (weighted by assignment A)
        if self.attr_centroid_w > 0:
            # Use squared distance potential: E = w * A * ||x-c||^2
            diff_to_centroids = x_h_input.unsqueeze(2) - centroids_h.detach().unsqueeze(0).unsqueeze(
                0
            )  # Treat centroids as fixed points for this gradient
            dist_sq_to_centroids = diff_to_centroids.abs().pow(2).sum(dim=-1)  # B, T, K (Real)
            energy_attr = (A.detach() * dist_sq_to_centroids).sum()  # Sum over B, T, K. Detach A.
            total_energy = total_energy + self.attr_centroid_w * energy_attr
            energy_terms["E_attr_centroid"] = energy_attr.item()

        # 2. Attraction to Global Queen
        if self.attr_global_w > 0 and global_queen is not None:
            # Use detached global queen
            diff_to_global = x_h_input - global_queen.detach().unsqueeze(1)  # (B,T,D)
            dist_sq_to_global = diff_to_global.abs().pow(2).sum(dim=-1)  # (B,T) Real
            energy_global_attr = dist_sq_to_global.sum()  # Sum over B, T
            total_energy = total_energy + self.attr_global_w * energy_global_attr
            energy_terms["E_attr_global"] = energy_global_attr.item()

        # 3. Pairwise Token Repulsion (within radius, smoothed)
        if self.repel_token_w > 0 and T > 1:
            logger.debug("Calculating Token Repulsion Energy...")
            x_flat = x_h_input.view(B * T, D)
            # Calculate pairwise squared distances efficiently
            dist_sq_flat = torch.zeros((B * T, B * T), device=x_h.device, dtype=torch.float32)  # Placeholder type
            try:
                if self.is_complex:
                    x_norm_sq = x_flat.abs().pow(2).sum(-1)
                    xy_dot_real = torch.real(torch.matmul(x_flat, x_flat.t().conj()))
                    dist_sq_flat = x_norm_sq.unsqueeze(1) + x_norm_sq.unsqueeze(0) - 2 * xy_dot_real
                else:
                    # cdist can be memory intensive for large BT
                    # Manual calculation might be better if T is large
                    # dist_sq_flat = torch.cdist(x_flat, x_flat, p=2).pow(2)
                    x_norm_sq = x_flat.pow(2).sum(-1)
                    xy_dot = torch.matmul(x_flat, x_flat.t())
                    dist_sq_flat = x_norm_sq.unsqueeze(1) + x_norm_sq.unsqueeze(0) - 2 * xy_dot
                dist_sq_flat = dist_sq_flat.float().clamp(min=0)  # Ensure real float and non-negative

                radius_mask = (dist_sq_flat < self.repel_radius_sq).float()
                radius_mask.fill_diagonal_(0)

                if self.repel_potential_type == "lennard_jones_stub":
                    logger.warning("Lennard-Jones potential stub.")
                    potential = self.repel_token_w * radius_mask / (dist_sq_flat + self.repel_eps)  # Fallback
                else:  # 'smoothed_inverse_sq'
                    potential = self.repel_token_w * radius_mask / (dist_sq_flat + self.repel_eps)

                # Sum upper triangle only
                energy_repel = torch.triu(potential, diagonal=1).sum()
                total_energy = total_energy + energy_repel
                energy_terms["E_repel_token"] = energy_repel.item()
                logger.debug(f"FormalForceV2: Token Repulsion Energy = {energy_repel.item()}")
            except Exception as e:
                logger.error(f"Token Repulsion energy calculation failed: {e}", exc_info=True)

        # 4. Alignment Energy (Requires velocity state - Placeholder)
        if self.align_w > 0:
            logger.warning("FormalForceV2: Alignment energy term requires velocity state, not implemented.")
            # Placeholder: E_align = w_align * sum_{i,j} mask_ij * (1 - cos(v_i, v_j))

        logger.debug(f"FormalForceV2 - Total Energy Terms: {energy_terms}")

        # --- Calculate Gradient ---
        if total_energy == 0.0 or not x_h_input.requires_grad:
            logger.debug("FormalForceV2: Total energy is zero or input requires no grad. Force is zero.")
            return torch.zeros_like(x_h)

        start_grad = time.time()
        try:
            grad_outputs = torch.ones_like(total_energy)
            gradients = torch.autograd.grad(
                outputs=total_energy,
                inputs=x_h_input,
                grad_outputs=grad_outputs,
                retain_graph=False,  # No need to retain graph for update rule
                create_graph=False,  # Don't need gradients of gradients
                allow_unused=True,
            )[0]
            grad_time = time.time() - start_grad
            logger.debug(f"FormalForceV2: Gradient calculation took {grad_time:.4f} sec.")

            if gradients is None:
                logger.warning("FormalForceV2: Gradient computation returned None. Returning zero force.")
                return torch.zeros_like(x_h)

            # Force is the negative gradient
            force = -gradients
            force_norm = torch.linalg.vector_norm(force, dim=-1).mean().item()
            logger.debug(f"FormalForceV2: Calculated force avg norm = {force_norm:.4f}")

            # Detach the final force tensor from the computation graph involving the energy potential
            return force.detach()

        except Exception as e:
            logger.error(f"FormalForceV2: Gradient calculation failed: {e}", exc_info=True)
            return torch.zeros_like(x_h)  # Return zero force on error
