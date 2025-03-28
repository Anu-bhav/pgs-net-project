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
            config (Dict): Full PGS_FFN configuration dictionary.
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
        self.align_w = self.params.get("alignment_weight", 0.0)  # Alignment weight (requires velocity state)
        self.is_complex = is_complex
        self.dtype = dtype
        logger.info(
            f"Initialized PotentialEnergyForceV2 (AttrCen={self.attr_centroid_w}, AttrGlob={self.attr_global_w}, RepelTok={self.repel_token_w}, AlignW={self.align_w})."
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
        # Ensure input requires grad for autograd
        x_h_input = x_h if x_h.requires_grad else x_h.detach().clone().requires_grad_(True)
        B, T, D = x_h_input.shape
        if K_max := centroids_h.shape[0] == 0:
            return torch.zeros_like(x_h)  # Handle case with no centroids

        total_energy = torch.tensor(0.0, device=x_h.device, dtype=torch.float32)  # Energy must be scalar float
        energy_terms = {}  # For analysis/debugging

        # Use detached assignments and queen/centroids for potential calculation
        A_detached = A.detach().float()  # Ensure float
        centroids_detached = centroids_h.detach()
        global_queen_detached = global_queen.detach() if global_queen is not None else None

        # 1. Attraction to Centroids (weighted by A)
        if self.attr_centroid_w > 0:
            diff_to_centroids = x_h_input.unsqueeze(2) - centroids_detached.unsqueeze(0).unsqueeze(0)  # B, T, K, D
            # Use squared distance potential: E = w * A * ||x-c||^2
            dist_sq_to_centroids = diff_to_centroids.abs().pow(2).sum(dim=-1)  # B, T, K (Real)
            energy_attr = (A_detached * dist_sq_to_centroids).sum() / (B * T)  # Average energy per token
            total_energy = total_energy + self.attr_centroid_w * energy_attr
            energy_terms["E_attr_centroid"] = energy_attr.item()

        # 2. Attraction to Global Queen
        if self.attr_global_w > 0 and global_queen_detached is not None:
            diff_to_global = x_h_input - global_queen_detached.unsqueeze(1)  # (B,T,D)
            dist_sq_to_global = diff_to_global.abs().pow(2).sum(dim=-1)  # (B,T) Real
            energy_global_attr = dist_sq_to_global.sum() / (B * T)  # Average energy per token
            total_energy = total_energy + self.attr_global_w * energy_global_attr
            energy_terms["E_attr_global"] = energy_global_attr.item()

        # 3. Pairwise Token Repulsion (within radius, smoothed)
        if self.repel_token_w > 0 and T > 1:
            logger.debug("Calculating Token Repulsion Energy...")
            repulsion_energy_val = 0.0
            try:
                x_flat = x_h_input.view(B * T, D)
                # Calculate pairwise squared distances efficiently (handle complex)
                dist_sq_flat = torch.zeros((B * T, B * T), device=x_h.device, dtype=torch.float32)
                if self.is_complex:
                    x_norm_sq = x_flat.abs().pow(2).sum(-1)
                    xy_dot_real = torch.real(torch.matmul(x_flat, x_flat.t().conj()))
                    dist_sq_flat = x_norm_sq.unsqueeze(1) + x_norm_sq.unsqueeze(0) - 2 * xy_dot_real
                else:
                    x_norm_sq = x_flat.pow(2).sum(-1)
                    xy_dot = torch.matmul(x_flat, x_flat.t())
                    dist_sq_flat = x_norm_sq.unsqueeze(1) + x_norm_sq.unsqueeze(0) - 2 * xy_dot
                dist_sq_flat = dist_sq_flat.float().clamp(min=0)  # Ensure real float and non-negative

                # Apply radius mask
                radius_mask = (dist_sq_flat < self.repel_radius_sq).float()
                radius_mask.fill_diagonal_(0)

                if self.repel_potential_type == "lennard_jones_stub":
                    logger.warning("Lennard-Jones potential stub.")
                    potential = self.repel_token_w * radius_mask / (dist_sq_flat + self.repel_eps)  # Fallback
                else:  # 'smoothed_inverse_sq'
                    potential = self.repel_token_w * radius_mask / (dist_sq_flat + self.repel_eps)

                # Sum upper triangle only, average per token pair approx (N*(N-1)/2 pairs)
                num_pairs = (B * T) * (B * T - 1) / 2.0
                energy_repel = torch.triu(potential, diagonal=1).sum() / max(1.0, num_pairs)  # Average over potential pairs
                total_energy = total_energy + energy_repel
                repulsion_energy_val = energy_repel.item()
                logger.debug(f"FormalForceV2: Avg Token Repulsion Energy = {repulsion_energy_val:.4g}")
            except Exception as e:
                logger.error(f"Token Repulsion energy calculation failed: {e}", exc_info=True)
            energy_terms["E_repel_token"] = repulsion_energy_val

        # 4. Alignment Energy (Placeholder - Requires velocity)
        if self.align_w > 0:
            logger.warning("FormalForceV2: Alignment energy term requires velocity state, not implemented.")
            energy_terms["E_align"] = 0.0

        logger.debug(f"FormalForceV2 - Total Energy Terms: {energy_terms}, Total = {total_energy.item():.4g}")

        # --- Calculate Gradient ---
        if total_energy == 0.0 or not x_h_input.requires_grad:
            logger.debug("FormalForceV2: Total energy is zero or input requires no grad. Force is zero.")
            return torch.zeros_like(x_h)

        start_grad = time.time()
        force = torch.zeros_like(x_h)  # Default zero force
        try:
            grad_outputs = torch.ones_like(total_energy)
            gradients = torch.autograd.grad(
                outputs=total_energy,
                inputs=x_h_input,
                grad_outputs=grad_outputs,
                retain_graph=False,
                create_graph=False,  # No need for higher order gradients typically
                allow_unused=True,  # Allow parts of graph to be unused
            )[0]
            grad_time = time.time() - start_grad
            logger.debug(f"FormalForceV2: Gradient calculation took {grad_time:.4f} sec.")

            if gradients is not None:
                force = -gradients  # Force is negative gradient
                force_norm_avg = torch.linalg.vector_norm(force, dim=-1).mean().item()
                logger.debug(f"FormalForceV2: Calculated force avg norm = {force_norm_avg:.4g}")
                # Clamp force magnitude? Optional safety measure
                # max_force_norm = 10.0
                # current_norm = torch.linalg.vector_norm(force, dim=-1, keepdim=True)
                # force = force * torch.clamp(max_force_norm / current_norm.clamp(min=1e-6), max=1.0)

            else:
                logger.warning("FormalForceV2: Gradient computation returned None.")

        except Exception as e:
            logger.error(f"FormalForceV2: Gradient calculation failed: {e}", exc_info=True)

        # Detach the final force tensor
        return force.detach()

    def extra_repr(self) -> str:
        return f"attr_c={self.attr_centroid_w}, attr_g={self.attr_global_w}, repel_t={self.repel_token_w}"
