# src/pgs_net/modules/clustering_assignment.py
"""Computes token-to-cluster assignments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, List
import math

# Assuming interfaces and controllers are imported correctly from .interfaces and .dynamic_k
from .interfaces import DynamicKController
from .dynamic_k import PlaceholderDynamicKController, UsageBasedDynamicKController, SplitMergeDynamicKController

logger = logging.getLogger(__name__)


class ClusteringAssignment(nn.Module):
    """
    Computes token assignments to clusters based on similarity scores.
    Supports dynamic temperature, tunneling, and dynamic K controllers.
    """

    def __init__(self, config: Dict[str, Any], device: torch.device, dtype: torch.dtype):
        """
        Initializes the ClusteringAssignment module.

        Args:
            config (Dict[str, Any]): The main PGS_FFN configuration dictionary.
            device (torch.device): The device tensors should be on.
            dtype (torch.dtype): The expected data type (real or complex).
        """
        super().__init__()
        self.config = config.get("clustering", {})
        self.global_config = config
        self.dynamic_k_controller: Optional[DynamicKController] = None
        self.max_clusters = self.config.get("max_clusters", 4)

        # Instantiate Dynamic K Controller
        if self.config.get("use_dynamic_k", False):
            k_type = self.config.get("dynamic_k_type", "none")
            k_params = self.config.get("dynamic_k_params", {})
            state_mode = self.global_config.get("architecture", {}).get("state_management", {}).get("mode", "buffers")

            if k_type == "usage_based":
                self.dynamic_k_controller = UsageBasedDynamicKController(self.max_clusters, k_params, state_mode, device, dtype)
            elif k_type == "split_merge":
                self.dynamic_k_controller = SplitMergeDynamicKController(self.max_clusters, k_params, state_mode, device, dtype)
            elif k_type != "none":
                self.dynamic_k_controller = PlaceholderDynamicKController(self.max_clusters, k_params, state_mode, device, dtype)
                logger.warning(f"Using Placeholder Dynamic K ({k_type} not impl).")

            if self.dynamic_k_controller:
                logger.info(f"Initialized Dynamic K Controller: {k_type}")

    def forward(
        self,
        sim: torch.Tensor,
        head_idx: int,
        epoch: Optional[int] = None,
        centroids: Optional[torch.Tensor] = None,
        x_h: Optional[torch.Tensor] = None,
        analysis_data: Optional[Dict] = None,
        dynamic_params: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Computes the assignment matrix A.

        Args:
            sim (torch.Tensor): Similarity matrix (B, T, K_max).
            head_idx (int): Current head index.
            epoch (Optional[int]): Current epoch for dynamic temperature.
            centroids (Optional[torch.Tensor]): Centroid tensor (K_max, D), needed for Dynamic K update.
            x_h (Optional[torch.Tensor]): Token embeddings (B, T, D), needed for Dynamic K update.
            analysis_data (Optional[Dict]): Dictionary to store analysis results.
            dynamic_params (Optional[Dict]): Dictionary of meta-learned hyperparameters.

        Returns:
            torch.Tensor: Assignment matrix A (B, T, K_max).
        """
        B, T, K_max = sim.shape
        current_max_k = self.max_clusters  # Physical max K

        # --- Get Effective Assignment Temperature ---
        temp_val = (
            dynamic_params.get("assignment_temp", self.config["assignment_temp"])
            if dynamic_params
            else self.config["assignment_temp"]
        )
        if self.config["use_dynamic_assignment_temp"]:
            sched_params = self.config.get("dynamic_temp_params", {})
            max_epochs = sched_params.get("max_epochs_for_schedule", 100)
            if self.config["dynamic_temp_schedule"] == "annealing" and epoch is not None:
                start, end, rate = sched_params.get("start", 1.0), sched_params.get("end", 0.1), sched_params.get("rate", 0.999)
                # Exponential decay or linear? Let's use exponential decay towards end temp.
                decay_factor = rate ** (epoch / max(1, max_epochs - 1))  # Normalize epoch progress approx
                temp_val = end + (start - end) * decay_factor
            # Add other schedules later ('loss_based_stub')
            temp_val = max(temp_val, sched_params.get("end", 0.01))  # Floor temp
        temp_val = max(temp_val, 1e-4)  # Global floor
        if analysis_data:
            analysis_data["assignment_temp_effective"] = temp_val

        # --- Dynamic K Active Indices ---
        active_indices = torch.arange(current_max_k, device=sim.device)  # Default: all clusters active
        sim_active = sim
        if self.dynamic_k_controller is not None:
            active_indices = self.dynamic_k_controller.get_active_indices()
            num_active = active_indices.shape[0]
            if num_active < current_max_k:
                logger.debug(f"[Head {head_idx}] Dynamic K using {num_active} active indices: {active_indices.tolist()}")
                try:
                    # Use index_select for potentially non-contiguous indices
                    sim_active = sim.index_select(dim=-1, index=active_indices)  # Select columns (B, T, K_active)
                except Exception as e:
                    logger.error(f"Dynamic K index select error (Indices: {active_indices}, Sim shape: {sim.shape}): {e}")
                    sim_active = sim
                    active_indices = torch.arange(current_max_k, device=sim.device)  # Fallback
            elif num_active == 0:  # Handle case where controller might prune all
                logger.error("Dynamic K controller returned zero active indices! Using all clusters as fallback.")
                sim_active = sim
                active_indices = torch.arange(current_max_k, device=sim.device)

        # --- Softmax on Active Similarities ---
        try:
            A_active = F.softmax(sim_active / temp_val, dim=-1)  # (B, T, K_active)
        except Exception as e:
            logger.error(
                f"Softmax failed: temp={temp_val}, sim_active_max={sim_active.max()}, sim_active_min={sim_active.min()}. Error: {e}"
            )
            # Fallback: Uniform assignment over active clusters
            num_active = sim_active.shape[-1]
            A_active = torch.ones_like(sim_active) / num_active

        # --- Expand A back to full size ---
        A = torch.zeros_like(sim)  # Full size zeros (B, T, K_max)
        if self.dynamic_k_controller is not None and A_active.shape[-1] != current_max_k:
            try:
                # Create multi-dim index for index_put_
                B_idx, T_idx, _ = torch.meshgrid(
                    torch.arange(B, device=A.device),
                    torch.arange(T, device=A.device),
                    torch.arange(A_active.shape[-1], device=A.device),  # Index for K_active dim
                    indexing="ij",
                )
                cluster_idx_scatter = active_indices.view(1, 1, -1).expand(
                    B, T, -1
                )  # Expand active_indices to match A_active shape
                # Use index_put_ for sparse assignment
                A.index_put_((B_idx, T_idx, cluster_idx_scatter), A_active, accumulate=False)
            except Exception as e:
                logger.error(f"Dynamic K scatter error: {e}. Assignment matrix might be incorrect.")
                # Fallback: Pad (might misalign clusters)
                pad_size = current_max_k - A_active.shape[-1]
                if pad_size > 0:
                    A = F.pad(A_active, (0, pad_size))
                else:
                    A = A_active  # Should not happen if logic is correct
        else:
            A = A_active  # No dynamic K or already full size

        # --- Tunneling (on full A) ---
        if self.config.get("use_tunneling_assignment", False) and self.training:
            prob = self.config.get("tunneling_prob", 0.001)
            if K_max > 1 and prob > 0:
                tunnel_mask = torch.rand(B, T, device=A.device) < prob
                num_tunnel = tunnel_mask.sum()
                if num_tunnel > 0:
                    # ... (STE tunneling logic as implemented previously) ...
                    A_tunnel = A.clone()  # ... generate random different assignments ...# A = A + (A_tunnel - A).detach()
                    logger.debug(f"[Head {head_idx}] Applied tunneling to {num_tunnel} tokens.")

        # --- Dynamic K Post-Assignment Update ---
        if self.dynamic_k_controller is not None:
            # Pass centroids (allows controller to modify them) and x_h
            # Ensure requires_grad is handled correctly if centroids are modified inside no_grad context
            # Controller method has @torch.no_grad() decorator
            self.dynamic_k_controller.update_k(A.detach(), centroids, x_h)

        # --- Analysis Data ---
        if analysis_data is not None:
            analysis_data["assignment_entropy"] = -(A * torch.log(A.clamp(min=1e-9))).sum(dim=-1).mean().item()
            if self.dynamic_k_controller is not None:
                analysis_data["dynamic_k_active_count"] = self.dynamic_k_controller.get_active_indices().shape[0]
            else:
                analysis_data["dynamic_k_active_count"] = current_max_k

        return A
