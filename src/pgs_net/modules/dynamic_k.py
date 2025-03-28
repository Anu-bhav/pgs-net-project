# src/pgs_net/modules/dynamic_k.py
"""Dynamic K Controller Implementations"""

import logging
import math  # For isnan check
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .interfaces import DynamicKController

logger = logging.getLogger(__name__)


# --- Placeholder Controller ---
class PlaceholderDynamicKController(DynamicKController):
    """Placeholder: Always uses max_k clusters."""

    def __init__(self, max_k: int, params: Dict, state_mode: str, device: torch.device, dtype: torch.dtype):
        """
        Initializes the PlaceholderDynamicKController.

        Args:
            max_k (int): The maximum number of clusters.
            params (Dict): Configuration parameters (unused here).
            state_mode (str): State management mode (unused here).
            device (torch.device): Device for tensors.
            dtype (torch.dtype): Data type for tensors (unused here).

        """
        super().__init__()
        logger.warning("Using PlaceholderDynamicKController (Always uses max_k).")
        self.max_k = max_k
        self.device = device
        self._update_cache()  # Initialize cache

    @torch.no_grad()
    def update_k(
        self,
        current_assignments: Optional[torch.Tensor] = None,
        centroids: Optional[torch.Tensor] = None,
        x_h: Optional[torch.Tensor] = None,
    ) -> None:
        """No-op update for the placeholder."""
        pass  # No state changes

    def get_active_indices(self) -> torch.Tensor:
        """Returns indices from 0 to max_k - 1."""
        return self.active_indices_cache

    @torch.no_grad()
    def _update_cache(self) -> None:
        """Updates the cached list of active indices."""
        self.active_indices_cache = torch.arange(self.max_k, device=self.device)

    def extra_repr(self) -> str:
        return f"max_k={self.max_k}"


# --- Usage-Based Pruning/Reactivation Controller ---
class UsageBasedDynamicKController(DynamicKController):
    """Dynamically prunes/reactivates clusters based on EMA usage."""

    def __init__(self, max_k: int, params: Dict, state_mode: str, device: torch.device, dtype: torch.dtype):
        """
        Initializes the UsageBasedDynamicKController.

        Args:
            max_k (int): The maximum number of physical clusters.
            params (Dict): Configuration parameters from 'dynamic_k_params'. Expects keys like
                           'min_k', 'update_interval', 'usage_ema_decay', 'prune_threshold_factor',
                           'reactivation_threshold_factor'.
            state_mode (str): State management mode ('buffers' assumed for internal state).
            device (torch.device): Device for tensors.
            dtype (torch.dtype): Data type for tensors (used for buffer initialization).

        """
        super().__init__()
        logger.info("Initializing UsageBasedDynamicKController (Prune/Reactivate).")
        self.max_k_physical = max_k
        self.config_params = params
        self.min_k = params.get("min_k", 2)
        self.update_interval = params.get("update_interval", 10)
        self.usage_ema_decay = params.get("usage_ema_decay", 0.99)
        # Threshold factors relative to uniform usage (1 / num_active)
        self.prune_threshold_factor = params.get("prune_threshold_factor", 0.1)
        self.reactivation_threshold_factor = params.get("reactivation_threshold_factor", 0.5)
        self.state_mode = state_mode  # Currently unused, state managed with buffers
        self.device = device
        self.dtype = dtype
        self.steps_since_update = 0

        # State Buffers
        self.register_buffer("cluster_usage_ema", torch.ones(max_k, device=device, dtype=torch.float32) / max_k)
        self.register_buffer("active_mask", torch.zeros(max_k, dtype=torch.bool, device=device))
        # Ensure min_k is not greater than max_k
        self.min_k = min(self.min_k, self.max_k_physical)
        if self.min_k > 0:
            self.active_mask[: self.min_k] = True  # Start with min_k active
        else:  # Handle min_k=0 case if needed (start with 1 active?)
            logger.warning("min_k <= 0 specified for DynamicK, starting with 1 active cluster.")
            self.min_k = 1
            self.active_mask[0] = True

        self.active_indices_cache: Optional[torch.Tensor] = None
        self._update_cache()

    @torch.no_grad()
    def update_k(
        self,
        current_assignments: Optional[torch.Tensor],
        centroids: Optional[torch.Tensor] = None,
        x_h: Optional[torch.Tensor] = None,
    ) -> None:
        """Updates EMA usage and prunes/reactivates clusters."""
        self.steps_since_update += 1
        if self.steps_since_update < self.update_interval:
            return
        if current_assignments is None:
            logger.warning("DynK(UsageBased) update skipped: Missing current_assignments.")
            return

        logger.debug("DynamicK (UsageBased) Update Step...")
        self.steps_since_update = 0
        K_max_phys = current_assignments.shape[-1]
        if K_max_phys != self.max_k_physical:
            logger.error(f"Mismatch K_max in assignments ({K_max_phys}) vs controller ({self.max_k_physical}). Skipping update.")
            return

        # --- Update Usage EMA ---
        current_usage = current_assignments.sum(dim=(0, 1)).clamp(min=0).float()  # Ensure float, non-negative
        total_weight = current_usage.sum().clamp(min=1e-6)
        current_usage_norm = current_usage / total_weight
        self.cluster_usage_ema.mul_(self.usage_ema_decay).add_(current_usage_norm, alpha=1.0 - self.usage_ema_decay)
        # Clamp EMA value to avoid potential floating point issues leading to negative values
        self.cluster_usage_ema.clamp_(min=0.0)

        active_idxs_now = self.get_active_indices()
        if len(active_idxs_now) > 0:
            logger.debug(f"DynK - EMA Usage (Active): {self.cluster_usage_ema[active_idxs_now].cpu().numpy()}")
        else:
            logger.warning("DynK - No active clusters during update!")

        # --- Calculate Thresholds ---
        num_active_current = self.active_mask.sum().item()
        if num_active_current == 0:
            return  # Avoid division by zero
        target_uniform_usage = 1.0 / num_active_current
        prune_threshold = self.prune_threshold_factor * target_uniform_usage
        reactivate_threshold = self.reactivation_threshold_factor * target_uniform_usage

        # --- Reactivation ---
        inactive_indices = (~self.active_mask).nonzero(as_tuple=True)[0]
        if len(inactive_indices) > 0:
            usage_of_inactive = self.cluster_usage_ema[inactive_indices]
            can_reactivate_mask = usage_of_inactive > reactivate_threshold
            if can_reactivate_mask.any():
                # Reactivate the inactive cluster with the highest usage above threshold
                eligible_indices = inactive_indices[can_reactivate_mask]
                eligible_usages = usage_of_inactive[can_reactivate_mask]
                reactivate_idx_local = torch.argmax(eligible_usages)
                reactivate_idx_global = eligible_indices[reactivate_idx_local]
                self.active_mask[reactivate_idx_global] = True
                logger.info(
                    f"DynK - Reactivated cluster {reactivate_idx_global.item()} (Usage {self.cluster_usage_ema[reactivate_idx_global].item():.4f} > {reactivate_threshold:.4f}). New active: {self.active_mask.sum().item()}"
                )
                num_active_current += 1  # Update count for pruning check

        # --- Pruning ---
        if num_active_current > self.min_k:
            usage_of_active = self.cluster_usage_ema[self.active_mask]
            can_prune_mask = usage_of_active < prune_threshold
            active_indices_local = self.active_mask.nonzero(as_tuple=True)[0]  # Get current active indices again

            if can_prune_mask.any():
                # Prune the active cluster with the lowest usage among candidates
                prune_candidate_indices_local = can_prune_mask.nonzero(as_tuple=True)[
                    0
                ]  # Indices relative to active_indices_local
                prune_candidate_usages = usage_of_active[can_prune_mask]
                prune_idx_relative = torch.argmin(prune_candidate_usages)  # Index within candidates
                prune_idx_local = prune_candidate_indices_local[prune_idx_relative]  # Index relative to active_indices_local
                prune_idx_global = active_indices_local[prune_idx_local]  # Map back to global index

                if num_active_current > self.min_k:  # Double check count after potential reactivation
                    self.active_mask[prune_idx_global] = False
                    # Reset usage for pruned cluster to avoid immediate reactivation
                    self.cluster_usage_ema[prune_idx_global] = 0.0  # Reset usage to 0
                    logger.info(
                        f"DynK - Pruned cluster {prune_idx_global.item()} (Usage {prune_candidate_usages[prune_idx_relative].item():.4f} < {prune_threshold:.4f}). New active: {self.active_mask.sum().item()}"
                    )
                    num_active_current -= 1

        # --- Update Cache ---
        self._update_cache()

    def get_active_indices(self) -> torch.Tensor:
        # Ensure cache is updated if called before update_k initializes it properly
        if self.active_indices_cache is None:
            self._update_cache()
        return self.active_indices_cache

    @torch.no_grad()
    def _update_cache(self) -> None:
        """Updates the cached list of active indices based on the mask."""
        self.active_indices_cache = self.active_mask.nonzero(as_tuple=True)[0]

    def extra_repr(self) -> str:
        return f"max_k={self.max_k_physical}, min_k={self.min_k}, update_interval={self.update_interval}"


class SplitMergeDynamicKController(DynamicKController):
    """Dynamically splits high-variance and merges close clusters. Includes Usage Pruning/Reactivation."""

    def __init__(self, max_k: int, params: Dict, state_mode: str, device: torch.device, dtype: torch.dtype):
        """
        Initializes the SplitMergeDynamicKController.

        Args:
            max_k (int): Max physical centroids.
            params (Dict): Config params ('min_k', 'update_interval', 'usage_ema_decay',
                           'prune_threshold_factor', 'reactivation_threshold_factor',
                           'split_variance_threshold_factor', 'split_method', 'split_init_factor',
                           'merge_distance_threshold_factor', 'merge_method').
            state_mode (str): State management mode.
            device, dtype: Tensor device and data type.

        """
        super().__init__()
        logger.info("Initializing SplitMergeDynamicKController.")
        self.max_k_physical = max_k
        self.config_params = params
        self.min_k = params.get("min_k", 2)
        self.update_interval = params.get("update_interval", 10)
        self.usage_ema_decay = params.get("usage_ema_decay", 0.98)
        self.prune_threshold_factor = params.get("prune_threshold_factor", 0.1)
        self.reactivation_threshold_factor = params.get("reactivation_threshold_factor", 0.5)
        self.split_variance_threshold_factor = params.get("split_variance_threshold_factor", 2.0)
        self.split_method = params.get("split_method", "perturb")
        self.split_init_factor = params.get("split_init_factor", 0.05)
        self.merge_distance_threshold_factor = params.get("merge_distance_threshold_factor", 0.15)
        self.merge_method = params.get("merge_method", "average")
        self.state_mode = state_mode  # Note: This controller modifies centroids directly, assumes 'buffers' or direct access.
        self.device = device
        self.dtype = dtype
        self.is_complex = dtype.is_complex
        self.steps_since_update = 0

        # State Buffers
        self.register_buffer("cluster_usage_ema", torch.ones(max_k, device=device, dtype=torch.float32) / max_k)
        self.register_buffer(
            "cluster_variance_ema", torch.ones(max_k, device=device, dtype=torch.float32)
        )  # Variance of L2 norms squared
        self.register_buffer("active_mask", torch.zeros(max_k, dtype=torch.bool, device=device))
        # Ensure min_k is valid
        self.min_k = min(self.min_k, self.max_k_physical)
        if self.min_k > 0:
            self.active_mask[: self.min_k] = True
        else:
            self.min_k = 1
            self.active_mask[0] = True
            logger.warning("min_k <= 0, starting with 1 active.")

        self.active_indices_cache: Optional[torch.Tensor] = None
        self._update_cache()

    @torch.no_grad()
    def update_k(
        self, current_assignments: Optional[torch.Tensor], centroids: Optional[torch.Tensor], x_h: Optional[torch.Tensor]
    ) -> None:
        """Performs prune, reactivate, split, merge operations based on EMA stats."""
        self.steps_since_update += 1
        if self.steps_since_update < self.update_interval:
            return
        if current_assignments is None or centroids is None or x_h is None:
            logger.warning("DynK(SplitMerge) update skipped: Missing inputs.")
            return

        # Ensure centroids are modifiable if needed (remove from graph?)
        # We modify centroids.data directly to avoid graph issues if centroids are parameters
        if not isinstance(centroids, nn.Parameter):
            logger.warning("DynK(SplitMerge) received non-Parameter centroids. Modifications may not persist.")

        logger.debug("DynamicK (Split/Merge) Update Step...")
        self.steps_since_update = 0
        B, T, K_max_phys = current_assignments.shape
        K_current_max, D = centroids.shape
        if K_max_phys != self.max_k_physical or K_current_max != self.max_k_physical:
            logger.error("Mismatch K_max in inputs/controller state. Skipping update.")
            return

        # --- 1. Update EMA Stats ---
        current_usage = current_assignments.sum(dim=(0, 1)).clamp(min=0).float()
        total_weight = current_usage.sum().clamp(min=1e-6)
        current_usage_norm = current_usage / total_weight
        self.cluster_usage_ema.mul_(self.usage_ema_decay).add_(current_usage_norm, alpha=1.0 - self.usage_ema_decay).clamp_(min=0)

        # Variance EMA (based on L2 norm squared)
        token_norms_sq = x_h.abs().pow(2).sum(-1) if self.is_complex else x_h.pow(2).sum(-1)  # (B, T) Real
        A = current_assignments.float()  # Ensure float for calculations
        cluster_weights = A.sum(dim=(0, 1)).clamp(min=1e-8)  # (K,)
        e_norm_sq = (A * token_norms_sq.unsqueeze(-1)).sum(dim=(0, 1)) / cluster_weights  # E[Norm^2]
        e_norm4 = (A * token_norms_sq.pow(2).unsqueeze(-1)).sum(dim=(0, 1)) / cluster_weights  # E[Norm^4]
        current_variance = (e_norm4 - e_norm_sq.pow(2)).clamp(min=0)  # Variance of norm^2
        active_idxs_now = self.active_mask.nonzero(as_tuple=True)[0]  # Use current mask before potential changes
        if len(active_idxs_now) > 0:
            self.cluster_variance_ema[active_idxs_now].mul_(self.usage_ema_decay).add_(
                current_variance[active_idxs_now], alpha=1.0 - self.usage_ema_decay
            ).clamp_(min=0)
            # logger.debug(f"DynK - EMA Usage (Active): {self.cluster_usage_ema[active_idxs_now].cpu().numpy()}")
            # logger.debug(f"DynK - EMA Variance (Active): {self.cluster_variance_ema[active_idxs_now].cpu().numpy()}")

        # --- Calculate Thresholds ---
        num_active_current = self.active_mask.sum().item()
        if num_active_current == 0:
            logger.warning("DynK - No active clusters!")
            return
        target_uniform_usage = 1.0 / num_active_current
        prune_threshold = self.config_params["prune_threshold_factor"] * target_uniform_usage
        reactivate_threshold = self.reactivation_threshold_factor * target_uniform_usage
        avg_active_variance = self.cluster_variance_ema[self.active_mask].mean().item() if num_active_current > 0 else 1.0
        split_threshold = self.config_params["split_variance_threshold_factor"] * avg_active_variance

        avg_dist = 0.0
        merge_threshold_sq = float("inf")
        if num_active_current > 1:
            active_centroids = centroids[self.active_mask]
            diffs = active_centroids.unsqueeze(1) - active_centroids.unsqueeze(0)
            dist_sq = diffs.abs().pow(2).sum(dim=-1)
            dist_sq.diagonal().fill_(float("inf"))
            finite_dists_sqrt = dist_sq[dist_sq != float("inf")].sqrt()
            if finite_dists_sqrt.numel() > 0:
                avg_dist = finite_dists_sqrt.mean().item()
            if avg_dist > 1e-6:
                merge_threshold_sq = (self.config_params["merge_distance_threshold_factor"] * avg_dist) ** 2

        # --- 2. Reactivation ---
        inactive_indices = (~self.active_mask).nonzero(as_tuple=True)[0]
        if len(inactive_indices) > 0:
            usage_of_inactive = self.cluster_usage_ema[inactive_indices]
            can_reactivate_mask = usage_of_inactive > reactivate_threshold
            if can_reactivate_mask.any():
                eligible_indices = inactive_indices[can_reactivate_mask]
                eligible_usages = usage_of_inactive[can_reactivate_mask]
                reactivate_idx_global = eligible_indices[torch.argmax(eligible_usages)]
                self.active_mask[reactivate_idx_global] = True
                logger.info(f"DynK - Reactivated cluster {reactivate_idx_global.item()} (Usage > {reactivate_threshold:.4f}).")
                num_active_current += 1

        # --- 3. Pruning ---
        if num_active_current > self.min_k:
            usage_of_active = self.cluster_usage_ema[self.active_mask]
            can_prune_mask = usage_of_active < prune_threshold
            active_indices_local = self.active_mask.nonzero(as_tuple=True)[0]  # Get current active indices
            if can_prune_mask.any():
                prune_candidate_indices_global = active_indices_local[can_prune_mask]  # Global indices to consider
                prune_candidate_usages = usage_of_active[can_prune_mask]
                prune_idx_global = prune_candidate_indices_global[torch.argmin(prune_candidate_usages)]
                if num_active_current > self.min_k:  # Double check count
                    self.active_mask[prune_idx_global] = False
                    self.cluster_usage_ema[prune_idx_global] = 0.0  # Reset usage
                    self.cluster_variance_ema[prune_idx_global] = (
                        avg_active_variance if avg_active_variance > 0 else 1.0
                    )  # Reset variance
                    logger.info(f"DynK - Pruned cluster {prune_idx_global.item()} (Usage < {prune_threshold:.4f}).")
                    num_active_current -= 1

        # --- 4. Splitting ---
        if num_active_current < self.max_k_physical:
            high_variance_candidates = (self.cluster_variance_ema > split_threshold) & self.active_mask
            if high_variance_candidates.any():
                split_candidate_indices = high_variance_candidates.nonzero(as_tuple=True)[0]
                split_idx_global = split_candidate_indices[torch.argmax(self.cluster_variance_ema[split_candidate_indices])]
                inactive_indices = (~self.active_mask).nonzero(as_tuple=True)[0]
                if len(inactive_indices) > 0:
                    new_idx = inactive_indices[0]
                    logger.info(
                        f"DynK - Splitting cluster {split_idx_global.item()} (Var {self.cluster_variance_ema[split_idx_global].item():.3f} > {split_threshold:.3f}) -> new slot {new_idx.item()}."
                    )
                    # Modify centroids.data directly
                    if self.split_method == "pca_stub":
                        logger.warning("PCA split method stub.")
                    perturbation = torch.randn_like(centroids[split_idx_global]) * self.config_params["split_init_factor"]
                    centroids.data[new_idx] = centroids.data[split_idx_global] + perturbation
                    centroids.data[split_idx_global] -= perturbation
                    # Update state
                    self.active_mask[new_idx] = True
                    split_usage = self.cluster_usage_ema[split_idx_global] / 2.0
                    self.cluster_usage_ema[new_idx] = split_usage
                    self.cluster_usage_ema[split_idx_global] = split_usage
                    self.cluster_variance_ema[new_idx] = avg_active_variance if avg_active_variance > 0 else 1.0
                    self.cluster_variance_ema[split_idx_global] = avg_active_variance if avg_active_variance > 0 else 1.0
                    num_active_current += 1
                else:
                    logger.debug("DynK - Split candidate found, but no inactive slots.")

        # --- 5. Merging ---
        if num_active_current > self.min_k and avg_dist > 1e-6:
            active_idxs_now = self.active_mask.nonzero(as_tuple=True)[0]  # Recompute active indices
            if len(active_idxs_now) > 1:
                active_centroids = centroids[active_idxs_now]
                diffs = active_centroids.unsqueeze(1) - active_centroids.unsqueeze(0)
                dist_sq = diffs.abs().pow(2).sum(dim=-1)
                dist_sq.diagonal().fill_(float("inf"))
                min_dist_sq_val, min_idx_flat = torch.min(dist_sq.view(-1), dim=0)
                if min_dist_sq_val < merge_threshold_sq:
                    row_idx = min_idx_flat // len(active_idxs_now)
                    col_idx = min_idx_flat % len(active_idxs_now)
                    global_idx1 = active_idxs_now[row_idx]
                    global_idx2 = active_idxs_now[col_idx]
                    if global_idx1 != global_idx2 and num_active_current > self.min_k:
                        # Merge idx2 into idx1 (keep idx1 active)
                        logger.info(
                            f"DynK - Merging cluster {global_idx2.item()} into {global_idx1.item()} (Dist^2 {min_dist_sq_val.item():.3f} < {merge_threshold_sq:.3f})."
                        )
                        usage1 = self.cluster_usage_ema[global_idx1]
                        usage2 = self.cluster_usage_ema[global_idx2]
                        total_usage = usage1 + usage2
                        w1 = usage1 / total_usage.clamp(1e-8)
                        w2 = usage2 / total_usage.clamp(1e-8)
                        self.cluster_variance_ema[global_idx1] = (
                            w1 * self.cluster_variance_ema[global_idx1] + w2 * self.cluster_variance_ema[global_idx2]
                        )
                        self.cluster_usage_ema[global_idx1] = total_usage
                        # Merge Centroids
                        if self.merge_method == "average":
                            centroids.data[global_idx1] = (centroids.data[global_idx1] + centroids.data[global_idx2]) / 2.0
                        elif self.merge_method == "keep_dominant":
                            if usage2 > usage1:
                                centroids.data[global_idx1] = centroids.data[global_idx2]
                        # Deactivate removed cluster
                        self.active_mask[global_idx2] = False
                        self.cluster_usage_ema[global_idx2] = 0.0
                        self.cluster_variance_ema[global_idx2] = 0.0
                        num_active_current -= 1

        # --- Final Update Cache ---
        self._update_cache()

    def get_active_indices(self) -> torch.Tensor:
        if self.active_indices_cache is None:
            self._update_cache()  # Ensure cache exists
        return self.active_indices_cache

    @torch.no_grad()
    def _update_cache(self) -> None:
        """Updates the cached list of active indices based on the mask."""
        self.active_indices_cache = self.active_mask.nonzero(as_tuple=True)[0]

    def extra_repr(self) -> str:
        return f"max_k={self.max_k_physical}, min_k={self.min_k}, update_interval={self.update_interval}, split={self.split_method}, merge={self.merge_method}"
