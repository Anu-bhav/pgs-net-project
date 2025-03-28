# src/pgs_net/modules/dynamic_k.py
""" Dynamic K Controller Implementations """
import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional, Tuple, List
from .interfaces import DynamicKController

logger = logging.getLogger(__name__)

# --- Placeholder ---
class PlaceholderDynamicKController(DynamicKController):
    # ... (Implementation as before) ...

# --- Usage Based Pruning Only ---
class UsageBasedDynamicKController(DynamicKController):
    """ Dynamically prunes/reactivates clusters based on EMA usage. """
    def __init__(self, max_k: int, params: Dict, state_mode: str, device: torch.device, dtype: torch.dtype):
        super().__init__()
        logger.info("Initializing UsageBasedDynamicKController (Prune/Reactivate).")
        self.max_k_physical = max_k
        self.config_params = params
        self.min_k = params.get('min_k', 2)
        self.update_interval = params.get('update_interval', 10)
        self.usage_ema_decay = params.get('usage_ema_decay', 0.99)
        self.prune_threshold_factor = params.get('prune_threshold_factor', 0.1) # Prune if usage < factor * (1/CurrentK)
        self.reactivation_threshold_factor = params.get('reactivation_threshold_factor', 0.5) # Reactivate if usage > factor * (1/CurrentK)
        self.state_mode = state_mode # Currently unused, state managed with buffers
        self.device = device; self.dtype = dtype
        self.steps_since_update = 0

        # State Buffers
        self.register_buffer('cluster_usage_ema', torch.ones(max_k, device=device, dtype=torch.float32) / max_k)
        self.register_buffer('active_mask', torch.zeros(max_k, dtype=torch.bool, device=device))
        self.active_mask[:self.min_k] = True # Start with min_k active
        self._update_cache()

    @torch.no_grad()
    def update_k(self, current_assignments: Optional[torch.Tensor], centroids: Optional[torch.Tensor]=None, x_h: Optional[torch.Tensor]=None) -> None:
        self.steps_since_update += 1
        if self.steps_since_update < self.update_interval: return
        if current_assignments is None: return

        logger.debug("DynamicK (UsageBased) Update Step...")
        self.steps_since_update = 0
        K_max_phys = current_assignments.shape[-1]

        # --- Update Usage EMA ---
        current_usage = current_assignments.sum(dim=(0, 1)).clamp(min=0) # Ensure non-negative
        total_weight = current_usage.sum().clamp(min=1e-6)
        current_usage = current_usage / total_weight
        self.cluster_usage_ema.mul_(self.usage_ema_decay).add_(current_usage, alpha=1.0 - self.usage_ema_decay)
        active_idxs_now = self.get_active_indices()
        logger.debug(f"DynK - EMA Usage (Active): {self.cluster_usage_ema[active_idxs_now].cpu().numpy()}")

        # --- Calculate Thresholds ---
        num_active_before = self.active_mask.sum().item()
        if num_active_before == 0: return
        target_uniform_usage = 1.0 / num_active_before
        prune_threshold = self.prune_threshold_factor * target_uniform_usage
        reactivate_threshold = self.reactivation_threshold_factor * target_uniform_usage

        # --- Reactivation ---
        inactive_indices = (~self.active_mask).nonzero(as_tuple=True)[0]
        if len(inactive_indices) > 0:
            can_reactivate = (self.cluster_usage_ema[inactive_indices] > reactivate_threshold)
            if can_reactivate.any():
                reactivate_idx_local = torch.argmax(self.cluster_usage_ema[inactive_indices[can_reactivate]])
                reactivate_idx_global = inactive_indices[can_reactivate][reactivate_idx_local]
                self.active_mask[reactivate_idx_global] = True
                logger.info(f"DynK - Reactivated cluster {reactivate_idx_global} (Usage > {reactivate_threshold:.4f}). New active: {self.active_mask.sum()}")
                num_active_before += 1

        # --- Pruning ---
        if num_active_before > self.min_k:
            can_prune = (self.cluster_usage_ema < prune_threshold) & self.active_mask
            if can_prune.any():
                prune_candidate_indices = can_prune.nonzero(as_tuple=True)[0]
                prune_candidate_values = self.cluster_usage_ema[prune_candidate_indices]
                prune_idx_global = prune_candidate_indices[torch.argmin(prune_candidate_values)]
                if num_active_before > self.min_k: # Double check
                    self.active_mask[prune_idx_global] = False
                    self.cluster_usage_ema[prune_idx_global] = 1.0 / K_max_phys # Reset usage
                    logger.info(f"DynK - Pruned cluster {prune_idx_global} (Usage < {prune_threshold:.4f}). New active: {self.active_mask.sum()}")
                    num_active_before -= 1

        # --- Update Cache ---
        self._update_cache()

    def get_active_indices(self) -> torch.Tensor:
        return self.active_indices_cache

    def _update_cache(self) -> None:
        self.active_indices_cache = self.active_mask.nonzero(as_tuple=True)[0]


class SplitMergeDynamicKController(DynamicKController):
    """ Dynamically splits high-variance and merges close clusters. Includes Usage Pruning/Reactivation. """
    def __init__(self, max_k: int, params: Dict, state_mode: str, device: torch.device, dtype: torch.dtype):
        super().__init__()
        logger.info("Initializing SplitMergeDynamicKController.")
        self.max_k_physical = max_k
        self.config_params = params
        self.min_k = params.get('min_k', 2)
        self.update_interval = params.get('update_interval', 10)
        self.usage_ema_decay = params.get('usage_ema_decay', 0.98)
        self.prune_threshold_factor = params.get('prune_threshold_factor', 0.1)
        self.reactivation_threshold_factor = params.get('reactivation_threshold_factor', 0.5)
        self.split_variance_threshold_factor = params.get('split_variance_threshold_factor', 2.0)
        self.split_method = params.get('split_method', 'perturb')
        self.split_init_factor = params.get('split_init_factor', 0.05)
        self.merge_distance_threshold_factor = params.get('merge_distance_threshold_factor', 0.15)
        self.merge_method = params.get('merge_method', 'average')
        self.state_mode = state_mode
        self.device = device; self.dtype = dtype; self.is_complex = dtype.is_complex
        self.steps_since_update = 0

        # State Buffers
        self.register_buffer('cluster_usage_ema', torch.ones(max_k, device=device, dtype=torch.float32) / max_k)
        self.register_buffer('cluster_variance_ema', torch.ones(max_k, device=device, dtype=torch.float32))
        self.register_buffer('active_mask', torch.zeros(max_k, dtype=torch.bool, device=device))
        self.active_mask[:self.min_k] = True
        self._update_cache()

    @torch.no_grad()
    def update_k(self, current_assignments: Optional[torch.Tensor], centroids: Optional[torch.Tensor], x_h: Optional[torch.Tensor]) -> None:
        self.steps_since_update += 1
        if self.steps_since_update < self.update_interval: return
        if current_assignments is None or centroids is None or x_h is None:
            logger.warning("DynK(SplitMerge) update skipped: Missing inputs.")
            return

        logger.debug("DynamicK (Split/Merge) Update Step...")
        self.steps_since_update = 0
        B, T, K_max_phys = current_assignments.shape
        if K_max_phys != self.max_k_physical: logger.error("Mismatch K_max!"); return # Safety check
        K_current_max, D = centroids.shape

        # --- 1. Update EMA Stats ---
        current_usage = current_assignments.sum(dim=(0, 1)).clamp(min=0) # (K_max_phys,)
        total_weight = current_usage.sum().clamp(min=1e-6)
        current_usage_norm = current_usage / total_weight
        self.cluster_usage_ema.mul_(self.usage_ema_decay).add_(current_usage_norm, alpha=1.0 - self.usage_ema_decay)

        # Variance EMA (based on L2 norm squared)
        token_norms_sq = x_h.abs().pow(2).sum(-1) if self.is_complex else x_h.pow(2).sum(-1) # (B, T) Real
        A = current_assignments
        cluster_weights = A.sum(dim=(0, 1)).clamp(min=1e-8) # (K,)
        e_norm_sq = (A * token_norms_sq.unsqueeze(-1)).sum(dim=(0, 1)) / cluster_weights
        # Variance requires E[Norm]^2 - more complex if norm is not squared L2
        # Simplified: Use variance of squared norms directly as proxy
        e_norm4 = (A * token_norms_sq.pow(2).unsqueeze(-1)).sum(dim=(0, 1)) / cluster_weights
        current_variance = (e_norm4 - e_norm_sq.pow(2)).clamp(min=0) # Variance of norm^2
        active_idxs_now = self.get_active_indices() # Get current active indices
        if len(active_idxs_now) > 0:
            self.cluster_variance_ema[active_idxs_now].mul_(self.usage_ema_decay).add_(current_variance[active_idxs_now], alpha=1.0 - self.usage_ema_decay)
            logger.debug(f"DynK - EMA Usage (Active): {self.cluster_usage_ema[active_idxs_now].cpu().numpy()}")
            logger.debug(f"DynK - EMA Variance (Active): {self.cluster_variance_ema[active_idxs_now].cpu().numpy()}")
        else: logger.warning("DynK - No active clusters found during EMA update.")


        # --- Calculate Thresholds ---
        num_active_current = self.active_mask.sum().item()
        if num_active_current == 0: return
        target_uniform_usage = 1.0 / num_active_current
        prune_threshold = self.config_params['prune_threshold_factor'] * target_uniform_usage
        reactivate_threshold = self.reactivation_threshold_factor * target_uniform_usage
        avg_active_variance = self.cluster_variance_ema[self.active_mask].mean() if num_active_current > 0 else 1.0
        split_threshold = self.config_params['split_variance_threshold_factor'] * avg_active_variance

        avg_dist = 0.0
        merge_threshold_sq = float('inf')
        if num_active_current > 1:
            active_centroids = centroids[self.active_mask] # Get only active centroids
            diffs = active_centroids.unsqueeze(1) - active_centroids.unsqueeze(0)
            dist_sq = diffs.abs().pow(2).sum(dim=-1)
            dist_sq.diagonal().fill_(float('inf'))
            finite_dists = dist_sq[dist_sq != float('inf')]
            if finite_dists.numel() > 0: avg_dist = finite_dists.sqrt().mean()
            merge_threshold_sq = (self.config_params['merge_distance_threshold_factor'] * avg_dist).pow(2)

        # --- 2. Reactivation ---
        inactive_indices = (~self.active_mask).nonzero(as_tuple=True)[0]
        if len(inactive_indices) > 0:
            can_reactivate = (self.cluster_usage_ema[inactive_indices] > reactivate_threshold)
            if can_reactivate.any():
                reactivate_idx_local = torch.argmax(self.cluster_usage_ema[inactive_indices[can_reactivate]])
                reactivate_idx_global = inactive_indices[can_reactivate][reactivate_idx_local]
                self.active_mask[reactivate_idx_global] = True
                logger.info(f"DynK - Reactivated cluster {reactivate_idx_global} (Usage > {reactivate_threshold:.4f}).")
                num_active_current += 1

        # --- 3. Pruning ---
        if num_active_current > self.min_k:
            can_prune = (self.cluster_usage_ema < prune_threshold) & self.active_mask
            if can_prune.any():
                prune_candidate_indices = can_prune.nonzero(as_tuple=True)[0]
                prune_idx_global = prune_candidate_indices[torch.argmin(self.cluster_usage_ema[prune_candidate_indices])]
                if num_active_current > self.min_k: # Double check count
                    self.active_mask[prune_idx_global] = False
                    self.cluster_usage_ema[prune_idx_global] = 1.0 / K_max_phys # Reset
                    self.cluster_variance_ema[prune_idx_global] = avg_active_variance if avg_active_variance > 0 else 1.0 # Reset variance
                    logger.info(f"DynK - Pruned cluster {prune_idx_global} (Usage < {prune_threshold:.4f}).")
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
                    logger.info(f"DynK - Splitting cluster {split_idx_global} (Var {self.cluster_variance_ema[split_idx_global]:.3f} > {split_threshold:.3f}) -> new slot {new_idx}.")
                    # Perform split (modify centroids.data)
                    if self.split_method == 'pca_stub': logger.warning("PCA split method stub.")
                    perturbation = torch.randn_like(centroids[split_idx_global]) * self.config_params['split_init_factor']
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

        # --- 5. Merging ---
        if num_active_current > self.min_k and avg_dist > 1e-6:
            active_idxs_now = self.active_mask.nonzero(as_tuple=True)[0] # Recompute active indices
            if len(active_idxs_now) > 1: # Check again
                active_centroids = centroids[active_idxs_now]
                diffs = active_centroids.unsqueeze(1) - active_centroids.unsqueeze(0); dist_sq = diffs.abs().pow(2).sum(dim=-1)
                dist_sq.diagonal().fill_(float('inf'))
                if (dist_sq < merge_threshold_sq).any():
                    min_dist_sq_val, min_idx_flat = torch.min(dist_sq.view(-1), dim=0)
                    row_idx = min_idx_flat // len(active_idxs_now); col_idx = min_idx_flat % len(active_idxs_now)
                    global_idx1 = active_idxs_now[row_idx]; global_idx2 = active_idxs_now[col_idx]
                    if num_active_current > self.min_k:
                        # Ensure indices are different (should be due to diagonal inf)
                        if global_idx1 == global_idx2: logger.error("Merge selected same index!"); return
                        # Merge idx2 into idx1
                        logger.info(f"DynK - Merging cluster {global_idx2} into {global_idx1} (Dist^2 {min_dist_sq_val:.3f} < {merge_threshold_sq:.3f}).")
                        # Update stats
                        usage1 = self.cluster_usage_ema[global_idx1]; usage2 = self.cluster_usage_ema[global_idx2]
                        total_usage = usage1 + usage2
                        if total_usage > 1e-8: w1 = usage1 / total_usage; w2 = usage2 / total_usage
                        else: w1 = w2 = 0.5
                        self.cluster_variance_ema[global_idx1] = w1 * self.cluster_variance_ema[global_idx1] + w2 * self.cluster_variance_ema[global_idx2] # Weighted avg var
                        self.cluster_usage_ema[global_idx1] = total_usage
                        # Merge Centroids
                        if self.merge_method == 'average': centroids.data[global_idx1] = (centroids.data[global_idx1] + centroids.data[global_idx2]) / 2.0
                        elif self.merge_method == 'keep_dominant': 
                            if usage2 > usage1: centroids.data[global_idx1] = centroids.data[global_idx2]
                        # Deactivate removed cluster
                        self.active_mask[global_idx2] = False; # Reset stats for removed
                        self.cluster_usage_ema[global_idx2] = 0.0; self.cluster_variance_ema[global_idx2] = 0.0
                        num_active_current -= 1

        # --- Final Update Cache ---
        self._update_cache()

    def get_active_indices(self) -> torch.Tensor:
        return self.active_indices_cache

    @torch.no_grad()
    def _update_cache(self) -> None:
        self.active_indices_cache = self.active_mask.nonzero(as_tuple=True)[0]