# src/pgs_net/modules/queen_computation.py
"""Computes Local/Global Queens and related Aux Losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, Tuple, Union
from ..config import DEFAULT_PGS_FFN_CONFIG  # Import default for structured access

logger = logging.getLogger(__name__)


# Helper for shared parameters
def get_shared_parameter(
    module: nn.Module, param_name: str, shared_params_dict: Dict[str, nn.Parameter], default_factory
) -> nn.Parameter:
    """Gets or creates a shared parameter and registers it locally."""
    if param_name not in shared_params_dict:
        shared_params_dict[param_name] = default_factory()
        logger.debug(f"Created shared parameter: {param_name} with shape {shared_params_dict[param_name].shape}")
    shared_attr_name = f"_shared_{param_name}"
    if not hasattr(module, shared_attr_name):
        # Use setattr to store the actual parameter reference from the shared dict
        setattr(module, shared_attr_name, shared_params_dict[param_name])
    return shared_params_dict[param_name]


class QueenComputation(nn.Module):
    """
    Computes local queens per cluster and aggregates them into a global queen.
    Handles momentum for the global queen and calculates auxiliary losses.
    Supports parameter sharing and dynamic hyperparameters.
    """

    def __init__(
        self,
        d_head: int,
        num_heads: int,
        max_clusters: int,
        config: Dict[str, Any],
        shared_params: Optional[Dict[str, nn.Parameter]] = None,
    ):
        """
        Initializes the QueenComputation module.

        Args:
            d_head (int): Dimension per head.
            num_heads (int): Total number of heads.
            max_clusters (int): Maximum number of clusters (K_max).
            config (Dict[str, Any]): Full PGS_FFN configuration.
            shared_params (Optional[Dict[str, nn.Parameter]]): Dictionary for shared parameters.
        """
        super().__init__()
        self.config = config.get("queens", DEFAULT_PGS_FFN_CONFIG["queens"])
        self.arch_config = config.get("architecture", DEFAULT_PGS_FFN_CONFIG["architecture"])
        self.param_sharing_config = self.arch_config.get("param_sharing", {})
        self.shared_params = shared_params if shared_params is not None else {}
        self.state_mode = self.arch_config.get("state_management", {}).get("mode", "buffers")
        self.is_complex = self.arch_config.get("use_complex_representation", False)
        self.dtype = torch.complex64 if self.is_complex else torch.float32
        self.max_clusters = max_clusters
        self.d_head = d_head
        self.num_heads = num_heads

        # --- Parameters ---
        self.use_learnable_global_weights = self.config.get("use_learnable_global_weights", True)
        share_weights = self.param_sharing_config.get("share_queen_weights", False)
        if self.use_learnable_global_weights:
            factory_qw_logits_head = lambda: nn.Parameter(torch.zeros(num_heads, max_clusters))  # Head-specific (H, K)
            factory_qw_logits_shared = lambda: nn.Parameter(torch.zeros(max_clusters))  # Shared (K,)
            if share_weights:
                # Store reference to shared param, accessed via property getter
                self._shared_queen_global_logits = get_shared_parameter(
                    self, "queen_global_logits", self.shared_params, factory_qw_logits_shared
                )
            else:
                self.global_weights_logits_local = factory_qw_logits_head()  # Head-specific (H, K)
            logger.info(f"Using learnable global queen weights (Shared={share_weights}).")
        else:
            logger.info("Using uniform global queen weights.")

        # --- State Buffers (if mode='buffers') ---
        if self.state_mode == "buffers" and self.config.get("use_global_queen_momentum", True):
            self.register_buffer("global_queen_momentum", torch.zeros(num_heads, 1, d_head, dtype=self.dtype))
            logger.info("Initialized buffer for global queen momentum.")

    # Property getter for consistent access to logits
    @property
    def effective_global_weights_logits(self) -> nn.Parameter:
        if hasattr(self, "_shared_queen_global_logits"):
            return self._shared_queen_global_logits
        elif hasattr(self, "global_weights_logits_local"):
            return self.global_weights_logits_local
        else:
            return None  # Should not happen if learnable weights enabled

    def forward(
        self,
        x_h: torch.Tensor,
        A: torch.Tensor,
        centroids_h: torch.Tensor,
        head_idx: int,
        state_in: Optional[Dict] = None,
        dynamic_params: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Performs queen computation."""
        B, T, D = x_h.shape
        K_max = self.max_clusters
        state_out: Dict[str, torch.Tensor] = {}
        current_momentum: Optional[torch.Tensor] = None
        logger.debug(f"[Head {head_idx}] Queen Computation started.")

        # --- Get Effective Hyperparameters ---
        use_momentum = (
            dynamic_params.get("use_global_queen_momentum", self.config.get("use_global_queen_momentum", True))
            if dynamic_params
            else self.config.get("use_global_queen_momentum", True)
        )
        momentum_decay_val = (
            dynamic_params.get("momentum_decay", self.config.get("momentum_decay", 0.9))
            if dynamic_params
            else self.config.get("momentum_decay", 0.9)
        )
        momentum_decay = torch.tensor(momentum_decay_val, device=x_h.device, dtype=torch.float32)  # Ensure tensor, float
        diversity_weight = (
            dynamic_params.get("diversity_aux_loss_weight", self.config.get("diversity_aux_loss_weight", 0.1))
            if dynamic_params
            else self.config.get("diversity_aux_loss_weight", 0.1)
        )
        entropy_weight = (
            dynamic_params.get("entropy_weight", self.config.get("entropy_weight", 0.001))
            if dynamic_params
            else self.config.get("entropy_weight", 0.001)
        )
        use_diversity = self.config.get("use_diversity_aux_loss", True)
        use_entropy = self.config.get("use_entropy_regularization", False)
        entropy_target = self.config.get("entropy_target", "assignments")
        entropy_type = self.config.get("entropy_type", "maximize")

        # --- Retrieve State ---
        momentum_state_key = "momentum"
        if use_momentum:
            state_mode = self.arch_config.get("state_management", {}).get("mode", "buffers")
            logger.debug(f"[Head {head_idx}] Queen Momentum Mode: {state_mode}. Using momentum: True")
            if state_mode == "buffers":
                current_momentum = self.global_queen_momentum[head_idx]  # Shape (1, D)
            elif state_mode == "external" and state_in is not None:
                current_momentum = state_in.get(momentum_state_key)  # Shape (1, D) expected
            if current_momentum is None:
                current_momentum = torch.zeros(1, self.d_head, dtype=self.dtype, device=x_h.device)
            else:
                logger.debug(f"[Head {head_idx}] Momentum state norm: {torch.linalg.vector_norm(current_momentum).item():.3f}")
        else:
            logger.debug(f"[Head {head_idx}] Using momentum: False")

        # --- 1. Local Queens ---
        A_trans = A.transpose(1, 2).float()  # (B, K_max, T) - Ensure float for bmm if x_h is complex
        x_h_float = x_h if not self.is_complex else x_h.to(torch.complex64)  # Ensure consistent type for bmm
        local_queens = torch.bmm(A_trans, x_h_float)  # (B, K_max, D)
        cluster_sum_weights = A_trans.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, K_max, 1)
        local_queens = local_queens / cluster_sum_weights
        local_queens = torch.nan_to_num(local_queens, nan=0.0, posinf=0.0, neginf=0.0)

        # --- 2. Global Queen ---
        if self.use_learnable_global_weights and hasattr(self, "effective_global_weights_logits"):
            logits_param = self.effective_global_weights_logits
            if logits_param.dim() == 2:  # Head-specific (H, K)
                logits_to_use = logits_param[head_idx]
            else:  # Shared (K,)
                logits_to_use = logits_param
            global_w_h = F.softmax(logits_to_use, dim=0)  # (K_max,)
        else:
            global_w_h = torch.ones(K_max, device=x_h.device) / K_max

        new_global_queen = (local_queens * global_w_h.view(1, -1, 1)).sum(dim=1)  # (B, D)

        # --- 3. Apply Momentum ---
        global_queen = new_global_queen
        if use_momentum and current_momentum is not None:
            detached_new_queen_batch_avg = new_global_queen.detach().mean(0, keepdim=True)  # (1, D)
            decay = momentum_decay.to(current_momentum.device)

            updated_momentum = decay * current_momentum + (1.0 - decay) * detached_new_queen_batch_avg  # (1, D)
            logger.debug(
                f"[Head {head_idx}] Momentum Update: Decay={decay.item():.3f}, OldNorm={torch.linalg.vector_norm(current_momentum).item():.3f}, NewNorm={torch.linalg.vector_norm(updated_momentum).item():.3f}"
            )

            if self.state_mode == "buffers":
                self.global_queen_momentum[head_idx] = updated_momentum
            elif self.state_mode == "external":
                state_out[momentum_state_key] = updated_momentum

            global_queen = updated_momentum.expand(B, -1)  # Expand buffer state (B, D)

        # --- 4. Auxiliary Losses (Training only) ---
        total_aux_loss = torch.tensor(0.0, device=x_h.device, dtype=torch.float32)
        if self.training:
            # Diversity Loss
            if use_diversity and diversity_weight > 1e-6 and K_max > 1:
                q_norms = torch.linalg.vector_norm(local_queens, ord=2, dim=-1, keepdim=True).clamp(min=1e-8)
                norm_local_q = local_queens / q_norms
                cos_sim = torch.bmm(norm_local_q, norm_local_q.transpose(1, 2).conj()).real
                mask = (1.0 - torch.eye(K_max, device=x_h.device)).unsqueeze(0)
                num_pairs = K_max * (K_max - 1)
                if num_pairs > 0:
                    diversity_loss = (cos_sim * mask).sum(dim=(1, 2)) / num_pairs
                    total_aux_loss += diversity_weight * diversity_loss.mean()
                    # logger.debug(f"[Head {head_idx}] Diversity Loss: {diversity_loss.mean().item():.4f} (Weight: {diversity_weight:.4f})")

            # Entropy Regularization
            if use_entropy and entropy_weight > 1e-6:
                entropy = 0.0
                if entropy_target == "assignments":
                    entropy_A = -(A * torch.log(A.clamp(min=1e-9))).sum(dim=-1).mean()  # Avg over B, T
                    entropy = entropy_A
                elif entropy_target == "global_weights" and self.use_learnable_global_weights:
                    weights_prob = global_w_h  # Shape (K_max,)
                    entropy_W = -(weights_prob * torch.log(weights_prob.clamp(min=1e-9))).sum()
                    entropy = entropy_W
                sign = -1.0 if entropy_type == "maximize" else 1.0
                total_aux_loss += sign * entropy_weight * entropy
                # logger.debug(f"[Head {head_idx}] Entropy Loss ({entropy_target}, {entropy_type}): {entropy.item():.4f} (Weight: {entropy_weight:.4f})")

        # logger.debug(f"[Head {head_idx}] Queen Computation complete.")
        return local_queens, global_queen, total_aux_loss, state_out

    def extra_repr(self) -> str:
        learn_w = self.config.get("use_learnable_global_weights", True)
        share_w = self.param_sharing_config.get("share_queen_weights", False)
        mom = self.config.get("use_global_queen_momentum", True)
        return f"state={self.state_mode}, momentum={mom}, learn_w={learn_w}, share_w={share_w and learn_w}"
