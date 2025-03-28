# src/pgs_net/modules/queen_computation.py
""" Computes Local/Global Queens and related Aux Losses. """
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, Tuple
from ..config import DEFAULT_PGS_FFN_CONFIG # Import default for structured access

logger = logging.getLogger(__name__)

# Helper for shared parameters
def get_shared_parameter(module, param_name, shared_params_dict, default_factory): # ... (implementation)

class QueenComputation(nn.Module):
    """
    Computes local queens per cluster and aggregates them into a global queen.
    Handles momentum for the global queen and calculates auxiliary losses.
    Supports parameter sharing and dynamic hyperparameters.
    """
    def __init__(self, d_head: int, num_heads: int, max_clusters: int, config: Dict[str, Any], shared_params: Optional[Dict[str, nn.Parameter]] = None):
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
        self.config = config.get('queens', DEFAULT_PGS_FFN_CONFIG['queens'])
        self.arch_config = config.get('architecture', DEFAULT_PGS_FFN_CONFIG['architecture'])
        self.param_sharing_config = self.arch_config.get('param_sharing', {})
        self.shared_params = shared_params if shared_params is not None else {}
        self.state_mode = self.arch_config.get('state_management', {}).get('mode', 'buffers')
        self.is_complex = self.arch_config.get('use_complex_representation', False)
        self.dtype = torch.complex64 if self.is_complex else torch.float32
        self.max_clusters = max_clusters
        self.d_head = d_head
        self.num_heads = num_heads # Needed if params are head-specific

        # --- Parameters ---
        # Global Weights Logits
        self.use_learnable_global_weights = self.config.get('use_learnable_global_weights', True)
        if self.use_learnable_global_weights:
            factory_qw_logits_head = lambda: nn.Parameter(torch.zeros(num_heads, max_clusters)) # Head-specific (H, K)
            factory_qw_logits_shared = lambda: nn.Parameter(torch.zeros(max_clusters)) # Shared (K,)
            share_weights = self.param_sharing_config.get('share_queen_weights', False)
            if share_weights:
                 self.global_weights_logits = get_shared_parameter(self, "queen_global_logits", self.shared_params, factory_qw_logits_shared) # Name matches getter logic
            else: self.global_weights_logits = factory_qw_logits_head()
            logger.info(f"Using learnable global queen weights (Shared={share_weights}).")
        else: logger.info("Using uniform global queen weights.")

        # --- State Buffers (if mode='buffers') ---
        if self.state_mode == 'buffers' and self.config.get('use_global_queen_momentum', True):
            # Shape: H, 1, D (complex or real) - stores EMA avg across batch
            self.register_buffer("global_queen_momentum", torch.zeros(num_heads, 1, d_head, dtype=self.dtype))
            logger.info("Initialized buffer for global queen momentum.")

    def forward(self, x_h: torch.Tensor, A: torch.Tensor, centroids_h: torch.Tensor, head_idx: int, state_in: Optional[Dict] = None, dynamic_params: Optional[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Performs queen computation.

        Args:
            x_h (torch.Tensor): Token embeddings for the current head (B, T, D).
            A (torch.Tensor): Assignment matrix (B, T, K_max).
            centroids_h (torch.Tensor): Centroids for the current head (K_max, D).
            head_idx (int): Index of the current head.
            state_in (Optional[Dict]): Input state for 'external' mode (e.g., {'momentum': tensor}).
            dynamic_params (Optional[Dict]): Dynamically learned hyperparameters.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
                - local_queens (B, K_max, D)
                - global_queen (B, D)
                - total_aux_loss (scalar float tensor)
                - state_out (dictionary containing state for 'external' mode)
        """
        B, T, D = x_h.shape
        K_max = self.max_clusters
        state_out: Dict[str, torch.Tensor] = {}
        current_momentum: Optional[torch.Tensor] = None
        logger.debug(f"[Head {head_idx}] Queen Computation started.")

        # --- Get Effective Hyperparameters ---
        use_momentum = dynamic_params.get('use_global_queen_momentum', self.config['use_global_queen_momentum']) if dynamic_params else self.config['use_global_queen_momentum']
        momentum_decay = dynamic_params.get('momentum_decay', self.config['momentum_decay']) if dynamic_params else self.config['momentum_decay']
        if not isinstance(momentum_decay, torch.Tensor): momentum_decay = torch.tensor(momentum_decay) # Ensure tensor for math
        diversity_weight = dynamic_params.get('diversity_aux_loss_weight', self.config['diversity_aux_loss_weight']) if dynamic_params else self.config['diversity_aux_loss_weight']
        entropy_weight = dynamic_params.get('entropy_weight', self.config['entropy_weight']) if dynamic_params else self.config['entropy_weight']
        use_diversity = self.config.get('use_diversity_aux_loss', True)
        use_entropy = self.config.get('use_entropy_regularization', False)
        entropy_target = self.config.get('entropy_target', 'assignments')
        entropy_type = self.config.get('entropy_type', 'maximize')

        # --- Retrieve State ---
        momentum_state_key = 'momentum' # Key for external state dict
        if use_momentum:
            state_mode = self.arch_config.get('state_management', {}).get('mode', 'buffers') # Get mode again
            logger.debug(f"[Head {head_idx}] Queen Momentum Mode: {state_mode}. Using momentum: True")
            if state_mode == 'buffers':
                current_momentum = self.global_queen_momentum[head_idx] # Shape (1, D)
                if current_momentum is not None: logger.debug(f"[Head {head_idx}] Buffer momentum norm: {current_momentum.norm().item():.3f}")
            elif state_mode == 'external' and state_in is not None:
                current_momentum = state_in.get(momentum_state_key) # Shape (1, D) expected
                if current_momentum is not None: logger.debug(f"[Head {head_idx}] External momentum norm: {current_momentum.norm().item():.3f}")
            # Initialize momentum state if it's None (first step or missing)
            if current_momentum is None:
                current_momentum = torch.zeros(1, self.d_head, dtype=self.dtype, device=x_h.device)
                logger.debug(f"[Head {head_idx}] Initialized momentum state.")
        else:
            logger.debug(f"[Head {head_idx}] Using momentum: False")

        # --- 1. Local Queens ---
        A_trans = A.transpose(1, 2) # (B, K_max, T)
        # Weighted average: (B, K_max, T) @ (B, T, D) -> (B, K_max, D)
        local_queens = torch.bmm(A_trans, x_h)
        cluster_sum_weights = A_trans.sum(dim=-1, keepdim=True).clamp(min=1e-8) # (B, K_max, 1)
        local_queens = local_queens / cluster_sum_weights
        # Handle cases where cluster sum is zero (NaNs) -> replace with zero vector?
        local_queens = torch.nan_to_num(local_queens, nan=0.0, posinf=0.0, neginf=0.0)

        # --- 2. Global Queen ---
        if self.use_learnable_global_weights:
             if self.param_sharing_config.get('share_queen_weights', False):
                 logits_to_use = self.global_weights_logits # Shared (K_max,)
             else: logits_to_use = self.global_weights_logits[head_idx] # Head-specific (K_max,)
             global_w_h = F.softmax(logits_to_use, dim=0) # (K_max,) - Real weights
        else: global_w_h = torch.ones(K_max, device=x_h.device) / K_max # Uniform weights

        # Weighted sum: (B, K_max, D) * (1, K_max, 1) -> sum(dim=1) -> (B, D)
        new_global_queen = (local_queens * global_w_h.view(1, -1, 1)).sum(dim=1)

        # --- 3. Apply Momentum ---
        global_queen = new_global_queen # Default if no momentum
        if use_momentum and current_momentum is not None:
             detached_new_queen_batch_avg = new_global_queen.detach().mean(0, keepdim=True) # Average over batch (1, D)
             decay = momentum_decay.to(current_momentum.device) # Use dynamic value, ensure device match

             # Update momentum state (shared or external)
             updated_momentum = decay * current_momentum + (1.0 - decay) * detached_new_queen_batch_avg
             logger.debug(f"[Head {head_idx}] Momentum Update: Decay={decay.item():.3f}, OldNorm={current_momentum.norm().item():.3f}, NewNorm={updated_momentum.norm().item():.3f}")

             if self.state_mode == 'buffers':
                  self.global_queen_momentum[head_idx] = updated_momentum # Update buffer
             elif self.state_mode == 'external':
                  state_out[momentum_state_key] = updated_momentum # Store for output

             # Use the updated momentum state for the forward pass (expand to batch size)
             global_queen = updated_momentum.expand(B, -1) # (B, D)

        # --- 4. Auxiliary Losses (Training only) ---
        total_aux_loss = torch.tensor(0.0, device=x_h.device, dtype=torch.float32) # Aux losses should be float
        if self.training:
            # Diversity Loss
            if use_diversity and diversity_weight > 1e-6 and K_max > 1:
                # Normalize local queens (use complex-aware norm)
                q_norms = torch.linalg.vector_norm(local_queens, ord=2, dim=-1, keepdim=True).clamp(min=1e-8)
                norm_local_q = local_queens / q_norms # (B, K_max, D)
                # Cosine sim: Re( q1 @ q2.conj().T )
                cos_sim = torch.bmm(norm_local_q, norm_local_q.transpose(1, 2).conj()).real # (B, K_max, K_max)
                mask = (1.0 - torch.eye(K_max, device=x_h.device)).unsqueeze(0) # Exclude self-similarity
                num_pairs = K_max * (K_max - 1)
                if num_pairs > 0:
                     diversity_loss = (cos_sim * mask).sum(dim=(1, 2)) / num_pairs # Mean pairwise sim per batch item
                     total_aux_loss += diversity_weight * diversity_loss.mean() # Average over batch
                     logger.debug(f"[Head {head_idx}] Diversity Loss: {diversity_loss.mean().item():.4f} (Weight: {diversity_weight:.4f})")

            # Entropy Regularization
            if use_entropy and entropy_weight > 1e-6:
                 entropy = 0.0
                 if entropy_target == 'assignments':
                      entropy_A = - (A * torch.log(A.clamp(min=1e-9))).sum(dim=-1) # (B, T)
                      entropy = entropy_A.mean() # Average over batch and time
                 elif entropy_target == 'global_weights' and self.use_learnable_global_weights:
                      # Use the actual weights used in aggregation
                      weights_prob = global_w_h # Shape (K_max,)
                      entropy_W = - (weights_prob * torch.log(weights_prob.clamp(min=1e-9))).sum()
                      entropy = entropy_W # Already scalar

                 sign = -1.0 if entropy_type == 'maximize' else 1.0
                 total_aux_loss += sign * entropy_weight * entropy
                 logger.debug(f"[Head {head_idx}] Entropy Loss ({entropy_target}, {entropy_type}): {entropy.item():.4f} (Weight: {entropy_weight:.4f})")
        # else: logger.debug(f"[Head {head_idx}] Skipping aux losses (eval mode).")

        logger.debug(f"[Head {head_idx}] Queen Computation complete.")
        return local_queens, global_queen, total_aux_loss, state_out

    def extra_repr(self) -> str:
        return f"state_mode={self.state_mode}, use_momentum={self.config.get('use_global_queen_momentum', True)}"