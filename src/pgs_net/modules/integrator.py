# src/pgs_net/modules/integrator.py
""" Integrates forces to produce the final token update. """
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, Tuple
import math
from ..config import DEFAULT_PGS_FFN_CONFIG
from .normalization import Normalization, RMSNormImpl, AdaptiveGroupNorm # Import norm types
from .complex_utils import ComplexRMSNorm # Keep direct import for type check maybe
from .placeholders import StubNormalization # Import stub

logger = logging.getLogger(__name__)

# Helper for shared parameters
def get_shared_parameter(module, param_name, shared_params_dict, default_factory): # ... (implementation)

class UpdateIntegrator(nn.Module):
    """
    Integrates the calculated force vector using momentum, decay, gating,
    clipping, normalization, and applies the residual connection.
    Manages state for token momentum and Boids alignment.
    Supports parameter sharing and dynamic hyperparameters.
    """
    def __init__(self, d_head: int, num_heads: int, config: Dict, shared_params: Optional[Dict] = None):
        """ Initializes the UpdateIntegrator. """
        super().__init__()
        self.config = config.get('integration', DEFAULT_PGS_FFN_CONFIG['integration'])
        self.arch_config = config.get('architecture', DEFAULT_PGS_FFN_CONFIG['architecture'])
        self.force_config = config.get('update_forces', DEFAULT_PGS_FFN_CONFIG['update_forces']) # Needed for Boids check
        self.state_mode = self.arch_config.get('state_management', {}).get('mode', 'buffers')
        self.boids_state_type = self.arch_config.get('state_management', {}).get('boids_alignment_state_type', 'avg_update')
        self.param_sharing_config = self.arch_config.get('param_sharing', {})
        self.shared_params = shared_params if shared_params is not None else {}

        self.is_complex = self.arch_config.get('use_complex_representation', False)
        self.dtype = torch.complex64 if self.is_complex else torch.float32
        self.d_head = d_head; self.num_heads = num_heads

        # --- Parameters / Layers (Potentially Shared) ---
        # Decay Logits
        if self.config.get('use_learnable_decay', True):
            factory_decay = lambda: nn.Parameter(torch.ones(num_heads) * 2.0) # Init near sigmoid(2)~0.88
            factory_shared_decay = lambda: nn.Parameter(torch.tensor(2.0))
            share = self.param_sharing_config.get('share_integrator_params', False)
            if share: self.decay_logits = get_shared_parameter(self, "decay_logits", self.shared_params, factory_shared_decay)
            else: self.decay_logits = factory_decay()
            logger.info(f"Using learnable decay (Shared={share}).")
        # Gate Layer
        if self.config.get('use_gate', True):
            gate_input_dim = 2 * d_head if self.is_complex else d_head
            factory_gate = lambda: nn.Linear(gate_input_dim, 1)
            share = self.param_sharing_config.get('share_integrator_params', False) # Share gate layer? Makes sense.
            self.gate_layer = get_shared_parameter(self, "gate_layer", self.shared_params, factory_gate) if share else factory_gate()
            logger.info(f"Using update gate (Shared={share}).")

        # --- Normalization Layer ---
        self.norm_layer: Optional[Normalization] = None
        norm_eps = self.config.get('rms_norm_eps', 1e-8)
        if self.config.get('use_rms_norm', True): # Note: Flag name is now misleading if type != rmsnorm
             norm_type = self.arch_config.get("normalization_type", "rmsnorm")
             logger.info(f"Initializing normalization layer: {norm_type}")
             if norm_type == "adagroupnorm":
                 num_groups = self.arch_config.get("adagroupnorm_groups", 4)
                 if d_head % num_groups == 0:
                      self.norm_layer = AdaptiveGroupNorm(num_groups, d_head, eps=norm_eps, use_complex=self.is_complex)
                 else: logger.error(f"Cannot use AdaGroupNorm: d_head={d_head} not divisible by groups={num_groups}. Using RMSNorm fallback."); norm_type="rmsnorm"
             # Default or fallback to RMSNorm
             if norm_type == "rmsnorm":
                 self.norm_layer = RMSNormImpl(d_head, eps=norm_eps, use_complex=self.is_complex)
             elif norm_type != 'none': # Handle unknown types or explicitly disabled norm
                 self.norm_layer = StubNormalization() # Use stub if unknown type specified
                 logger.warning(f"Unknown normalization type '{norm_type}'. Using identity.")
        else: logger.info("Normalization disabled for integrator.")


        # --- Dropout ---
        self.dropout: Optional[nn.Dropout] = None
        if self.config.get('use_dropout', True):
            self.dropout_rate = self.config.get('dropout_rate', 0.1)
            if self.dropout_rate > 0:
                 self.dropout = nn.Dropout(p=self.dropout_rate)
                 logger.info(f"Using dropout with p={self.dropout_rate}.")
            else: logger.info("Dropout enabled but rate is 0.")
        else: logger.info("Dropout disabled for integrator.")

        # --- State Buffers (if mode='buffers') ---
        self.use_token_momentum = self.config.get('use_token_momentum', False)
        self.needs_boids_state = self.force_config.get('use_boids_rules', False) # Check if boids is on globally
        if self.state_mode == 'buffers':
             if self.use_token_momentum:
                 logger.warning("Using simplified 'buffers' state for token momentum (avg force).")
                 self.register_buffer("last_token_force_avg", torch.zeros(num_heads, 1, self.d_head, dtype=self.dtype))
             if self.needs_boids_state and self.boids_state_type == 'avg_update':
                 logger.info("Using simplified 'buffers' state for Boids alignment (avg update).")
                 self.register_buffer("boids_last_processed_update_avg", torch.zeros(num_heads, 1, self.d_head, dtype=self.dtype))


    def forward(self,
                x_h: torch.Tensor,
                base_update_force: torch.Tensor,
                density_mod_factor: torch.Tensor,
                head_idx: int,
                state_in: Optional[Dict] = None,
                dynamic_params: Optional[Dict] = None,
                analysis_data: Optional[Dict] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Integrates the force, applies dynamics, and computes the residual update.

        Args:
            x_h: Token embeddings (B, T, D).
            base_update_force: Raw combined force (B, T, D).
            density_mod_factor: Density modulation factor (B, T, 1) or scalar 1.0.
            head_idx: Current head index.
            state_in: Input state (for external mode).
            dynamic_params: Dynamic hyperparameters.
            analysis_data: Dictionary for analysis results.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict]:
                - x_out: Updated token embeddings (B, T, D).
                - aux_loss: Auxiliary loss from this module (usually 0).
                - state_out: Output state (for external mode).
        """
        B, T, D = x_h.shape
        force = base_update_force
        state_out: Dict[str, torch.Tensor] = {}
        integ_analysis = {} if analysis_data is not None else None
        logger.debug(f"[Head {head_idx}] Update Integration started. Initial force norm avg: {torch.linalg.vector_norm(force, dim=-1).mean().item():.4f}")

        # --- Conditional Computation Gating ---
        update_mask = torch.ones(B, T, 1, device=x_h.device, dtype=torch.bool)
        if self.arch_config.get('use_conditional_computation', False):
             force_norm_sq = force.abs().pow(2).sum(dim=-1, keepdim=True) # Real norm sq
             threshold_sq = self.arch_config.get('conditional_comp_threshold', 0.01) ** 2
             update_mask = force_norm_sq > threshold_sq
             skip_ratio = 1.0 - update_mask.float().mean().item()
             logger.debug(f"[Head {head_idx}] CondComp: Skipping {skip_ratio*100:.2f}% updates (Threshold^2={threshold_sq:.4g})")
             if integ_analysis is not None: integ_analysis["conditional_skip_ratio"] = skip_ratio
             force = force * update_mask.to(force.dtype) # Zero out forces for skipped tokens

        # --- Token Momentum ---
        if self.use_token_momentum:
            token_mom_decay = dynamic_params.get('token_momentum_decay', self.config['token_momentum_decay']) if dynamic_params else self.config['token_momentum_decay']
            if not isinstance(token_mom_decay, torch.Tensor): token_mom_decay = torch.tensor(token_mom_decay)
            token_mom_decay = token_mom_decay.to(force.device)
            logger.debug(f"[Head {head_idx}] Token Momentum Mode: {self.state_mode}, Decay={token_mom_decay.item():.3f}")

            current_force_avg = force.mean(dim=(0, 1), keepdim=True) # (1, 1, D) - Masked average
            if self.state_mode == 'buffers':
                 last_force_avg = self.last_token_force_avg[head_idx] # Shape (1, D) -> (1, 1, D)
                 force_avg_with_mom = token_mom_decay * last_force_avg + current_force_avg
                 self.last_token_force_avg[head_idx] = force_avg_with_mom.detach() # Update buffer
                 momentum_gain = force_avg_with_mom / current_force_avg.clamp(min=1e-8) # Approx gain
                 force = force * momentum_gain # Apply gain to original masked force
                 if integ_analysis: integ_analysis["token_momentum_buffer_norm"] = last_force_avg.norm().item()
            elif self.state_mode == 'external':
                 mom_state_key = f'token_momentum_h{head_idx}'
                 last_force_state = state_in.get(mom_state_key) if state_in else None # Expect (1,1,D) avg or full (B,T,D)? Assume avg.
                 if last_force_state is None: last_force_state = torch.zeros_like(current_force_avg)
                 if last_force_state.shape != current_force_avg.shape: # Handle shape mismatch if full state expected
                     logger.warning(f"Token momentum state shape mismatch: Expected {current_force_avg.shape}, Got {last_force_state.shape}. Using average.")
                     last_force_state = last_force_state.mean(dim=(0,1), keepdim=True) # Fallback to average
                 updated_state = token_mom_decay * last_force_state + current_force_avg
                 state_out[mom_state_key] = updated_state.detach() # Store average state
                 momentum_gain = updated_state / current_force_avg.clamp(min=1e-8)
                 force = force * momentum_gain
                 if integ_analysis: integ_analysis["token_momentum_external_norm"] = last_force_state.norm().item()
            if integ_analysis: integ_analysis["force_norm_after_momentum"] = torch.linalg.vector_norm(force, dim=-1).mean().item()

        # --- Decay & Density ---
        use_learn_decay = dynamic_params.get('use_learnable_decay', self.config['use_learnable_decay']) if dynamic_params else self.config['use_learnable_decay']
        decay_val: Union[float, torch.Tensor] = 1.0
        if use_learn_decay and hasattr(self, 'decay_logits'):
             share = self.param_sharing_config.get('share_integrator_params', False)
             logits = self.decay_logits if not share else self.decay_logits # Access potentially shared
             decay_val = torch.sigmoid(logits[head_idx] if not share else logits)
        else: decay_val = dynamic_params.get('fixed_decay_value', self.config['fixed_decay_value']) if dynamic_params else self.config['fixed_decay_value']
        if not isinstance(decay_val, torch.Tensor): decay_val = torch.tensor(decay_val)
        decay_val = decay_val.to(force.device)

        # Apply density modulation to decay factor
        density_target = self.force_config.get('density_modulation_target', [])
        effective_decay = decay_val * density_mod_factor if 'decay' in density_target else decay_val # density_mod_factor is (B, T, 1) or 1.0
        scaled_force = effective_decay * force
        if integ_analysis:
            integ_analysis["effective_decay_avg"] = effective_decay.mean().item() if isinstance(effective_decay, torch.Tensor) else effective_decay
            integ_analysis["force_norm_after_decay"] = torch.linalg.vector_norm(scaled_force, dim=-1).mean().item()

        # --- Gate ---
        gated_force = scaled_force
        use_gate = dynamic_params.get('use_gate', self.config['use_gate']) if dynamic_params else self.config['use_gate']
        if use_gate and hasattr(self, 'gate_layer'):
            # Input to gate layer uses original x_h (unconditioned by cross-layer?) - Yes.
            gate_input = torch.cat([x_h.real, x_h.imag], dim=-1) if self.is_complex else x_h
            gate_val = torch.sigmoid(self.gate_layer(gate_input)) # (B, T, 1) real
            gated_force = gate_val * scaled_force
            if integ_analysis:
                 integ_analysis["gate_value_avg"] = gate_val.mean().item()
                 integ_analysis["force_norm_after_gate"] = torch.linalg.vector_norm(gated_force, dim=-1).mean().item()

        # --- Integration Steps on Gated Force ---
        processed_update = gated_force # Start with the result

        # --- Dropout ---
        if self.dropout is not None and self.training:
            processed_update = self.dropout(processed_update)
            if integ_analysis: integ_analysis["update_norm_after_dropout"] = torch.linalg.vector_norm(processed_update, dim=-1).mean().item()

        # --- Clipping (Before Norm) ---
        norm_before_clip = torch.linalg.vector_norm(processed_update, dim=-1, keepdim=True)
        if integ_analysis: integ_analysis["update_norm_before_clip"] = norm_before_clip.mean().item()
        if self.arch_config.get('use_gradient_clipping', False):
            clip_thr = self.arch_config.get('gradient_clip_threshold', 1.0)
            scale = (clip_thr / norm_before_clip.clamp(min=1e-6)).clamp(max=1.0) # Real scale factor
            clipped_update = processed_update * scale
            if integ_analysis:
                 integ_analysis["clip_scale_avg"] = scale.mean().item()
                 integ_analysis["update_norm_after_clip"] = torch.linalg.vector_norm(clipped_update, dim=-1).mean().item()
                 if (scale < 0.999).any(): logger.debug(f"[Head {head_idx}] Update clipped. Max Norm: {norm_before_clip.max().item():.3f} > {clip_thr}")
            processed_update = clipped_update

        # --- Normalization ---
        norm_before_norm = torch.linalg.vector_norm(processed_update, dim=-1, keepdim=True)
        if integ_analysis and 'update_norm_before_norm' not in integ_analysis: integ_analysis["update_norm_before_norm"] = norm_before_norm.mean().item()
        if self.norm_layer is not None:
            processed_update = self.norm_layer(processed_update)
            if integ_analysis: integ_analysis["update_norm_after_norm"] = torch.linalg.vector_norm(processed_update, dim=-1).mean().item()
            logger.debug(f"[Head {head_idx}] Applied Normalization: {self.arch_config.get('normalization_type', 'rmsnorm')}")

        # --- Store State for Boids Alignment ---
        if self.needs_boids_state:
             boids_state_key = f'boids_last_update_h{head_idx}'
             if self.state_mode == 'buffers':
                 # Store average of the processed update
                 self.boids_last_processed_update_avg[head_idx] = processed_update.mean(dim=(0, 1), keepdim=True).detach()
             elif self.state_mode == 'external':
                 # Store full update state (B, T, D) if requested, else average
                 if self.boids_state_type == 'full_update':
                      state_out[boids_state_key] = processed_update.detach()
                 else: # avg_update
                      state_out[boids_state_key] = processed_update.mean(dim=(0,1), keepdim=True).detach() # Store avg

        # --- Residual Connection ---
        x_out = x_h + processed_update

        # --- Update main analysis dict ---
        if analysis_data is not None and integ_analysis is not None:
            analysis_data.update(integ_analysis)

        logger.debug(f"[Head {head_idx}] Integration complete. Output norm avg: {torch.linalg.vector_norm(x_out, dim=-1).mean().item():.4f}")
        return x_out, torch.tensor(0.0, device=x_h.device), state_out

    def extra_repr(self) -> str:
        return f"state_mode={self.state_mode}, use_momentum={self.use_token_momentum}, use_gate={hasattr(self, 'gate_layer')}, norm={self.arch_config.get('normalization_type', 'rmsnorm')}, dropout={self.config.get('dropout_rate', 0.0) if self.dropout else 0.0}"