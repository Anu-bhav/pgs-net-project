# src/pgs_net/modules/integrator.py
"""Integrates forces to produce the final token update."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, Tuple, Union
import math
from ..config import DEFAULT_PGS_FFN_CONFIG
from .normalization import Normalization, RMSNormImpl, AdaptiveGroupNorm  # Import norm types
from .complex_utils import ComplexRMSNorm, ComplexLinear  # Need CL for gate maybe? No, gate input is real projection
from .placeholders import StubNormalization  # Import stub

logger = logging.getLogger(__name__)


# Helper for shared parameters
# Assume this helper function is defined elsewhere (e.g., in utils or pgs_ffn) or define here
def get_shared_parameter(
    module: nn.Module, param_name: str, shared_params_dict: Dict[str, nn.Parameter], default_factory
) -> nn.Parameter:
    """Gets or creates a shared parameter and registers it locally."""
    if param_name not in shared_params_dict:
        shared_params_dict[param_name] = default_factory()
        logger.debug(f"Created shared parameter: {param_name} with shape {shared_params_dict[param_name].shape}")
    # Register the shared parameter in the current module so it's included in state_dict etc.
    # Use a unique name to avoid conflicts if multiple modules share the same base name
    shared_attr_name = f"_shared_{param_name}"
    if not hasattr(module, shared_attr_name):
        # Use setattr to store the actual parameter reference from the shared dict
        setattr(module, shared_attr_name, shared_params_dict[param_name])
    return shared_params_dict[param_name]


class UpdateIntegrator(nn.Module):
    """
    Integrates the calculated force vector using momentum, decay, gating,
    clipping, normalization, and applies the residual connection.
    Manages state for token momentum and Boids alignment.
    Supports parameter sharing and dynamic hyperparameters.
    """

    def __init__(self, d_head: int, num_heads: int, config: Dict, shared_params: Optional[Dict] = None):
        """
        Initializes the UpdateIntegrator.

        Args:
            d_head (int): Dimension per head.
            num_heads (int): Total number of heads.
            config (Dict): Full PGS_FFN configuration.
            shared_params (Optional[Dict]): Dictionary for shared parameters.
        """
        super().__init__()
        self.config = config.get("integration", DEFAULT_PGS_FFN_CONFIG["integration"])
        self.arch_config = config.get("architecture", DEFAULT_PGS_FFN_CONFIG["architecture"])
        self.force_config = config.get("update_forces", DEFAULT_PGS_FFN_CONFIG["update_forces"])  # Needed for Boids check
        self.state_mode = self.arch_config.get("state_management", {}).get("mode", "buffers")
        self.boids_state_type = self.arch_config.get("state_management", {}).get("boids_alignment_state_type", "avg_update")
        self.param_sharing_config = self.arch_config.get("param_sharing", {})
        self.shared_params = shared_params if shared_params is not None else {}

        self.is_complex = self.arch_config.get("use_complex_representation", False)
        self.dtype = torch.complex64 if self.is_complex else torch.float32
        self.d_head = d_head
        self.num_heads = num_heads

        # --- Parameters / Layers (Potentially Shared) ---
        share_integrator = self.param_sharing_config.get("share_integrator_params", False)
        # Decay Logits
        self.decay_logits: Optional[nn.Parameter] = None  # For head-specific
        self._shared_decay_logits: Optional[nn.Parameter] = None  # For shared access
        if self.config.get("use_learnable_decay", True):
            factory_decay_head = lambda: nn.Parameter(torch.ones(num_heads) * 2.0)  # Init near sigmoid(2)~0.88
            factory_decay_shared = lambda: nn.Parameter(torch.tensor(2.0))
            if share_integrator:
                # Store reference under _shared name using helper
                get_shared_parameter(self, "decay_logits", self.shared_params, factory_decay_shared)
            else:
                self.decay_logits = factory_decay_head()  # Head-specific (H,)
            logger.info(f"Using learnable decay (Shared={share_integrator}).")
        # Gate Layer
        self.gate_layer: Optional[nn.Linear] = None  # For head-specific (if not shared)
        self._shared_gate_layer: Optional[nn.Linear] = None  # For shared access
        if self.config.get("use_gate", True):
            gate_input_dim = 2 * d_head if self.is_complex else d_head
            factory_gate = lambda: nn.Linear(gate_input_dim, 1)
            # Gate layer usually applied per token, sharing across heads makes sense
            # Use get_shared_parameter which registers '_shared_gate_layer'
            get_shared_parameter(self, "gate_layer", self.shared_params, factory_gate)
            logger.info(f"Using update gate (Shared=True).")  # Assuming shared is default intent

        # --- Normalization Layer ---
        self.norm_layer: Optional[Normalization] = None
        norm_eps = self.config.get("rms_norm_eps", 1e-8)
        if self.config.get("use_rms_norm", True):  # Flag controls if *any* norm is used
            norm_type = self.arch_config.get("normalization_type", "rmsnorm")
            logger.info(f"Initializing normalization layer: {norm_type}")
            if norm_type == "adagroupnorm":
                num_groups = self.arch_config.get("adagroupnorm_groups", 4)
                if d_head % num_groups == 0:
                    try:
                        self.norm_layer = AdaptiveGroupNorm(num_groups, d_head, eps=norm_eps, use_complex=self.is_complex)
                    except NotImplementedError:
                        logger.warning("Complex AdaGroupNorm not implemented. Using RMSNorm fallback.")
                        norm_type = "rmsnorm"  # Force fallback
                    except Exception as e:
                        logger.error(f"Failed to init AdaGroupNorm: {e}. Using RMSNorm fallback.")
                        norm_type = "rmsnorm"
                else:
                    logger.error(
                        f"Cannot use AdaGroupNorm: d_head={d_head} not divisible by groups={num_groups}. Using RMSNorm fallback."
                    )
                    norm_type = "rmsnorm"

            if norm_type == "rmsnorm":  # Default or fallback
                self.norm_layer = RMSNormImpl(d_head, eps=norm_eps, use_complex=self.is_complex)
            elif norm_type != "none" and self.norm_layer is None:  # Handle unknown types
                self.norm_layer = StubNormalization()
                logger.warning(f"Unknown normalization type '{norm_type}'. Using identity.")
        else:
            logger.info("Normalization disabled for integrator.")

        # --- Dropout ---
        self.dropout: Optional[nn.Dropout] = None
        if self.config.get("use_dropout", True):
            self.dropout_rate = self.config.get("dropout_rate", 0.1)
            if self.dropout_rate > 0:
                self.dropout = nn.Dropout(p=self.dropout_rate)
                logger.info(f"Using dropout with p={self.dropout_rate}.")
            else:
                logger.info("Dropout enabled but rate is 0.")
        else:
            logger.info("Dropout disabled for integrator.")

        # --- State Buffers (if mode='buffers') ---
        self.use_token_momentum = self.config.get("use_token_momentum", False)
        self.needs_boids_state = self.force_config.get("use_boids_rules", False)  # Check if boids is on globally
        if self.state_mode == "buffers":
            if self.use_token_momentum:
                logger.debug("Initializing buffer for token momentum (avg force).")
                self.register_buffer("last_token_force_avg", torch.zeros(num_heads, 1, self.d_head, dtype=self.dtype))
            if self.needs_boids_state and self.boids_state_type == "avg_update":
                logger.info("Initializing buffer for Boids alignment (avg update).")
                self.register_buffer("boids_last_processed_update_avg", torch.zeros(num_heads, 1, self.d_head, dtype=self.dtype))

    # --- Property Getters for Consistent Parameter Access ---
    @property
    def effective_decay_logits(self) -> Optional[nn.Parameter]:
        """Gets the decay logits parameter (shared or local)."""
        share = self.param_sharing_config.get("share_integrator_params", False)
        if share:
            return getattr(self, "_shared_decay_logits", None)
        else:
            return getattr(self, "decay_logits_local", None)

    @property
    def effective_gate_layer(self) -> Optional[nn.Linear]:
        """Gets the gate layer parameter (shared or local - assuming shared)."""
        # Gate layer assumed shared via get_shared_parameter
        return getattr(self, "_shared_gate_layer", None)

    def forward(
        self,
        x_h: torch.Tensor,
        base_update_force: torch.Tensor,
        density_mod_factor: torch.Tensor,
        head_idx: int,
        state_in: Optional[Dict] = None,
        dynamic_params: Optional[Dict] = None,
        analysis_data: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Integrates the force, applies dynamics, and computes the residual update.

        Args:
            x_h (torch.Tensor): Token embeddings (B, T, D).
            base_update_force (torch.Tensor): Raw combined force (B, T, D).
            density_mod_factor (torch.Tensor): Density modulation factor (B, T, 1) or scalar 1.0.
            head_idx (int): Current head index.
            state_in (Optional[Dict]): Input state (for external mode).
            dynamic_params (Optional[Dict]): Dynamic hyperparameters.
            analysis_data (Optional[Dict]): Dictionary for analysis results.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict]:
                - x_out: Updated token embeddings (B, T, D).
                - aux_loss: Auxiliary loss from this module (usually 0).
                - state_out: Output state (for external mode).
        """
        B, T, D = x_h.shape
        force = base_update_force  # Start with the raw combined force
        state_out: Dict[str, torch.Tensor] = {}
        integ_analysis = {} if analysis_data is not None else None
        log_prefix = f"[H{head_idx}] Int - "  # Prefix for logs
        logger.debug(f"{log_prefix}Started. Force In Norm: {torch.linalg.vector_norm(force, dim=-1).mean().item():.4f}")

        # --- Conditional Computation Gating ---
        update_mask = torch.ones(B, T, 1, device=x_h.device, dtype=torch.bool)
        use_cond_comp = self.arch_config.get("use_conditional_computation", False)
        if use_cond_comp:
            with torch.no_grad():
                force_norm_sq = force.abs().pow(2).sum(dim=-1, keepdim=True)
                threshold = (
                    dynamic_params.get("conditional_comp_threshold", self.arch_config.get("conditional_comp_threshold", 0.01))
                    if dynamic_params
                    else self.arch_config.get("conditional_comp_threshold", 0.01)
                )
                threshold_sq = threshold**2
                update_mask = force_norm_sq > threshold_sq
                skip_ratio = 1.0 - update_mask.float().mean().item()
                # logger.debug(f"{log_prefix}CondComp: Skipping {skip_ratio*100:.2f}% (Thresh^2={threshold_sq:.4g})")
                if integ_analysis is not None:
                    integ_analysis["conditional_skip_ratio"] = skip_ratio
            force = force * update_mask.to(force.dtype)  # Apply mask

        # --- Token Momentum ---
        if self.use_token_momentum:
            token_mom_decay_val = (
                dynamic_params.get("token_momentum_decay", self.config["token_momentum_decay"])
                if dynamic_params
                else self.config["token_momentum_decay"]
            )
            token_mom_decay = torch.tensor(token_mom_decay_val, device=force.device, dtype=torch.float32)
            # logger.debug(f"{log_prefix}Token Momentum: Mode={self.state_mode}, Decay={token_mom_decay.item():.3f}")

            current_force_avg = force.mean(dim=(0, 1), keepdim=True)  # (1, 1, D) - Masked average
            last_force_state = None
            mom_state_key = f"token_momentum_h{head_idx}"

            if self.state_mode == "buffers":
                last_force_state = self.last_token_force_avg[head_idx : head_idx + 1]  # Keep dims (1, 1, D)
            elif self.state_mode == "external":
                last_force_state = state_in.get(mom_state_key) if state_in else None  # Expect (1,1,D) avg
                if last_force_state is None:
                    last_force_state = torch.zeros_like(current_force_avg)
                elif last_force_state.shape != current_force_avg.shape:  # Handle shape mismatch
                    logger.warning(
                        f"{log_prefix}Token momentum external state shape mismatch: Expected {current_force_avg.shape}, Got {last_force_state.shape}. Averaging."
                    )
                    last_force_state = last_force_state.mean(
                        dim=tuple(range(last_force_state.dim() - 1)), keepdim=True
                    )  # Avg all but last dim

            if last_force_state is not None:
                if integ_analysis:
                    integ_analysis["token_momentum_state_norm_in"] = torch.linalg.vector_norm(last_force_state).item()
                # Ensure calculation happens in float32 for stability if using AMP
                with torch.cuda.amp.autocast(enabled=False):
                    force_avg_with_mom = token_mom_decay * last_force_state.float() + current_force_avg.float()
                # Update state (buffer or output dict)
                if self.state_mode == "buffers":
                    self.last_token_force_avg[head_idx : head_idx + 1] = force_avg_with_mom.detach().to(self.dtype)
                elif self.state_mode == "external":
                    state_out[mom_state_key] = force_avg_with_mom.detach().to(self.dtype)
                # Apply momentum gain (approximate) - use float32 for division
                momentum_gain = force_avg_with_mom / current_force_avg.float().clamp(min=1e-8)
                force = force * momentum_gain.to(force.dtype)  # Apply gain, ensure dtype match
                if integ_analysis:
                    integ_analysis["force_norm_after_momentum"] = torch.linalg.vector_norm(force, dim=-1).mean().item()

        # --- Decay & Density ---
        use_learn_decay = (
            dynamic_params.get("use_learnable_decay", self.config.get("use_learnable_decay", True))
            if dynamic_params
            else self.config.get("use_learnable_decay", True)
        )
        decay_val: Union[float, torch.Tensor] = 1.0
        logits_param = self.effective_decay_logits  # Use property getter
        if use_learn_decay and logits_param is not None:
            decay_val = torch.sigmoid(logits_param[head_idx] if logits_param.dim() > 0 else logits_param)
        else:
            decay_val = (
                dynamic_params.get("fixed_decay_value", self.config.get("fixed_decay_value", 0.9))
                if dynamic_params
                else self.config.get("fixed_decay_value", 0.9)
            )
        if not isinstance(decay_val, torch.Tensor):
            decay_val = torch.tensor(decay_val)
        decay_val = decay_val.to(force.device).float()  # Ensure float tensor

        density_target = self.force_config.get("density_modulation_target", [])
        effective_decay = decay_val * density_mod_factor.float() if "decay" in density_target else decay_val
        scaled_force = effective_decay.to(force.dtype) * force  # Apply decay/density modulation
        if integ_analysis:
            integ_analysis["effective_decay_avg"] = effective_decay.mean().item()
            integ_analysis["force_norm_after_decay"] = torch.linalg.vector_norm(scaled_force, dim=-1).mean().item()

        # --- Gate ---
        gated_force = scaled_force
        use_gate = (
            dynamic_params.get("use_gate", self.config.get("use_gate", True))
            if dynamic_params
            else self.config.get("use_gate", True)
        )
        gate_layer_eff = self.effective_gate_layer  # Use property getter
        if use_gate and gate_layer_eff is not None:
            with torch.cuda.amp.autocast(enabled=False):  # Input to linear layer often better in float32
                gate_input = torch.cat([x_h.real, x_h.imag], dim=-1).float() if self.is_complex else x_h.float()
                try:
                    gate_val = torch.sigmoid(gate_layer_eff(gate_input))  # (B, T, 1) real
                    gated_force = gate_val.to(scaled_force.dtype) * scaled_force
                    if integ_analysis:
                        integ_analysis["gate_value_avg"] = gate_val.mean().item()
                        integ_analysis["force_norm_after_gate"] = torch.linalg.vector_norm(gated_force, dim=-1).mean().item()
                except Exception as e:
                    logger.error(f"{log_prefix}Gate layer failed: {e}", exc_info=True)
        elif use_gate:
            logger.warning(f"{log_prefix}Gate enabled but layer not found.")

        # --- Integration Steps on Gated Force ---
        processed_update = gated_force

        # --- Dropout ---
        if self.dropout is not None and self.training:
            processed_update = self.dropout(processed_update)
            if integ_analysis:
                integ_analysis["update_norm_after_dropout"] = torch.linalg.vector_norm(processed_update, dim=-1).mean().item()

        # --- Clipping (Before Norm) ---
        norm_before_clip = torch.linalg.vector_norm(processed_update, dim=-1, keepdim=True)
        if integ_analysis:
            integ_analysis["update_norm_before_clip"] = norm_before_clip.mean().item()
        if self.arch_config.get("use_gradient_clipping", False):
            clip_thr = self.arch_config.get("gradient_clip_threshold", 1.0)
            scale = (clip_thr / norm_before_clip.clamp(min=1e-6)).clamp(max=1.0)  # Real scale factor
            clipped_update = processed_update * scale.to(processed_update.dtype)  # Ensure scale dtype matches
            if integ_analysis:
                integ_analysis["clip_scale_avg"] = scale.mean().item()
                integ_analysis["update_norm_after_clip"] = torch.linalg.vector_norm(clipped_update, dim=-1).mean().item()
                if (scale < 0.999).any():
                    logger.debug(f"{log_prefix}Update clipped. Max Norm: {norm_before_clip.max().item():.3f} > {clip_thr}")
            processed_update = clipped_update

        # --- Normalization ---
        norm_before_norm_val = torch.linalg.vector_norm(processed_update, dim=-1, keepdim=True).mean().item()
        if integ_analysis and "update_norm_before_norm" not in integ_analysis:
            integ_analysis["update_norm_before_norm"] = norm_before_norm_val
        if self.norm_layer is not None:
            try:
                processed_update = self.norm_layer(processed_update)
                if integ_analysis:
                    integ_analysis["update_norm_after_norm"] = torch.linalg.vector_norm(processed_update, dim=-1).mean().item()
                # logger.debug(f"{log_prefix}Applied Normalization: {self.arch_config.get('normalization_type', 'rmsnorm')}")
            except Exception as e:
                logger.error(f"{log_prefix}Normalization failed: {e}", exc_info=True)
        # else: logger.debug(f"{log_prefix}Normalization skipped.")

        # --- Store State for Boids Alignment ---
        if self.needs_boids_state:
            boids_state_key = f"boids_last_update_h{head_idx}"  # Use consistent key
            if self.state_mode == "buffers":
                self.boids_last_processed_update_avg[head_idx] = processed_update.mean(dim=(0, 1), keepdim=True).detach()
            elif self.state_mode == "external":
                if self.boids_state_type == "full_update":
                    state_out[boids_state_key] = processed_update.detach()  # (B, T, D)
                else:  # avg_update
                    state_out[boids_state_key] = processed_update.mean(dim=(0, 1), keepdim=True).detach()  # (1, 1, D)

        # --- Residual Connection ---
        # Ensure processed_update has same dtype as x_h, esp after potential float32 intermediate steps
        x_out = x_h + processed_update.to(x_h.dtype)

        # --- Update main analysis dict ---
        if analysis_data is not None and integ_analysis is not None:
            analysis_data.update(integ_analysis)

        logger.debug(
            f"{log_prefix}Integration complete. Output norm avg: {torch.linalg.vector_norm(x_out, dim=-1).mean().item():.4f}"
        )
        return x_out, torch.tensor(0.0, device=x_h.device, dtype=torch.float32), state_out

    def extra_repr(self) -> str:
        norm_type = self.arch_config.get("normalization_type", "rmsnorm") if self.config.get("use_rms_norm") else "none"
        dropout_rate = self.dropout_rate if self.dropout else 0.0
        return f"state={self.state_mode}, momentum={self.use_token_momentum}, gate={hasattr(self, 'gate_layer') or hasattr(self, '_shared_gate_layer')}, norm={norm_type}, dropout={dropout_rate:.2f}"
