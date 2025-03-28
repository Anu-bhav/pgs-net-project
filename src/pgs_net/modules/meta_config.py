# src/pgs_net/modules/meta_config.py
"""Meta-Learner implementations for dynamically adjusting hyperparameters."""

import logging
import math
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interfaces import MetaConfigLearner

logger = logging.getLogger(__name__)


class PlaceholderMetaConfig(MetaConfigLearner):
    """Placeholder: Returns empty dict (no dynamic hyperparameters)."""

    def __init__(self, d_model: int, config: Dict):
        super().__init__()
        logger.warning("Using PlaceholderMetaConfig (No dynamic hyperparameters).")

    def get_dynamic_config(self, current_state: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, float, int, bool]]:
        return {}

    def _prepare_context_vector(self, current_state: Dict[str, Any]) -> Optional[torch.Tensor]:
        return None  # No context needed

    def _get_device(self) -> torch.device:
        return torch.device("cpu")  # No parameters


class AdvancedHyperNetworkMetaConfig(MetaConfigLearner):
    """Uses a multi-layer MLP (HyperNetwork) with context embedding to predict hyperparameters."""

    def __init__(self, d_model: int, config: Dict[str, Any]):
        """
        Initializes the AdvancedHyperNetworkMetaConfig.

        Args:
            d_model (int): Model dimension (unused here, but kept for interface consistency).
            config (Dict[str, Any]): Full PGS_FFN configuration.

        """
        super().__init__()
        self.meta_config = config.get("meta_learning", {})
        self.target_params: List[str] = self.meta_config.get("meta_config_target_params", [])
        self.context_sources: List[str] = self.meta_config.get("meta_context_sources", [])
        self.hypernet_params: Dict[str, Any] = self.meta_config.get("hypernetwork_params", {})
        self.target_ranges: Dict[str, List[float]] = self.meta_config.get("meta_target_param_ranges", {})
        # Store global config needed for context preparation (e.g., max_clusters)
        self.global_config = config
        logger.info(f"Initialized AdvancedHyperNetworkMetaConfig. Targets: {self.target_params}, Context: {self.context_sources}")

        # --- Context Feature Dimensions & Embedding ---
        self.use_context_embedding = self.hypernet_params.get("use_context_embedding", True)
        self.context_embedding_dim = self.hypernet_params.get("context_embedding_dim", 16)
        # Define expected raw dimension for each context source
        self.context_feature_dims: Dict[str, int] = {
            "epoch_norm": 1,
            "avg_input_norm": 1,
            "loss_feedback": 1,
            "avg_assignment_entropy": 1,
            **self.hypernet_params.get("context_feature_dims", {}),  # Allow overrides from config
        }
        raw_context_dim = sum(self.context_feature_dims.get(src, 0) for src in self.context_sources)
        logger.debug(f"MetaConfig raw context dimension: {raw_context_dim}")

        self.context_embed_proj: Optional[nn.Linear] = None
        final_context_dim = 0
        if raw_context_dim > 0:
            if self.use_context_embedding:
                try:
                    self.context_embed_proj = nn.Linear(raw_context_dim, self.context_embedding_dim)
                    final_context_dim = self.context_embedding_dim
                    logger.info(f"Using context embedding: RawDim={raw_context_dim} -> EmbedDim={final_context_dim}")
                except Exception as e:
                    logger.error(f"Failed to init context embed proj: {e}. Using raw context.", exc_info=True)
                    final_context_dim = raw_context_dim  # Fallback to raw
                    self.use_context_embedding = False
            else:
                final_context_dim = raw_context_dim
                logger.info(f"Using raw context features: Dim={final_context_dim}")
        else:
            logger.warning("HyperNetworkMetaConfig has no context sources defined.")

        # --- HyperNetwork MLP ---
        self.hypernet: Optional[nn.Sequential] = None
        if final_context_dim > 0 and self.target_params:
            hidden_dims = self.hypernet_params.get("hidden_dims", [64, 32])
            use_ln = self.hypernet_params.get("use_layer_norm", True)
            activation_str = self.hypernet_params.get("activation", "relu").lower()
            if activation_str == "gelu":
                activation = nn.GELU
            elif activation_str == "silu" or activation_str == "swish":
                activation = nn.SiLU
            else:
                activation = nn.ReLU  # Default ReLU

            output_dim = len(self.target_params)
            logger.info(
                f"Building HyperNetwork MLP: Input={final_context_dim}, Hidden={hidden_dims}, Output={output_dim}, LN={use_ln}, Act={activation.__name__}"
            )

            try:
                layers = []
                in_d = final_context_dim
                for h_dim in hidden_dims:
                    layers.append(nn.Linear(in_d, h_dim))
                    if use_ln:
                        layers.append(nn.LayerNorm(h_dim))
                    layers.append(activation())
                    in_d = h_dim
                layers.append(nn.Linear(in_d, output_dim))  # Final output layer (raw values)
                self.hypernet = nn.Sequential(*layers)
            except Exception as e:
                logger.error(f"Failed to build HyperNetwork MLP: {e}", exc_info=True)
                self.hypernet = None  # Disable if build fails
        else:
            logger.warning("HyperNetworkMetaConfig is inactive (no context or targets).")

        self.output_activation_type = self.hypernet_params.get("output_activation", "scaled_sigmoid_softplus")

    def _prepare_context_vector(self, current_state: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Constructs the raw context vector from available state info."""
        vec_parts = []
        dev = self._get_device()

        if "epoch_norm" in self.context_sources:
            max_epochs = current_state.get("max_epochs", 100)
            epoch = current_state.get("epoch", 0)
            norm_epoch = min(max(0.0, epoch / max(1.0, max_epochs)), 1.0) if max_epochs is not None else 0.0
            vec_parts.append(torch.tensor([norm_epoch], device=dev))

        if "avg_input_norm" in self.context_sources:
            norm_val = current_state.get("avg_input_norm", 1.0)
            # Use log1p for stability, clamp input?
            vec_parts.append(torch.tensor([math.log1p(max(0, norm_val))], device=dev))

        if "loss_feedback" in self.context_sources:
            loss_val = current_state.get("loss_feedback", 0.0)
            # Use log1p? Maybe clamp? Raw loss can vary hugely.
            vec_parts.append(torch.tensor([math.log1p(max(0, loss_val))], device=dev))

        if "avg_assignment_entropy" in self.context_sources:
            max_k = self.global_config.get("clustering", {}).get("max_clusters", 4)
            max_entropy = math.log(max_k) if max_k > 1 else 1.0
            entropy = current_state.get("avg_assignment_entropy", 0.0)
            # Normalize entropy to [0, 1]
            norm_entropy = max(0.0, min(1.0, entropy / max(1e-6, max_entropy)))
            vec_parts.append(torch.tensor([norm_entropy], device=dev))

        # Add other context sources here...

        if not vec_parts:
            return None
        try:
            # Ensure all parts are tensors before cat
            valid_parts = [p for p in vec_parts if isinstance(p, torch.Tensor)]
            if not valid_parts:
                return None
            return torch.cat(valid_parts).float()  # Shape (RawContextDim,)
        except Exception as e:
            logger.error(f"Failed to concatenate context vector parts: {e}")
            return None

    def get_dynamic_config(self, current_state: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, float, int, bool]]:
        """Generates dynamic hyperparameters using the hypernetwork."""
        if self.hypernet is None:
            return {}

        raw_context_vec = self._prepare_context_vector(current_state)
        if raw_context_vec is None:
            # logger.warning("MetaConfig: Could not prepare context vector.") # Can be verbose
            return {}

        # Embed context if configured
        context_input = raw_context_vec.unsqueeze(0)  # Add batch dim
        if self.use_context_embedding and hasattr(self, "context_embed_proj") and self.context_embed_proj is not None:
            try:
                context_input = self.context_embed_proj(context_input)
            except Exception as e:
                logger.error(f"Context embedding projection failed: {e}")
                return {}

        # Predict raw outputs
        try:
            # Ensure hypernet input matches dtype (should be float)
            raw_outputs = self.hypernet(context_input.float()).squeeze(0)  # Shape (NumTargets,)
        except Exception as e:
            logger.error(f"HyperNetwork forward pass failed: {e}", exc_info=True)
            return {}

        # --- Map raw outputs to target parameter ranges ---
        dynamic_params: Dict[str, torch.Tensor] = {}
        activation_type = self.output_activation_type

        for i, param_name in enumerate(self.target_params):
            if i >= len(raw_outputs):  # Safety check
                logger.error(f"Hypernetwork output dimension mismatch for parameter '{param_name}'.")
                continue
            raw_val = raw_outputs[i]
            target_range = self.target_ranges.get(param_name)

            try:
                if activation_type == "direct":
                    final_val = raw_val  # Use raw output
                else:  # Default: 'scaled_sigmoid_softplus'
                    if target_range is not None and len(target_range) == 2:
                        min_val, max_val = target_range
                        if min_val >= max_val:
                            logger.warning(f"Invalid range for {param_name}: {target_range}. Using Sigmoid.")
                            final_val = torch.sigmoid(raw_val)
                        else:
                            final_val = min_val + (max_val - min_val) * torch.sigmoid(raw_val)  # Scale sigmoid output
                    elif "temp" in param_name or "strength" in param_name or "_weight" in param_name:
                        final_val = F.softplus(raw_val) + 1e-6  # Ensure positive > 0
                    else:  # Fallback to sigmoid for [0, 1] range (e.g., decays, probs)
                        final_val = torch.sigmoid(raw_val)

                dynamic_params[param_name] = final_val
                # logger.debug(f"MetaConfig - Param '{param_name}': Raw={raw_val.item():.3f} -> Final={final_val.item():.3f}")
            except Exception as e:
                logger.error(f"Error processing hyperparameter '{param_name}': {e}")

        return dynamic_params

    def _get_device(self) -> torch.device:
        """Helper to get device of the hypernet parameters."""
        try:
            # Check if hypernet exists and has parameters
            if self.hypernet and len(list(self.parameters())) > 0:
                return next(self.parameters()).device
        except StopIteration:
            pass
        return torch.device("cpu")  # Fallback device

    # Need to inject global config during init if needed for context prep (e.g., max_clusters)
    def set_global_config(self, global_config: Dict):
        self.global_config = global_config

    def extra_repr(self) -> str:
        return f"targets={len(self.target_params)}, context={self.context_sources}, active={self.hypernet is not None}"
