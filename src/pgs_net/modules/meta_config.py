# src/pgs_net/modules/meta_config.py
""" Meta-Learner implementations. """
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Dict, Any, Optional, List, Union
from .interfaces import MetaConfigLearner

logger = logging.getLogger(__name__)

class PlaceholderMetaConfig(MetaConfigLearner):
     # ... (Implementation as before) ...

class AdvancedHyperNetworkMetaConfig(MetaConfigLearner):
    """ Uses a multi-layer MLP with context embedding to predict hyperparameters. """
    def __init__(self, d_model: int, config: Dict[str, Any]):
        """
        Initializes the AdvancedHyperNetworkMetaConfig.

        Args:
            d_model (int): Model dimension (used for potential context projections).
            config (Dict[str, Any]): Full PGS_FFN configuration.
        """
        super().__init__()
        self.meta_config = config.get('meta_learning', {})
        self.target_params: List[str] = self.meta_config.get('meta_config_target_params', [])
        self.context_sources: List[str] = self.meta_config.get('meta_context_sources', [])
        self.hypernet_params: Dict[str, Any] = self.meta_config.get('hypernetwork_params', {})
        self.target_ranges: Dict[str, List[float]] = self.meta_config.get('meta_target_param_ranges', {})
        logger.info(f"Initialized AdvancedHyperNetworkMetaConfig. Targets: {self.target_params}, Context: {self.context_sources}")

        # --- Context Feature Dimensions & Embedding ---
        self.use_context_embedding = self.hypernet_params.get('use_context_embedding', True)
        self.context_embedding_dim = self.hypernet_params.get('context_embedding_dim', 16)
        # Define expected raw dimension for each context source
        self.context_feature_dims: Dict[str, int] = {
            'epoch_norm': 1, 'avg_input_norm': 1, 'loss_feedback': 1, 'avg_assignment_entropy': 1,
            **self.hypernet_params.get('context_feature_dims', {}) # Allow overrides
        }
        raw_context_dim = sum(self.context_feature_dims.get(src, 0) for src in self.context_sources)

        self.context_embed_proj: Optional[nn.Linear] = None
        final_context_dim = 0
        if raw_context_dim > 0:
            if self.use_context_embedding:
                 self.context_embed_proj = nn.Linear(raw_context_dim, self.context_embedding_dim)
                 final_context_dim = self.context_embedding_dim
                 logger.info(f"Using context embedding: RawDim={raw_context_dim} -> EmbedDim={final_context_dim}")
            else:
                 final_context_dim = raw_context_dim
                 logger.info(f"Using raw context features: Dim={final_context_dim}")
        else: logger.warning("HyperNetworkMetaConfig has no context sources defined.")


        # --- HyperNetwork MLP ---
        self.hypernet: Optional[nn.Sequential] = None
        if final_context_dim > 0 and self.target_params:
            hidden_dims = self.hypernet_params.get('hidden_dims', [64, 32])
            use_ln = self.hypernet_params.get('use_layer_norm', True)
            activation = nn.ReLU # Could be configurable
            output_dim = len(self.target_params)

            layers = []
            in_d = final_context_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(in_d, h_dim))
                if use_ln: layers.append(nn.LayerNorm(h_dim))
                layers.append(activation())
                in_d = h_dim
            layers.append(nn.Linear(in_d, output_dim)) # Final output layer (raw values)
            self.hypernet = nn.Sequential(*layers)
            logger.info(f"HyperNetwork MLP initialized: {final_context_dim} -> {hidden_dims} -> {output_dim}")
        else:
            logger.warning("HyperNetworkMetaConfig is inactive (no context or targets).")

        self.output_activation_type = self.hypernet_params.get('output_activation', 'scaled_sigmoid_softplus')

    def _prepare_context_vector(self, current_state: Dict[str, Any]) -> Optional[torch.Tensor]:
         """ Constructs the raw context vector from available state info. """
         vec_parts = []
         dev = self._get_device()

         if 'epoch_norm' in self.context_sources:
              max_epochs = current_state.get('max_epochs', 100) # Need max_epochs passed in state
              norm_epoch = min(current_state.get('epoch', 0) / max(1.0, max_epochs), 1.0)
              vec_parts.append(torch.tensor([norm_epoch], device=dev))
         if 'avg_input_norm' in self.context_sources:
              # Normalize input norm? Depends on expected range. Clamp for stability?
              norm_val = current_state.get('avg_input_norm', 1.0)
              vec_parts.append(torch.tensor([math.log1p(norm_val)], device=dev)) # Log transform might stabilize
         if 'loss_feedback' in self.context_sources:
              # Use log loss? Smoothed loss? Clamp?
              loss_val = current_state.get('loss_feedback', 0.0)
              vec_parts.append(torch.tensor([math.log1p(loss_val)], device=dev)) # Log transform
         if 'avg_assignment_entropy' in self.context_sources:
              max_k = self.global_config.get('clustering', {}).get('max_clusters', 4) # Need access to global config here
              max_entropy = math.log(max_k) if max_k > 1 else 1.0
              entropy = current_state.get('avg_assignment_entropy', 0.0)
              vec_parts.append(torch.tensor([entropy / max(1e-6, max_entropy)]).clamp(0.0, 1.0))

         if not vec_parts: return None
         # Ensure all parts are tensors before cat
         valid_parts = [p for p in vec_parts if isinstance(p, torch.Tensor)]
         if not valid_parts: return None
         return torch.cat(valid_parts).float() # Shape (RawContextDim,)

    def get_dynamic_config(self, current_state: Dict[str, Any]) -> Dict[str, Union[torch.Tensor, float, int, bool]]:
        """ Generates dynamic hyperparameters using the hypernetwork. """
        if self.hypernet is None: return {}

        raw_context_vec = self._prepare_context_vector(current_state)
        if raw_context_vec is None:
            logger.warning("MetaConfig: Could not prepare context vector.")
            return {}

        # Embed context if configured
        if self.use_context_embedding and hasattr(self, 'context_embed_proj'):
             context_input = self.context_embed_proj(raw_context_vec).unsqueeze(0) # Add batch dim
        else:
             context_input = raw_context_vec.unsqueeze(0) # Add batch dim

        # Predict raw outputs
        try:
            raw_outputs = self.hypernet(context_input).squeeze(0) # Shape (NumTargets,)
        except Exception as e:
             logger.error(f"HyperNetwork forward pass failed: {e}")
             return {}

        # Map raw outputs to target parameter ranges
        dynamic_params: Dict[str, torch.Tensor] = {}
        for i, param_name in enumerate(self.target_params):
            raw_val = raw_outputs[i]
            target_range = self.target_ranges.get(param_name)

            if self.output_activation_type == 'direct':
                 final_val = raw_val
            else: # Default: 'scaled_sigmoid_softplus'
                 if target_range is not None and len(target_range) == 2:
                      min_val, max_val = target_range
                      if min_val >= max_val: logger.warning(f"Invalid range for {param_name}: {target_range}"); continue
                      final_val = min_val + (max_val - min_val) * torch.sigmoid(raw_val)
                 elif 'temp' in param_name or 'strength' in param_name or '_weight' in param_name:
                      # Use softplus for positive values, maybe with offset/scaling?
                      final_val = F.softplus(raw_val) + 1e-6 # Ensure strictly positive
                 else: # Fallback to sigmoid for [0, 1] range (e.g., decays, probs)
                      final_val = torch.sigmoid(raw_val)

            dynamic_params[param_name] = final_val
            logger.debug(f"MetaConfig - Param '{param_name}': Raw={raw_val.item():.3f} -> Final={final_val.item():.3f}")

        # Return dict of tensors (allows gradients to flow back to hypernet)
        return dynamic_params

    def _get_device(self) -> torch.device:
        """ Helper to get device of the hypernet parameters. """
        try:
            return next(self.parameters()).device
        except StopIteration: # No parameters in hypernet
            return torch.device('cpu')

    # Inject global config on init if needed for context prep (e.g., max_clusters)
    def set_global_config(self, global_config: Dict):
        self.global_config = global_config