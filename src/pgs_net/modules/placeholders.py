# src/pgs_net/modules/placeholders.py
"""Placeholder implementations for advanced/research components."""

import torch
import torch.nn as nn
import logging
from .interfaces import DynamicKController, FormalForceCalculator, MetaConfigLearner
from typing import Dict, Any, Optional, Tuple, List, Union

logger = logging.getLogger(__name__)


class PlaceholderDynamicKController(DynamicKController):
    """Placeholder: Always uses max_k clusters."""

    def __init__(self, max_k: int, params: Dict, state_mode: str, device: torch.device, dtype: torch.dtype):
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
        pass  # No-op

    def get_active_indices(self) -> torch.Tensor:
        # Returns indices from 0 to max_k - 1
        return self.active_indices_cache

    def _update_cache(self) -> None:
        self.active_indices_cache = torch.arange(self.max_k, device=self.device)


class PlaceholderFormalForce(FormalForceCalculator):
    """Placeholder: Returns zero force."""

    def __init__(self, config: Dict, is_complex: bool, dtype: torch.dtype):
        super().__init__()
        logger.warning("Using PlaceholderFormalForce (Returns zero force).")

    def calculate_force(
        self,
        x_h: torch.Tensor,
        A: torch.Tensor,
        local_queens: torch.Tensor,
        global_queen: torch.Tensor,
        centroids_h: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(x_h)


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
        # This module has no parameters, return CPU device as default
        return torch.device("cpu")


# Add stubs for other components if needed, e.g., advanced normalization/regularization
class StubNormalization(nn.Module):  # Not inheriting ABC to avoid abstract methods
    def __init__(self, *args, **kwargs):
        super().__init__()
        logger.warning("Using StubNormalization (Identity function).")

    def forward(self, x):
        return x


class StubRegularization(nn.Module):  # Not inheriting ABC
    def __init__(self, *args, **kwargs):
        super().__init__()
        logger.warning("Using StubRegularization (Returns zero loss).")

    def forward(self, *args, **kwargs):
        return torch.tensor(0.0)

    @torch.no_grad()
    def apply_constraints(self, module):
        pass
