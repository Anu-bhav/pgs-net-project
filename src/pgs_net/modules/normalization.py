# src/pgs_net/modules/normalization.py
"""Normalization layer implementations."""

import torch
import torch.nn as nn
import logging
from typing import Optional
from .interfaces import Normalization
from .complex_utils import ComplexRMSNorm

logger = logging.getLogger(__name__)


class RMSNormImpl(Normalization):
    """Wrapper for existing RMSNorm / ComplexRMSNorm."""

    def __init__(self, d_model: int, eps: float = 1e-8, use_complex: bool = False):
        super().__init__()
        NormClass = ComplexRMSNorm if use_complex else RMSNorm
        try:
            self.norm = NormClass(d_model, eps=eps)
            logger.info(f"Initialized RMSNormImpl (Complex={use_complex}, eps={eps})")
        except Exception as e:
            logger.error(f"Failed to initialize RMSNormImpl: {e}")
            # Provide a fallback identity layer
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class AdaptiveGroupNorm(Normalization):
    """Basic Adaptive Group Normalization (AdaGN) for REAL inputs - Simplified."""

    def __init__(self, num_groups: int, d_model: int, eps: float = 1e-6, use_complex: bool = False):
        super().__init__()
        if use_complex:
            logger.warning("AdaGroupNorm for complex not implemented. Using Identity.")
            self.identity = nn.Identity()  # Fallback
        elif d_model % num_groups != 0:
            logger.error(f"AdaGroupNorm: d_model={d_model} not divisible by num_groups={num_groups}. Using Identity.")
            self.identity = nn.Identity()
        else:
            self.num_groups = num_groups
            self.group_dim = d_model // num_groups
            # GroupNorm applied on C dimension, expect N, C, *
            # Input is (B, T, D=C). Treat T as spatial dim? (B, D, T)
            self.gn = nn.GroupNorm(
                num_groups=self.num_groups, num_channels=d_model, eps=eps, affine=True
            )  # Use built-in GN with affine
            logger.info(f"Initialized AdaptiveGroupNorm (Groups={num_groups}, Affine=True)")
            self.identity = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.identity is not None:
            return self.identity(x)
        # Input x: (B, T, D)
        B, T, D = x.shape
        # Reshape for GroupNorm: (B, C, L) = (B, D, T)
        x_reshaped = x.transpose(1, 2).contiguous()
        # Apply GroupNorm
        x_norm = self.gn(x_reshaped)
        # Reshape back: (B, T, D)
        return x_norm.transpose(1, 2)


# Define RMSNorm here if not in complex_utils
if not hasattr(nn, "RMSNorm"):  # Basic RMSNorm for real tensors if needed

    class RMSNorm(Normalization):
        def __init__(self, d_model: int, eps: float = 1e-8):
            super().__init__()
            self.eps = eps
            self.scale = nn.Parameter(torch.ones(d_model))

        def forward(self, x):
            rms = x.pow(2).mean(-1, keepdim=True).sqrt()
            shape = [1] * (x.dim() - 1) + [-1]
            return (x / (rms + self.eps)) * self.scale.view(*shape)
