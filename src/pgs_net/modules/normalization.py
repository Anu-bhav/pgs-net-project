# src/pgs_net/modules/normalization.py
"""Normalization layer implementations."""

import logging
from typing import Optional

import torch
import torch.nn as nn

from .complex_utils import ComplexRMSNorm
from .interfaces import Normalization

logger = logging.getLogger(__name__)

# Define Real RMSNorm here if needed, or assume imported if torch has it
if not hasattr(nn, "RMSNorm"):  # Basic RMSNorm for real tensors

    class RMSNorm(Normalization):
        """Simple RMS Normalization for real tensors."""

        def __init__(self, d_model: int, eps: float = 1e-8):
            super().__init__()
            self.eps = eps
            self.scale = nn.Parameter(torch.ones(d_model))
            logger.debug(f"Initialized basic RMSNorm (d_model={d_model}, eps={eps})")

        def forward(self, x):
            rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            shape = [1] * (x.dim() - 1) + [-1]  # Shape for broadcasting scale
            return (x / rms) * self.scale.view(*shape)

        def extra_repr(self) -> str:
            return f"{self.scale.numel()}, eps={self.eps}"

else:
    # Use torch.nn.LayerNorm if RMSNorm isn't available/defined?
    # Or strictly require custom/torch>=2.X RMSNorm? Let's use custom one above.
    pass


class RMSNormImpl(Normalization):
    """Wrapper for RMSNorm / ComplexRMSNorm."""

    def __init__(self, d_model: int, eps: float = 1e-8, use_complex: bool = False):
        super().__init__()
        NormClass = ComplexRMSNorm if use_complex else RMSNorm  # Use our defined RMSNorm
        try:
            self.norm = NormClass(d_model, eps=eps)
            logger.info(f"Initialized RMSNormImpl (Complex={use_complex}, eps={eps})")
        except Exception as e:
            logger.error(f"Failed to initialize RMSNormImpl: {e}", exc_info=True)
            self.norm = nn.Identity()  # Fallback identity layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

    def extra_repr(self) -> str:
        return f"wrapped={self.norm.__class__.__name__}"


class AdaptiveGroupNorm(Normalization):
    """Basic Adaptive Group Normalization (AdaGN) for REAL inputs - Simplified using nn.GroupNorm."""

    def __init__(self, num_groups: int, d_model: int, eps: float = 1e-6, use_complex: bool = False):
        super().__init__()
        self.identity = None
        if use_complex:
            logger.warning("AdaGroupNorm for complex not implemented. Using Identity.")
            self.identity = nn.Identity()  # Fallback
        elif d_model % num_groups != 0:
            logger.error(f"AdaGroupNorm: d_model={d_model} not divisible by num_groups={num_groups}. Using Identity.")
            self.identity = nn.Identity()
        else:
            self.num_groups = num_groups
            # Use built-in GroupNorm with affine=True (learnable scale/shift per group)
            # Expects (N, C, *) - we have (B, T, D=C), transpose to (B, D, T)
            self.gn = nn.GroupNorm(num_groups=self.num_groups, num_channels=d_model, eps=eps, affine=True)
            logger.info(f"Initialized AdaptiveGroupNorm (Groups={num_groups}, Affine=True)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.identity is not None:
            return self.identity(x)
        # Input x: (B, T, D)
        if x.dim() != 3:
            logger.warning(f"AdaGroupNorm expected 3D input (B, T, D), got {x.dim()}D. Skipping.")
            return x
        B, T, D = x.shape
        # Reshape for GroupNorm: (B, C, L) = (B, D, T)
        x_reshaped = x.transpose(1, 2).contiguous()
        # Apply GroupNorm
        x_norm = self.gn(x_reshaped)
        # Reshape back: (B, T, D)
        return x_norm.transpose(1, 2)

    def extra_repr(self) -> str:
        if self.identity:
            return "Identity"
        else:
            return f"num_groups={self.num_groups}, num_channels={getattr(self.gn, 'num_channels', '?')}"
