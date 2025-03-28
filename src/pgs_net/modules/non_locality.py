# src/pgs_net/modules/non_locality.py
"""Optional Non-Locality Module (e.g., Attention)."""

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class NonLocalityModule(nn.Module):
    """Placeholder module for adding non-local interactions, e.g., via MHA."""

    def __init__(self, d_model: int, config: Dict[str, Any]):
        """
        Initializes the NonLocalityModule based on configuration.

        Args:
            d_model (int): The input/output dimension of the module.
            config (Dict[str, Any]): The main PGS_FFN configuration dictionary.

        """
        super().__init__()
        self.arch_config = config.get("architecture", {})
        self.config = self.arch_config.get("non_locality_params", {})
        self.nl_type = self.arch_config.get("non_locality_type", "none")
        self.is_complex = self.arch_config.get("use_complex_representation", False)  # Check if main pipeline is complex
        logger.info(f"Initializing NonLocalityModule: Type={self.nl_type}")

        self.module = None
        if self.nl_type == "mha":
            embed_dim = self.config.get("embed_dim", d_model)
            num_heads = self.config.get("num_heads", 4)
            dropout = self.config.get("dropout", 0.1)
            if embed_dim != d_model:
                logger.warning(f"NonLocality MHA embed_dim {embed_dim} != d_model {d_model}. Ensure projections match.")
            if self.is_complex:
                logger.warning(
                    "NonLocality MHA does not natively support complex numbers. Operating on real part or magnitude might be needed."
                )
            try:
                self.module = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
                self.norm = nn.LayerNorm(embed_dim)  # Add norm for stability
                logger.info(f"Using MHA for Non-locality: embed_dim={embed_dim}, heads={num_heads}")
            except Exception as e:
                logger.error(f"Failed to init NonLocality MHA: {e}")
                self.module = None
        elif self.nl_type == "linear_attn_stub":
            logger.warning("Linear Attention non-locality stub.")
            self.module = nn.Identity()  # Placeholder
        elif self.nl_type == "gated_attn_stub":
            logger.warning("Gated Attention non-locality stub.")
            self.module = nn.Identity()  # Placeholder
        elif self.nl_type != "none":
            logger.warning(f"Unknown non-locality type '{self.nl_type}'. Module inactive.")
            self.module = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the non-local operation.

        Args:
            x (torch.Tensor): Input tensor, shape (B, T, D). Assumed REAL for MHA.

        Returns:
            torch.Tensor: Output tensor, same shape as input.

        """
        if self.module is not None and self.nl_type == "mha":
            # Assume input x is REAL (or handle complex appropriately before MHA)
            if x.is_complex():
                logger.warning("NonLocality MHA receiving complex input, using real part.")
                x_real = x.real
            else:
                x_real = x

            try:
                # Apply MHA (Query=Key=Value = x_real)
                attn_output, _ = self.module(x_real, x_real, x_real)
                # Add residual connection and norm
                x_out_real = self.norm(x_real + attn_output)

                # Reconstruct complex if necessary (simple recombination)
                if x.is_complex():
                    return torch.complex(x_out_real, x.imag)
                else:
                    return x_out_real
            except Exception as e:
                logger.error(f"NonLocality MHA forward failed: {e}", exc_info=True)
                return x  # Return original input on error
        elif isinstance(self.module, nn.Identity):  # Pass through for stubs
            return x
        else:  # Inactive module
            return x

    def extra_repr(self) -> str:
        return f"type={self.nl_type}, active={self.module is not None and not isinstance(self.module, nn.Identity)}"
