# src/pgs_net/modules/complex_utils.py
"""Utilities for complex number operations in PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ComplexLinear(nn.Module):
    """Linear layer for complex numbers: y = Wx + b, where W, x, y, b are complex."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initializes the ComplexLinear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool): If True, adds a learnable complex bias to the output.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weights for real and imaginary parts of the complex weight matrix W = W_re + i*W_im
        self.weight_real = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            # Complex bias b = b_re + i*b_im
            self.bias = nn.Parameter(torch.Tensor(out_features, dtype=torch.complex64))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes weights and bias using Kaiming uniform and zero respectively."""
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))
        if self.bias is not None:
            self.bias.data.zero_()  # Initialize complex bias to zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the complex linear transformation. Handles real inputs by promoting them.

        Args:
            x (torch.Tensor): Input tensor, shape (..., in_features). Can be real or complex.

        Returns:
            torch.Tensor: Output complex tensor, shape (..., out_features).
        """
        if not x.is_complex():
            # logger.debug("Input to ComplexLinear is not complex. Treating as real.")
            x = torch.complex(x, torch.zeros_like(x))

        # Linear transformation: (W_re + i*W_im) * (x_re + i*x_im)
        # = (W_re*x_re - W_im*x_im) + i*(W_re*x_im + W_im*x_re)
        out_real = F.linear(x.real, self.weight_real) - F.linear(x.imag, self.weight_imag)
        out_imag = F.linear(x.real, self.weight_imag) + F.linear(x.imag, self.weight_real)
        output = torch.complex(out_real, out_imag)

        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class ComplexRMSNorm(nn.Module):
    """RMSNorm adapted for complex numbers. Normalizes by RMS of magnitude."""

    def __init__(self, d_model: int, eps: float = 1e-8):
        """
        Initializes the ComplexRMSNorm layer.

        Args:
            d_model (int): The feature dimension to normalize over (last dimension).
            eps (float): Epsilon value for numerical stability.
        """
        super().__init__()
        # Scale is applied to the magnitude, so it's real
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies complex RMSNorm. Handles real inputs as a fallback.

        Args:
            x (torch.Tensor): Input tensor, shape (..., d_model). Can be real or complex.

        Returns:
            torch.Tensor: Normalized output tensor, same shape and dtype as input.
        """
        if not x.is_complex():
            # Fallback for real inputs
            rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
            x_norm = x / (rms + self.eps)
            # Correct scale application (needs broadcasting dims)
            scale_shape = [1] * (x.dim() - 1) + [-1]
            return self.scale.view(*scale_shape) * x_norm

        # Calculate RMS based on magnitude squared: E[|z|^2] = E[z * z.conj()]
        # Ensure stability by adding eps inside sqrt if needed, though adding after is common.
        mean_mag_sq = (x * x.conj()).real.mean(dim=-1, keepdim=True)
        rms = mean_mag_sq.sqrt()
        x_norm = x / (rms + self.eps)

        # Apply real scale (broadcasts correctly)
        scale_shape = [1] * (x.dim() - 1) + [-1]
        return self.scale.view(*scale_shape) * x_norm

    def extra_repr(self) -> str:
        return f"d_model={self.scale.numel()}, eps={self.eps}"
