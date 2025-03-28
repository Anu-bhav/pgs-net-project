# src/pgs_net/modules/adapters.py
"""Input and Output Adapters for handling complex representations."""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any
from .complex_utils import ComplexLinear

logger = logging.getLogger(__name__)

# --- Projection/Output Functions ---


def complex_project(tensor: torch.Tensor, method: str = "linear", proj_layer: Optional[nn.Module] = None) -> torch.Tensor:
    """
    Projects a real tensor to complex using the specified method.

    Args:
        tensor (torch.Tensor): Input real tensor.
        method (str): Projection method ('real_as_complex', 'linear').
        proj_layer (Optional[nn.Module]): Linear layer required if method='linear'.
                                         Should output 2*input_dim features.

    Returns:
        torch.Tensor: Output complex tensor.
    """
    if tensor.is_complex():
        return tensor  # Already complex

    method_options = ["real_as_complex", "linear"]
    if method not in method_options:
        logger.warning(
            f"Complex projection method '{method}' not recognized. Valid options: {method_options}. Defaulting to real_as_complex."
        )
        method = "real_as_complex"

    if method == "real_as_complex":
        return torch.complex(tensor, torch.zeros_like(tensor))
    elif method == "linear":
        if proj_layer is not None:
            projected = proj_layer(tensor)
            try:
                # Split the last dimension into two halves for real and imaginary parts
                real_part, imag_part = torch.chunk(projected, 2, dim=-1)
                return torch.complex(real_part, imag_part)
            except Exception as e:
                logger.error(f"Complex projection failed for linear method (check proj_layer output dim): {e}")
                return torch.complex(tensor, torch.zeros_like(tensor))  # Fallback
        else:
            logger.error("Complex projection method 'linear' requires proj_layer, but none provided. Falling back.")
            return torch.complex(tensor, torch.zeros_like(tensor))
    else:  # Should not happen due to validation above
        return torch.complex(tensor, torch.zeros_like(tensor))


def complex_output(tensor: torch.Tensor, method: str = "linear", output_layer: Optional[nn.Module] = None) -> torch.Tensor:
    """
    Gets a real output from a complex tensor using the specified method.

    Args:
        tensor (torch.Tensor): Input complex tensor.
        method (str): Output method ('real_part', 'magnitude', 'linear').
        output_layer (Optional[nn.Module]): Linear layer required if method='linear'.
                                           Should take 2*input_dim_complex features.

    Returns:
        torch.Tensor: Output real tensor.
    """
    if not tensor.is_complex():
        # logger.debug("Input to complex_output is already real.")
        return tensor  # Already real

    method_options = ["real_part", "magnitude", "linear"]
    if method not in method_options:
        logger.warning(
            f"Complex output method '{method}' not recognized. Valid options: {method_options}. Defaulting to real part."
        )
        method = "real_part"

    if method == "real_part":
        return tensor.real
    elif method == "magnitude":
        return tensor.abs()
    elif method == "linear":
        if output_layer is not None:
            # Assumes output_layer is nn.Linear taking 2*dim input
            try:
                # Concatenate real and imaginary parts along the feature dimension
                real_imag = torch.cat([tensor.real, tensor.imag], dim=-1)
                return output_layer(real_imag)
            except Exception as e:
                logger.error(f"Complex output failed for linear method: {e}")
                return tensor.real  # Fallback
        else:
            logger.error("Complex output method 'linear' requires output_layer, but none provided. Falling back.")
            return tensor.real
    else:  # Should not happen
        return tensor.real


# --- Adapter Modules ---


class InputAdapter(nn.Module):
    """Projects real input tensor to complex if configured."""

    def __init__(self, d_head: int, config: Dict[str, Any]):
        """
        Initializes the InputAdapter.

        Args:
            d_head (int): The dimension of the input features per head.
            config (Dict[str, Any]): The main PGS_FFN configuration dictionary.
        """
        super().__init__()
        self.config = config.get("architecture", {})  # Safely get sub-config
        self.use_complex = self.config.get("use_complex_representation", False)
        self.method = self.config.get("complex_projection_method", "real_as_complex")
        self.proj_layer: Optional[nn.Linear] = None
        logger.debug(f"InputAdapter: Complex={self.use_complex}, Method={self.method}")
        if self.use_complex and self.method == "linear":
            # Input: d_head real, Output: d_head complex (requires 2*d_head output)
            try:
                self.proj_layer = nn.Linear(d_head, 2 * d_head)
            except Exception as e:
                logger.error(f"Failed to initialize projection layer in InputAdapter: {e}")
                self.method = "real_as_complex"  # Fallback if layer creation fails

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies projection if complex mode is enabled."""
        if self.use_complex:
            # logger.debug(f"InputAdapter projecting to complex using method '{self.method}'.")
            return complex_project(x, self.method, self.proj_layer)
        else:
            # logger.debug("InputAdapter passing through real input.")
            return x

    def extra_repr(self) -> str:
        return f"use_complex={self.use_complex}, method={self.method}"


class OutputAdapter(nn.Module):
    """Projects complex output tensor back to real if configured."""

    def __init__(self, d_head: int, config: Dict[str, Any]):
        """
        Initializes the OutputAdapter.

        Args:
            d_head (int): The dimension of the potentially complex input features per head.
            config (Dict[str, Any]): The main PGS_FFN configuration dictionary.
        """
        super().__init__()
        self.config = config.get("architecture", {})
        self.use_complex_input = self.config.get("use_complex_representation", False)  # If the pipeline is complex
        self.method = self.config.get("complex_output_method", "real_part")
        self.output_layer: Optional[nn.Linear] = None
        logger.debug(f"OutputAdapter: ComplexInputExpected={self.use_complex_input}, Method={self.method}")
        if self.use_complex_input and self.method == "linear":
            # Input: d_head complex (takes 2*d_head real/imag), Output: d_head real
            try:
                self.output_layer = nn.Linear(2 * d_head, d_head)
            except Exception as e:
                logger.error(f"Failed to initialize output layer in OutputAdapter: {e}")
                self.method = "real_part"  # Fallback if layer creation fails

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies projection if input is complex."""
        # The check should be if x *is* complex, not based on config alone,
        # as AMP might change intermediate types.
        if x.is_complex():
            # logger.debug(f"OutputAdapter projecting complex input to real using method '{self.method}'.")
            return complex_output(x, self.method, self.output_layer)
        else:
            # logger.debug("OutputAdapter passing through real input.")
            return x

    def extra_repr(self) -> str:
        return f"method={self.method}"
