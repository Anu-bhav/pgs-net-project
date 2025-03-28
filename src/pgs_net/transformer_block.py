# src/pgs_net/transformer_block.py
"""Transformer Block using PGS_FFN."""

import logging
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEFAULT_PGS_FFN_CONFIG
from .pgs_ffn import PGS_FFN

logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """
    Standard Transformer Block using PGS_FFN as the feed-forward layer.
    Follows the Post-Normalization structure: Sublayer -> Dropout -> Add -> Norm
    """

    def __init__(self, d_model: int, n_heads_attn: int, pgs_ffn_config: Optional[Dict] = None, dropout: float = 0.1):
        """
        Initializes the TransformerBlock.

        Args:
            d_model (int): Dimension of the model.
            n_heads_attn (int): Number of heads for the self-attention mechanism.
            pgs_ffn_config (Optional[Dict]): Configuration dictionary for the PGS_FFN layer.
                                              Uses default if None.
            dropout (float): Dropout rate for sub-layer outputs.

        """
        super().__init__()
        if d_model <= 0 or n_heads_attn <= 0:
            raise ValueError("d_model and n_heads_attn must be positive integers.")
        if d_model % n_heads_attn != 0:
            logger.warning(f"d_model ({d_model}) is not perfectly divisible by n_heads_attn ({n_heads_attn}).")

        logger.info(f"Initializing TransformerBlock: d_model={d_model}, attn_heads={n_heads_attn}")
        self.d_model = d_model
        self.n_heads_attn = n_heads_attn

        # --- Self-Attention ---
        # Using batch_first=True convention for consistency
        try:
            self.self_attn = nn.MultiheadAttention(d_model, n_heads_attn, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
        except Exception as e:
            logger.error(f"Failed to initialize Self-Attention block: {e}", exc_info=True)
            raise

        # --- PGS Feed-Forward Network ---
        if pgs_ffn_config is None:
            logger.warning("No PGS_FFN config provided to TransformerBlock, using default.")
            pgs_ffn_config = deepcopy(DEFAULT_PGS_FFN_CONFIG)  # Use a copy of default
        else:
            pgs_ffn_config = deepcopy(pgs_ffn_config)  # Ensure passed config isn't modified

        # Determine num_heads for PGS_FFN (Can be different from attn_heads)
        # Read from config or default within PGS_FFN init
        pgs_num_heads = pgs_ffn_config.get("architecture", {}).get("num_heads", 4)  # Example default
        if d_model % pgs_num_heads != 0:
            logger.warning(
                f"PGS FFN heads ({pgs_num_heads}) doesn't divide d_model ({d_model}). PGS_FFN might adjust head count."
            )

        try:
            self.pgs_ffn = PGS_FFN(d_model, num_heads=pgs_num_heads, config=pgs_ffn_config)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)
        except Exception as e:
            logger.error(f"Failed to initialize PGS_FFN block: {e}", exc_info=True)
            raise

        # Store state management mode for convenience
        self.state_mode = self.pgs_ffn.config.get("architecture", {}).get("state_management", {}).get("mode", "buffers")
        self.collect_analysis = self.pgs_ffn.config.get("architecture", {}).get("collect_analysis_data", False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        pgs_state_in: Optional[Dict] = None,
        epoch: Optional[int] = None,
        loss_feedback: Optional[torch.Tensor] = None,  # Pass loss feedback down
        cross_layer_input: Optional[Dict] = None,  # Pass cross-layer info down
        max_epochs: Optional[int] = None,  # Pass max_epochs down
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict], Optional[Dict], Optional[Dict]]:
        """
        Forward pass for the Transformer Block.

        Args:
            x (torch.Tensor): Input tensor (B, T, D_model).
            attn_mask (Optional[torch.Tensor]): Attention mask for self-attention (e.g., causal mask).
                                                Shape should match MHA expectations (e.g., (T, T) or (B*H, T, T)).
            key_padding_mask (Optional[torch.Tensor]): Key padding mask for self-attention (B, T).
            pgs_state_in (Optional[Dict]): Input state for PGS_FFN if state_mode='external'.
            epoch (Optional[int]): Current epoch number.
            loss_feedback (Optional[torch.Tensor]): Loss feedback for meta-learner.
            cross_layer_input (Optional[Dict]): Info from previous layer.
            max_epochs (Optional[int]): Total epochs for meta-learner context.


        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[Dict], Optional[Dict], Optional[Dict]]:
                - output tensor (B, T, D_model)
                - block aux loss (scalar float tensor)
                - pgs_state_out (dict or None)
                - analysis_data (dict or None)
                - pgs_cross_layer_output (dict or None)

        """
        block_analysis_data: Optional[Dict] = {"self_attn": {}, "pgs_ffn": {}} if self.collect_analysis else None
        logger.debug(f"TransformerBlock Forward: Input shape={x.shape}")
        block_start_time = time.time()

        # 1. Multi-Head Self-Attention (Sublayer -> Dropout -> Add -> Norm)
        attn_start_time = time.time()
        try:
            # Ensure mask dtypes are correct (bool or float depending on MHA version/usage)
            attn_output, attn_weights = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=self.collect_analysis,
                average_attn_weights=False,
            )  # Get per-head weights if needed
        except Exception as e:
            logger.error(f"Self-Attention forward failed: {e}", exc_info=True)
            # Return input? Raise error? Let's return input to potentially allow training to continue.
            return x, torch.tensor(0.0, device=x.device), pgs_state_in, block_analysis_data, None

        x_attn = self.norm1(x + self.dropout1(attn_output))
        attn_time = time.time() - attn_start_time
        logger.debug(f"Self-Attention took {attn_time:.4f} seconds.")
        if block_analysis_data is not None:
            block_analysis_data["self_attn"]["output_norm"] = torch.linalg.vector_norm(x_attn, dim=-1).mean().item()
            if attn_weights is not None:
                # Detach and potentially move weights to CPU if large and only needed for logging
                block_analysis_data["self_attn"]["weights_shape"] = list(attn_weights.shape)  # Store shape instead of full tensor
                # block_analysis_data['self_attn']['weights'] = attn_weights.detach().cpu()

        # 2. PGS Feed-Forward Network (Sublayer -> Dropout -> Add -> Norm)
        pgs_start_time = time.time()
        try:
            ffn_output, block_aux_loss, pgs_state_out, pgs_analysis_data, pgs_cross_output = self.pgs_ffn(
                x_attn,  # Input is the output of the first Norm layer
                state_in=pgs_state_in,
                epoch=epoch,
                loss_feedback=loss_feedback,
                cross_layer_input=cross_layer_input,
                max_epochs=max_epochs,
            )
        except Exception as e:
            logger.error(f"PGS_FFN forward failed: {e}", exc_info=True)
            # Return input from previous step?
            return x_attn, torch.tensor(0.0, device=x.device), pgs_state_in, block_analysis_data, None

        # Apply Post-Norm structure: Dropout -> Add -> Norm
        x_ffn = self.norm2(x_attn + self.dropout2(ffn_output))
        pgs_time = time.time() - pgs_start_time
        logger.debug(f"PGS_FFN took {pgs_time:.4f} seconds.")
        if block_analysis_data is not None:
            block_analysis_data["pgs_ffn"] = pgs_analysis_data  # Store the detailed dict from PGS_FFN
            if pgs_analysis_data:  # Check if data was collected
                block_analysis_data["pgs_ffn"]["overall_output_norm"] = torch.linalg.vector_norm(x_ffn, dim=-1).mean().item()
                block_analysis_data["pgs_ffn"]["total_block_time_sec"] = time.time() - block_start_time

        logger.debug(f"TransformerBlock Forward Complete. Output shape={x_ffn.shape}")
        # Return block output, aux loss from PGS_FFN, state from PGS_FFN, analysis data, cross-layer output
        return x_ffn, block_aux_loss, pgs_state_out, block_analysis_data, pgs_cross_output

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, attn_heads={self.n_heads_attn}, pgs_state_mode={self.state_mode}"
