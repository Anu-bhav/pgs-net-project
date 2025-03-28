# src/pgs_net/modules/cross_layer.py
"""Cross-Layer Communication Handler."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import DEFAULT_PGS_FFN_CONFIG
from .complex_utils import ComplexLinear  # Assuming complex_utils is at the same level

logger = logging.getLogger(__name__)


class CrossLayerHandler(nn.Module):
    """Handles processing and injection of cross-layer information."""

    def __init__(self, d_model: int, d_head: int, num_heads: int, config: Dict[str, Any]):
        """
        Initializes the CrossLayerHandler.

        Args:
            d_model (int): Full model dimension.
            d_head (int): Dimension per head.
            num_heads (int): Number of heads.
            config (Dict[str, Any]): Full PGS_FFN configuration.

        """
        super().__init__()
        self.config = config.get("cross_layer", DEFAULT_PGS_FFN_CONFIG["cross_layer"])
        self.arch_config = config.get("architecture", DEFAULT_PGS_FFN_CONFIG["architecture"])
        self.is_complex = self.arch_config.get("use_complex_representation", False)
        self.dtype = torch.complex64 if self.is_complex else torch.float32
        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads

        self.method = self.config.get("cross_layer_type", "none")
        self.required_info = self.config.get("cross_layer_info", [])
        self.projection_dim = self.config.get("cross_layer_projection_dim", 64)
        self.attn_params = self.config.get("cross_layer_params", {})

        logger.info(
            f"Initializing CrossLayerHandler: Method={self.method}, ProjDim={self.projection_dim}, Info={self.required_info}"
        )

        # --- Layers based on Method ---
        self._initialize_layers()

    def _initialize_layers(self):
        """Initializes projection/attention layers based on method and required info."""
        # --- Calculate combined dimension of available info sources ---
        # This dimension depends on what's actually passed in cross_layer_input
        self.combined_info_dim = self._calculate_combined_info_dim()

        if self.method == "conditional_input":
            if self.combined_info_dim > 0:
                ProjectLinear = ComplexLinear if self.is_complex else nn.Linear
                # Projects combined info vector to d_head for addition
                try:
                    self.info_projector = ProjectLinear(self.combined_info_dim, self.d_head)
                except Exception as e:
                    logger.error(f"Failed init CrossLayer CondInput projector: {e}")
                    self.method = "none"
            else:
                logger.warning("CrossLayer CondInput: No info sources specified or dim calculation failed.")

        elif self.method == "attention":
            num_attn_heads = self.attn_params.get("num_attn_heads", 4)
            kv_proj_method = self.attn_params.get("kv_projection_method", "shared")
            variant = self.attn_params.get("attention_variant", "standard_mha")
            self.combine_method = self.attn_params.get("combine_method", "add_norm")
            kv_dim = self.combined_info_dim  # Dimension of concatenated K/V source features

            if kv_dim == 0:
                logger.error("CrossLayer Attention requires info source. Disabling.")
                self.method = "none"
                return

            # Project K/V source to projection_dim
            if kv_proj_method == "separate":
                self.k_proj = nn.Linear(kv_dim, self.projection_dim)
                self.v_proj = nn.Linear(kv_dim, self.projection_dim)
            else:  # Shared projection -> split later
                self.kv_proj = nn.Linear(kv_dim, self.projection_dim * 2)

            # Attention Layer (Query=d_model, Key/Value=projection_dim)
            if variant == "linear_attn_stub":
                logger.warning("Linear Attention stub.")
                variant = "standard_mha"
            if variant == "standard_mha":
                if self.is_complex:
                    logger.warning("CrossLayer Attention uses standard MHA, complex input/output needs careful handling.")
                try:
                    self.cross_attention = nn.MultiheadAttention(
                        self.d_model,
                        num_attn_heads,
                        kdim=self.projection_dim,
                        vdim=self.projection_dim,
                        batch_first=True,
                        dropout=0.1,
                    )
                except Exception as e:
                    logger.error(f"Failed init CrossLayer MHA: {e}")
                    self.method = "none"
                    return
            else:
                logger.error(f"Unsupported CrossLayer attention variant: {variant}")
                self.method = "none"
                return

            # Combine Layer (Norm or Gate)
            if self.combine_method == "add_norm":
                self.norm = nn.LayerNorm(self.d_model)
            elif self.combine_method == "gate":
                self.output_gate = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.Sigmoid())  # Per-dim gate

        elif self.method == "gating":
            if self.combined_info_dim > 0:
                # Project combined info to d_model to generate gates
                self.gate_projector = nn.Sequential(
                    nn.Linear(self.combined_info_dim, 64),
                    nn.ReLU(),  # Intermediate hidden layer
                    nn.Linear(64, self.d_model),
                    nn.Sigmoid(),  # Output gate per dimension
                )
            else:
                logger.warning("CrossLayer Gating has no info sources.")
                self.method = "none"

    def _calculate_combined_info_dim(self) -> int:
        """Helper to get dimension of combined K/V or Gate input based on required_info."""
        dim = 0
        # Add dims based on self.required_info, assuming standard shapes
        if "layer_output_tokens" in self.required_info:
            dim += self.d_model  # Full Dm
        if "avg_global_queen" in self.required_info:
            dim += self.d_head  # Head dim
        if "avg_geometry_mix_weights" in self.required_info:
            # Dimension depends on number of branches - hard to know without config?
            # Use projection_dim as a placeholder fixed dimension for processed weights
            dim += self.config.get("cross_layer_projection_dim", 32)  # Use fixed proj dim

        logger.debug(f"Calculated combined cross-layer info dimension: {dim}")
        return dim

    def _prepare_combined_info_vector(
        self, cross_layer_input: Dict[str, torch.Tensor], target_device: torch.device
    ) -> Optional[torch.Tensor]:
        """Prepares a combined vector from available info sources for projection/attention K/V/Gating."""
        if cross_layer_input is None:
            return None
        vec_parts = []
        ref_shape_bt = None  # Store (B, T) shape if available

        # Process required info sources
        if "layer_output_tokens" in self.required_info and "layer_output_tokens" in cross_layer_input:
            tokens = cross_layer_input["layer_output_tokens"].to(target_device)  # (B, T_prev, Dm)
            ref_shape_bt = tokens.shape[:2]
            # Project if needed? Assume used directly if main source. Handle dtype later.
            vec_parts.append(tokens.float())  # Use float for processing

        if "avg_global_queen" in self.required_info and "avg_global_queen" in cross_layer_input:
            avg_q = cross_layer_input["avg_global_queen"].to(target_device)  # (B, Dh)
            # Tile to match time dimension if ref_shape_bt exists
            T_dim = ref_shape_bt[1] if ref_shape_bt is not None else 1
            avg_q_tiled = avg_q.unsqueeze(1).expand(-1, T_dim, -1)  # (B, T_prev or 1, Dh)
            # Ensure correct feature dimension (pad/project if needed)
            if avg_q_tiled.shape[-1] != self.d_head:
                logger.warning("Mismatch Dh for avg_global_queen.")
            # Add projection layer if we always project info sources?
            # Simplified: Assume simple concatenation works dimension-wise
            vec_parts.append(avg_q_tiled.float())

        if "avg_geometry_mix_weights" in self.required_info and "avg_geometry_mix_weights" in cross_layer_input:
            weights = cross_layer_input["avg_geometry_mix_weights"].to(target_device)  # (NumBranches,) or (B, NumBranches)
            if weights.dim() == 1:
                weights = weights.unsqueeze(0).expand(ref_shape_bt[0] if ref_shape_bt else 1, -1)  # Expand B dim
            # Project weights to projection_dim
            if hasattr(self, "geom_weights_processor"):  # Check if defined during init based on combined_info_dim calc
                projected_weights = self.geom_weights_processor(weights.float())  # (B, ProjDim)
            else:  # Use weights directly if no specific processor (maybe concat with zeros?)
                projected_weights = weights.float()  # Use raw weights (ensure float)

            # Tile T dim if needed
            T_dim = ref_shape_bt[1] if ref_shape_bt is not None else 1
            if projected_weights.dim() == 2:
                projected_weights = projected_weights.unsqueeze(1).expand(-1, T_dim, -1)
            vec_parts.append(projected_weights)

        # --- Concatenate processed parts ---
        if not vec_parts:
            return None
        # Ensure all parts have same T dimension before concat
        max_t = max(p.shape[1] for p in vec_parts if p.dim() > 2) if any(p.dim() > 2 for p in vec_parts) else 1
        processed_parts = []
        for p in vec_parts:
            if p.dim() == 2:
                p = p.unsqueeze(1).expand(-1, max_t, -1)  # Tile T dim
            elif p.dim() == 3 and p.shape[1] != max_t:
                logger.warning(f"Skipping cross-layer part due to T dim mismatch: {p.shape[1]} vs {max_t}")
                continue
            processed_parts.append(p)

        if not processed_parts:
            return None
        try:
            return torch.cat(processed_parts, dim=-1)  # (B, max_t, CombinedInfoDim)
        except RuntimeError as e:
            logger.error(f"CrossLayer info concat error: {e}")
            return None

    # --- Injection/Collection Methods ---
    def collect_output_info(
        self,
        analysis_data: Optional[Dict],
        global_queens_list: List[torch.Tensor],
        layer_output_tokens: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Aggregates information from the current layer to pass to the next."""
        # ... (Implementation as refined previously, collecting info based on self.required_info) ...
        # Ensure layer_output_tokens is collected if requested.
        output_info = {}  # ... collect avg_queen, avg_geo_weights ...
        if "layer_output_tokens" in self.required_info and layer_output_tokens is not None:
            output_info["layer_output_tokens"] = layer_output_tokens.detach()
        return output_info if output_info else None

    def inject_input_info(self, x: torch.Tensor, cross_layer_input: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Modifies input x (B, T, Dm) based on received cross-layer info."""
        if cross_layer_input is None or not self.config["use_cross_layer_comm"] or self.method == "none":
            return x

        start_time = time.time()
        original_norm = x.norm().item()

        if self.method == "conditional_input":
            # Project combined info vector and add
            combined_info_vec = self._prepare_combined_info_vector(cross_layer_input, x.device)  # (B, T_kv, InfoDim)
            if hasattr(self, "info_projector") and combined_info_vec is not None:
                # Average info over T_kv dimension? Or use first token's info? Average.
                avg_info = combined_info_vec.mean(dim=1)  # (B, InfoDim)
                projected_info = self.info_projector(avg_info)  # (B, Dh)
                # Need to reshape for adding to x (B, T, Dm) -> Assumes Dh == Dm here! Needs fix if not.
                if projected_info.shape[-1] == x.shape[-1]:
                    x = x + projected_info.unsqueeze(1)  # Add to input (broadcast T)
                    logger.debug("CrossLayer: Added projected info via conditional input.")
                else:
                    logger.error("CrossLayer CondInput projection dim mismatch.")
            else:
                logger.warning("CrossLayer CondInput failed (no projector or info).")

        elif self.method == "attention":
            if hasattr(self, "cross_attention"):
                kv_input_combined = self._prepare_combined_info_vector(cross_layer_input, x.device)  # (B, T_kv, CombinedKVDim)
                if kv_input_combined is not None:
                    # --- Project K, V ---
                    if hasattr(self, "kv_proj"):  # Shared
                        kv_projected = self.kv_proj(kv_input_combined.float())
                        k, v = torch.chunk(kv_projected, 2, dim=-1)
                    elif hasattr(self, "k_proj"):
                        k = self.k_proj(kv_input_combined.float())
                        v = self.v_proj(kv_input_combined.float())
                    else:
                        k = v = kv_input_combined  # Assume dims match

                    # --- Perform Cross-Attention ---
                    # Assume real MHA for now
                    x_real = x.real if x.is_complex() else x
                    k_real = k.real if k.is_complex() else k
                    v_real = v.real if v.is_complex() else v
                    try:
                        # Query=x, Key=k, Value=v
                        attn_output, _ = self.cross_attention(x_real, k_real, v_real)  # (B, T, Dm)
                        # --- Combine output ---
                        if self.combine_method == "add_norm" and hasattr(self, "norm"):
                            x_modified_real = self.norm(x_real + attn_output)
                        elif self.combine_method == "gate" and hasattr(self, "output_gate"):
                            gate = self.output_gate(attn_output)
                            x_modified_real = x_real + gate * attn_output
                        else:
                            x_modified_real = x_real + attn_output  # Default add

                        # Recombine complex if needed
                        x = torch.complex(x_modified_real, x.imag) if x.is_complex() else x_modified_real
                        logger.debug("CrossLayer: Applied cross-attention modification.")
                    except Exception as e:
                        logger.error(f"CrossLayer Attention failed: {e}")
                else:
                    logger.warning("CrossLayer Attention: Could not prepare K/V source.")
            else:
                logger.warning("CrossLayer Attention enabled but layers missing.")

        elif self.method == "gating":
            if hasattr(self, "gate_projector"):
                combined_info_vec = self._prepare_combined_info_vector(cross_layer_input, x.device)  # (B, T_kv, InfoDim)
                if combined_info_vec is not None:
                    # Average info over T_kv dimension for global gate
                    avg_info = combined_info_vec.mean(dim=1)  # (B, InfoDim)
                    gate_val = self.gate_projector(avg_info.float())  # (B, Dm) -> Sigmoid output
                    x = x * gate_val.unsqueeze(1)  # Apply gate (broadcast T)
                    logger.debug("CrossLayer: Applied gating modification.")
                else:
                    logger.warning("CrossLayer Gating: No valid info source found.")
            else:
                logger.warning("CrossLayer Gating enabled but layers missing.")

        logger.debug(
            f"CrossLayer injection took {time.time() - start_time:.4f} sec. Norm change: {original_norm:.3f} -> {x.norm().item():.3f}"
        )
        return x
