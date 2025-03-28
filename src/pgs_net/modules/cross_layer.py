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
        self.geometry_config = config.get("geometry", DEFAULT_PGS_FFN_CONFIG["geometry"])  # Needed for branch count

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

    def _get_branch_count(self) -> int:
        """Determine expected number of geometry branches for projection."""
        # This is tricky as it might change dynamically. Use config for now.
        return len(self.geometry_config.get("branches", []))

    def _calculate_combined_info_dim(self) -> int:
        """Helper to get dimension of combined K/V or Gate input based on required_info."""
        dim = 0
        if "layer_output_tokens" in self.required_info:
            dim += self.d_model
        if "avg_global_queen" in self.required_info:
            dim += self.d_head
        if "avg_geometry_mix_weights" in self.required_info:
            # Use projection_dim as the target dimension for weights
            dim += self.projection_dim
        logger.debug(f"Calculated combined cross-layer info raw dimension: {dim}")
        return dim

    def _initialize_layers(self):
        """Initializes projection/attention layers based on method and required info."""
        self.combined_info_dim = self._calculate_combined_info_dim()

        # --- Projections for individual info sources (if needed before combining) ---
        self.info_source_projectors = nn.ModuleDict()
        if "avg_global_queen" in self.required_info and self.d_head != self.projection_dim:
            ProjLinear = ComplexLinear if self.is_complex else nn.Linear
            self.info_source_projectors["avg_global_queen"] = ProjLinear(self.d_head, self.projection_dim)
        if "avg_geometry_mix_weights" in self.required_info:
            num_branches = self._get_branch_count()
            if num_branches > 0:
                self.info_source_projectors["avg_geometry_mix_weights"] = nn.Linear(num_branches, self.projection_dim)

        if self.method == "conditional_input":
            # Project combined info vector to d_head for addition
            if self.combined_info_dim > 0:
                ProjectLinear = ComplexLinear if self.is_complex else nn.Linear
                try:
                    self.info_projector = ProjectLinear(self.combined_info_dim, self.d_head)
                except Exception as e:
                    logger.error(f"Failed init CrossLayer CondInput projector: {e}")
                    self.method = "none"
            else:
                logger.warning("CrossLayer CondInput: No info sources specified/dim calculation failed.")

        elif self.method == "attention":
            num_attn_heads = self.attn_params.get("num_attn_heads", 4)
            kv_proj_method = self.attn_params.get("kv_projection_method", "shared")
            variant = self.attn_params.get("attention_variant", "standard_mha")
            self.combine_method = self.attn_params.get("combine_method", "add_norm")
            # K/V source dimension (after potential individual projections)
            kv_source_dim = 0
            if "layer_output_tokens" in self.required_info:
                kv_source_dim += self.d_model  # Assumes token dim matches model dim
            if "avg_global_queen" in self.required_info:
                kv_source_dim += self.projection_dim  # Use projected dim
            if "avg_geometry_mix_weights" in self.required_info:
                kv_source_dim += self.projection_dim

            if kv_source_dim == 0:
                logger.error("CrossLayer Attention requires info source. Disabling.")
                self.method = "none"
                return

            # Project combined K/V source to self.projection_dim *for K and V*
            self.k_proj, self.v_proj = None, None
            if kv_proj_method == "separate":
                self.k_proj = nn.Linear(kv_source_dim, self.projection_dim)
                self.v_proj = nn.Linear(kv_source_dim, self.projection_dim)
            else:  # Shared projection -> split later
                self.kv_proj = nn.Linear(kv_source_dim, self.projection_dim * 2)

            # Attention Layer (Query=d_model, Key/Value=projection_dim)
            if variant == "linear_attn_stub":
                logger.warning("Linear Attention stub.")
                variant = "standard_mha"
            if variant == "standard_mha":
                if self.is_complex:
                    logger.warning("CrossLayer Attention uses standard MHA, treating inputs as real.")
                try:
                    # embed_dim is Query dim (d_model), kdim/vdim is Key/Value dim (projection_dim)
                    self.cross_attention = nn.MultiheadAttention(
                        self.d_model,
                        num_attn_heads,
                        kdim=self.projection_dim,
                        vdim=self.projection_dim,
                        batch_first=True,
                        dropout=self.attn_params.get("dropout", 0.1),
                    )
                except Exception as e:
                    logger.error(f"Failed init CrossLayer MHA: {e}")
                    self.method = "none"
                    return
            else:
                logger.error(f"Unsupported CrossLayer attention variant: {variant}")
                self.method = "none"
                return

            # Combine Layer
            if self.combine_method == "add_norm":
                self.norm = nn.LayerNorm(self.d_model)
            elif self.combine_method == "gate":
                self.output_gate = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.Sigmoid())

        elif self.method == "gating":
            gate_input_dim = self.combined_info_dim  # Use combined projected dim
            if gate_input_dim > 0:
                hidden_dim = max(32, gate_input_dim // 2)  # Small hidden layer
                self.gate_projector = nn.Sequential(
                    nn.Linear(gate_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self.d_model),
                    nn.Sigmoid(),  # Per-dimension gate (B, T, Dm)
                )
            else:
                logger.warning("CrossLayer Gating has no info sources.")
                self.method = "none"

    def _prepare_combined_info_vector(
        self, cross_layer_input: Dict[str, torch.Tensor], target_device: torch.device, target_seq_len: int
    ) -> Optional[torch.Tensor]:
        """Prepares a combined vector (B, T_target, CombinedDim) from available info sources."""
        if cross_layer_input is None:
            return None
        vec_parts = []
        processed_any = False

        # Process required info sources
        if "layer_output_tokens" in self.required_info and "layer_output_tokens" in cross_layer_input:
            tokens = cross_layer_input["layer_output_tokens"].to(target_device)  # (B, T_prev, Dm)
            # Project if needed
            if "layer_output_tokens" in self.info_source_projectors:
                tokens = self.info_source_projectors["layer_output_tokens"](tokens.float())  # Assume projector handles type
            # Ensure time dimension matches target_seq_len (e.g., take first token, avg, or pad/truncate?)
            # Simplification: Use token from previous layer corresponding to current T dim (assumes T is same)
            if tokens.shape[1] == target_seq_len:
                vec_parts.append(tokens.float())
                processed_any = True
            elif tokens.shape[1] > target_seq_len:
                vec_parts.append(tokens[:, :target_seq_len, :].float())
                processed_any = True
            else:  # Pad if T_prev < T
                padding = torch.zeros(
                    tokens.shape[0], target_seq_len - tokens.shape[1], tokens.shape[2], device=target_device, dtype=torch.float32
                )
                vec_parts.append(torch.cat([tokens.float(), padding], dim=1))
                processed_any = True

        if "avg_global_queen" in self.required_info and "avg_global_queen" in cross_layer_input:
            avg_q = cross_layer_input["avg_global_queen"].to(target_device)  # (B, Dh)
            # Project if needed
            if "avg_global_queen" in self.info_source_projectors:
                avg_q = self.info_source_projectors["avg_global_queen"](avg_q.float())  # (B, ProjDimQ)
            # Tile to match time dimension
            avg_q_tiled = avg_q.unsqueeze(1).expand(-1, target_seq_len, -1)  # (B, T_target, ProjDimQ)
            vec_parts.append(avg_q_tiled.float())
            processed_any = True

        if "avg_geometry_mix_weights" in self.required_info and "avg_geometry_mix_weights" in cross_layer_input:
            weights = cross_layer_input["avg_geometry_mix_weights"].to(target_device)  # (NumBranches,) or (B, NumBranches)
            if weights.dim() == 1:
                weights = weights.unsqueeze(0).expand(cross_layer_input["avg_global_queen"].shape[0], -1)  # Expand B dim
            # Project weights
            if "avg_geometry_mix_weights" in self.info_source_projectors:
                projected_weights = self.info_source_projectors["avg_geometry_mix_weights"](weights.float())  # (B, ProjDimW)
            else:
                projected_weights = weights.float()  # Use raw weights
            # Tile T dim
            projected_weights_tiled = projected_weights.unsqueeze(1).expand(-1, target_seq_len, -1)
            vec_parts.append(projected_weights_tiled)
            processed_any = True

        if not processed_any:
            return None
        # Concatenate along feature dimension
        try:
            return torch.cat(vec_parts, dim=-1)  # (B, T_target, CombinedInfoDim)
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
        if not self.config.get("use_cross_layer_comm", False):
            return None
        output_info: Dict[str, torch.Tensor] = {}
        info_keys = self.config.get("cross_layer_info", [])

        if "avg_global_queen" in info_keys and global_queens_list:
            try:
                avg_global_queen = torch.stack(global_queens_list, dim=0).mean(dim=0).detach()
                output_info["avg_global_queen"] = avg_global_queen
                # logger.debug("CrossLayer: Collected average global queen.")
            except Exception as e:
                logger.error(f"Failed to collect avg global queen: {e}")

        if "avg_geometry_mix_weights" in info_keys and analysis_data and analysis_data.get("heads"):
            try:
                weights_list = [
                    h_data.get("geometry_mix_weights")
                    for h_data in analysis_data["heads"]
                    if h_data and "geometry_mix_weights" in h_data
                ]
                if weights_list and all(w is not None for w in weights_list):
                    avg_weights = torch.stack(weights_list, dim=0).mean(dim=0).detach()
                    output_info["avg_geometry_mix_weights"] = avg_weights
                    # logger.debug("CrossLayer: Collected average geometry weights.")
            except Exception as e:
                logger.error(f"Failed to collect avg geometry weights: {e}")

        if "layer_output_tokens" in info_keys and layer_output_tokens is not None:
            output_info["layer_output_tokens"] = layer_output_tokens.detach()
            # logger.debug("CrossLayer: Collected layer output tokens.")

        return output_info if output_info else None

    def inject_input_info(self, x: torch.Tensor, cross_layer_input: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Modifies input x (B, T, Dm) based on received cross-layer info BEFORE head split."""
        if cross_layer_input is None or not self.config["use_cross_layer_comm"] or self.method == "none":
            return x

        start_time = time.time()
        original_norm = torch.linalg.vector_norm(x, dim=-1).mean().item()
        B, T, Dm = x.shape
        modified_x = x  # Start with original

        if self.method == "conditional_input":
            logger.warning("CrossLayer CondInput injection needs refinement for pre-split.")
            # If we add directly to x (B,T,Dm), the projected info needs to be Dm, not Dh.
            # Requires recalculating combined_info_dim and info_projector output size.
            # Simplified stub: return original x for now.
            pass

        elif self.method == "attention":
            if hasattr(self, "cross_attention"):
                kv_input_combined = self._prepare_combined_info_vector(cross_layer_input, x.device, T)  # Target T is current T
                if kv_input_combined is not None:
                    # Project K, V
                    keys, values = None, None
                    if hasattr(self, "kv_proj"):
                        kv_projected = self.kv_proj(kv_input_combined.float())
                        keys, values = torch.chunk(kv_projected, 2, dim=-1)
                    elif hasattr(self, "k_proj"):
                        keys = self.k_proj(kv_input_combined.float())
                        values = self.v_proj(kv_input_combined.float())
                    if keys is not None and values is not None:
                        # Assume real MHA
                        x_real = x.real if x.is_complex() else x
                        k_real = keys.real if keys.is_complex() else keys
                        v_real = values.real if values.is_complex() else values
                        try:
                            # Query=x, Key=k, Value=v
                            attn_output, _ = self.cross_attention(x_real, k_real, v_real)
                            # Combine
                            if self.combine_method == "add_norm" and hasattr(self, "norm"):
                                x_modified_real = self.norm(x_real + attn_output)
                            elif self.combine_method == "gate" and hasattr(self, "output_gate"):
                                gate = self.output_gate(attn_output)
                                x_modified_real = x_real + gate * attn_output
                            else:
                                x_modified_real = x_real + attn_output
                            # Recombine complex
                            modified_x = torch.complex(x_modified_real, x.imag) if x.is_complex() else x_modified_real
                            logger.debug("CrossLayer: Applied cross-attention modification.")
                        except Exception as e:
                            logger.error(f"CrossLayer Attention forward failed: {e}", exc_info=True)
                    else:
                        logger.warning("CrossLayer Attention: K/V projection failed.")
                else:
                    logger.warning("CrossLayer Attention: Could not prepare K/V source.")
            else:
                logger.warning("CrossLayer Attention enabled but layers missing.")

        elif self.method == "gating":
            if hasattr(self, "gate_projector"):
                combined_info_vec = self._prepare_combined_info_vector(cross_layer_input, x.device, T)
                if combined_info_vec is not None:
                    # Average info over T dimension for global gate? Or apply per token? Per token.
                    gate_val = self.gate_projector(combined_info_vec.float())  # (B, T, Dm) -> Sigmoid output
                    modified_x = x * gate_val.to(x.dtype)  # Apply gate (element-wise)
                    logger.debug("CrossLayer: Applied gating modification.")
                else:
                    logger.warning("CrossLayer Gating: No valid info source found.")
            else:
                logger.warning("CrossLayer Gating enabled but layers missing.")

        final_norm = torch.linalg.vector_norm(modified_x, dim=-1).mean().item()
        logger.debug(
            f"CrossLayer injection took {time.time() - start_time:.4f} sec. Norm change: {original_norm:.3f} -> {final_norm:.3f}"
        )
        return modified_x

    def extra_repr(self) -> str:
        return f"method={self.method}, required_info={self.required_info}"
