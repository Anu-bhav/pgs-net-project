# src/pgs_net/modules/force_calculator.py
"""Computes the combined update force vector."""

import logging
import time
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..config import DEFAULT_PGS_FFN_CONFIG
from .complex_utils import ComplexLinear
from .formal_force import PlaceholderFormalForce, PotentialEnergyForceV2
from .interfaces import FormalForceCalculator, NeighborSearch
from .neighbor_search import FaissNeighborSearch, NaiveNeighborSearch

logger = logging.getLogger(__name__)


# Helper for shared parameters
def get_shared_parameter(
    module: nn.Module, param_name: str, shared_params_dict: Dict[str, nn.Parameter], default_factory
) -> nn.Parameter:
    """Gets or creates a shared parameter and registers it locally."""
    if param_name not in shared_params_dict:
        shared_params_dict[param_name] = default_factory()
        logger.debug(f"Created shared parameter: {param_name} with shape {shared_params_dict[param_name].shape}")
    shared_attr_name = f"_shared_{param_name}"
    if not hasattr(module, shared_attr_name):
        setattr(module, shared_attr_name, shared_params_dict[param_name])
    return shared_params_dict[param_name]


class UpdateForceCalculator(nn.Module):
    """
    Calculates the raw update force for each token by combining influences
    from local/global queens, optional physics fields, Boids rules, or formal potential gradients.
    """

    def __init__(self, d_head: int, num_heads: int, max_clusters: int, config: Dict, shared_params: Optional[Dict] = None):
        """Initializes the UpdateForceCalculator."""
        super().__init__()
        self.config_force = config.get("update_forces", DEFAULT_PGS_FFN_CONFIG["update_forces"])
        self.arch_config = config.get("architecture", DEFAULT_PGS_FFN_CONFIG["architecture"])
        self.param_sharing_config = self.arch_config.get("param_sharing", {})
        self.shared_params = shared_params if shared_params is not None else {}

        self.is_complex = self.arch_config.get("use_complex_representation", False)
        self.dtype = torch.complex64 if self.is_complex else torch.float32
        self.d_head = d_head
        self.num_heads = num_heads
        self.max_clusters = max_clusters

        # --- Neighbor Search Module (for Boids) ---
        self.neighbor_search_module: Optional[NeighborSearch] = None
        if self.config_force.get("use_boids_rules", False):
            k = self.config_force.get("boids_neighbor_k", 5)
            if k > 0:  # Only init if k > 0
                use_faiss = self.config_force.get("boids_use_faiss", True) and faiss_available
                if use_faiss:
                    try:
                        self.neighbor_search_module = FaissNeighborSearch(
                            k,
                            self.d_head,
                            self.is_complex,
                            index_type=self.config_force.get("boids_faiss_index_type", "IndexFlatL2"),
                            update_every=self.config_force.get("boids_update_index_every", 1),
                            use_gpu=True,  # Assume GPU preferred
                        )
                    except Exception as e:
                        logger.error(f"Failed to init FaissNeighborSearch: {e}. Falling back.", exc_info=True)
                        use_faiss = False
                if not use_faiss:  # Fallback to Naive if Faiss failed or not requested
                    self.neighbor_search_module = NaiveNeighborSearch(k, self.is_complex)
            else:
                logger.warning("Boids enabled but k=0, no neighbor search module created.")

        # --- Formal Force Calculator ---
        self.formal_force_calc: Optional[FormalForceCalculator] = None
        self.use_formal_force = self.config_force.get("use_formal_force", False)
        if self.use_formal_force:
            force_type = self.config_force.get("formal_force_type", "none")
            if force_type == "potential_v2":
                self.formal_force_calc = PotentialEnergyForceV2(config, self.is_complex, self.dtype)
            # Add elif for other formal types...
            elif force_type != "none":
                self.formal_force_calc = PlaceholderFormalForce(config, self.is_complex, self.dtype)
                logger.warning(f"Using Placeholder Formal Force ({force_type} not impl).")
            # Ensure we don't use it if instantiation failed or type is none
            if self.formal_force_calc is None or isinstance(self.formal_force_calc, PlaceholderFormalForce):
                self.use_formal_force = False

        # --- Other Parameters / Layers ---
        # Fitness Layer
        share_force = self.param_sharing_config.get("share_force_params", False)
        if self.config_force.get("use_fitness_modulation", False):
            factory_fit = lambda: (ComplexLinear if self.is_complex else nn.Linear)(d_head, 1)
            self.fitness_layer = (
                get_shared_parameter(self, "fitness_layer", self.shared_params, factory_fit) if share_force else factory_fit()
            )
            logger.info(f"Fitness modulation enabled (Shared={share_force}).")

        # Interaction Field Charges
        if self.config_force.get("use_interaction_fields", False):
            if self.config_force.get("learnable_charges", False):
                factory_charge_head = lambda: nn.Parameter(torch.randn(num_heads, max_clusters))
                factory_charge_shared = lambda: nn.Parameter(torch.randn(max_clusters))
                if share_force:
                    self._shared_centroid_charges = get_shared_parameter(
                        self, "centroid_charges", self.shared_params, factory_charge_shared
                    )
                else:
                    self.centroid_charges_local = factory_charge_head()
                logger.info(f"Interaction fields enabled with learnable charges (Shared={share_force}).")
            else:
                # Fixed charges buffer - needs access per head index
                self.register_buffer("centroid_charges_fixed", torch.ones(num_heads, max_clusters))
                logger.info("Interaction fields enabled with fixed charges (1.0).")

    # --- Forward Pass ---
    def forward(
        self,
        x_h: torch.Tensor,
        A: torch.Tensor,
        local_queens: torch.Tensor,
        global_queen: torch.Tensor,
        centroids_h: torch.Tensor,
        head_idx: int,
        state_in: Optional[Dict] = None,  # Rarely used here
        dynamic_params: Optional[Dict] = None,
        analysis_data: Optional[Dict] = None,
        last_update_state: Optional[torch.Tensor] = None,  # State from Integrator
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Calculates the combined raw update force and density modulation factor."""
        B, T, D = x_h.shape
        K_max = centroids_h.shape[0]
        force_analysis = {} if analysis_data is not None else None
        state_out: Dict[str, torch.Tensor] = {}
        logger.debug(f"[Head {head_idx}] Update Force Calculation started.")

        # --- Formal Force Calculation (Overrides others if enabled) ---
        if self.use_formal_force and self.formal_force_calc is not None:
            logger.debug("Using Formal Force Calculation.")
            combined_force = self.formal_force_calc.calculate_force(x_h, A, local_queens, global_queen, centroids_h)
            density_mod_factor = self._calculate_density_mod_factor(
                A, T, analysis_data=force_analysis
            )  # Still calculate density factor
            if force_analysis is not None:
                force_analysis["force_formal_norm_avg"] = torch.linalg.vector_norm(combined_force, dim=-1).mean().item()
            return combined_force, density_mod_factor, state_out  # Return early

        # --- Standard Force Pipeline ---
        # --- Get Effective Hyperparameters ---
        alpha = (
            dynamic_params.get("local_influence_weight", self.config_force["local_influence_weight"])
            if dynamic_params
            else self.config_force["local_influence_weight"]
        )
        beta = (
            dynamic_params.get("global_influence_weight", self.config_force["global_influence_weight"])
            if dynamic_params
            else self.config_force["global_influence_weight"]
        )
        alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha  # Use scalar values for weights
        beta = beta.item() if isinstance(beta, torch.Tensor) else beta

        # --- Base Local/Global Influence ---
        # Ensure A is float for bmm with complex x_h
        local_influence = torch.bmm(A.float(), local_queens)  # (B, T, K_max) @ (B, K_max, D) -> (B, T, D)
        local_dir = local_influence - x_h
        global_dir = global_queen.unsqueeze(1) - x_h

        # --- Fitness Modulation (P) ---
        P_factor = 1.0
        if self.config_force.get("use_fitness_modulation", False) and hasattr(self, "fitness_layer"):
            fitness_layer = (
                self._shared_fitness_layer if self.param_sharing_config.get("share_force_params", False) else self.fitness_layer
            )
            with torch.cuda.amp.autocast(enabled=False):  # Run fitness linear layers in float32 for stability?
                fit_token = fitness_layer(x_h.float()).real  # Ensure real output for sigmoid
                fit_local = fitness_layer(local_influence.float()).real
            P = torch.sigmoid(fit_local - fit_token)  # (B, T, 1)
            local_dir = P * local_dir  # Apply modulation
            P_factor = P  # Store for analysis maybe
            if force_analysis is not None:
                force_analysis["fitness_mod_P_avg"] = P.mean().item()

        # Combine base forces using potentially dynamic weights
        combined_force = alpha * local_dir + beta * global_dir
        if force_analysis is not None:  # Log base component norms
            force_analysis["force_local_dir_norm_avg"] = torch.linalg.vector_norm(local_dir, dim=-1).mean().item()
            force_analysis["force_global_dir_norm_avg"] = torch.linalg.vector_norm(global_dir, dim=-1).mean().item()

        # --- Interaction Fields ---
        if self.config_force.get("use_interaction_fields", False):
            charges_param = None
            learnable = self.config_force.get("learnable_charges", False)
            share = self.param_sharing_config.get("share_force_params", False)
            if learnable:
                charges_param = self._shared_centroid_charges if share else self.centroid_charges_local[head_idx]
            elif hasattr(self, "centroid_charges_fixed"):
                charges_param = self.centroid_charges_fixed[head_idx]

            if charges_param is not None:
                charges = charges_param.detach()  # Use detached charges as fixed sources for force calc
                token_charge = 1.0
                strength = (
                    dynamic_params.get("interaction_field_strength", self.config_force["interaction_field_strength"])
                    if dynamic_params
                    else self.config_force["interaction_field_strength"]
                )
                if not isinstance(strength, torch.Tensor):
                    strength = torch.tensor(strength)
                strength = strength.item()  # Scalar strength

                diff = x_h.unsqueeze(2) - centroids_h.detach().unsqueeze(0).unsqueeze(0)
                dist = torch.linalg.vector_norm(diff, ord=2, dim=-1).clamp(min=1e-6)  # Real distance
                field_force_contrib = torch.zeros_like(diff)
                if self.config_force["interaction_field_type"] == "coulomb":
                    force_mag_over_dist = (strength * token_charge * charges.view(1, 1, K_max)) / dist.pow(3)
                    field_force_contrib = -force_mag_over_dist.unsqueeze(-1) * diff  # Complex if diff is complex
                # Aggregate force based on assignment A (use detached A)
                total_field_force = (A.detach().unsqueeze(-1) * field_force_contrib).sum(dim=2)
                combined_force = combined_force + total_field_force
                if force_analysis is not None:
                    force_analysis["force_field_norm_avg"] = torch.linalg.vector_norm(total_field_force, dim=-1).mean().item()
            else:
                logger.warning(f"[Head {head_idx}] Interaction fields enabled but charges parameter not found.")

        # --- Boids Rules ---
        if self.config_force.get("use_boids_rules", False) and self.neighbor_search_module is not None:
            logger.debug(f"[Head {head_idx}] Calculating Boids forces...")
            neighbor_indices, neighbor_vecs = self.neighbor_search_module.find_neighbors(x_h)  # (B,T,k), (B,T,k,D)

            if neighbor_vecs is not None and neighbor_vecs.numel() > 0 and neighbor_indices is not None:
                valid_neighbor_mask = (neighbor_indices != -1).unsqueeze(-1)  # (B, T, k, 1) for broadcasting
                num_neighbors = valid_neighbor_mask.sum(dim=2).clamp(min=1)  # (B, T, 1)

                # --- Get dynamic weights ---
                sep_w = (
                    dynamic_params.get("boids_separation_weight", self.config_force["boids_separation_weight"])
                    if dynamic_params
                    else self.config_force["boids_separation_weight"]
                )
                coh_w = (
                    dynamic_params.get("boids_cohesion_weight", self.config_force["boids_cohesion_weight"])
                    if dynamic_params
                    else self.config_force["boids_cohesion_weight"]
                )
                align_w = (
                    dynamic_params.get("boids_alignment_weight", self.config_force["boids_alignment_weight"])
                    if dynamic_params
                    else self.config_force["boids_alignment_weight"]
                )
                if not isinstance(sep_w, torch.Tensor):
                    sep_w = torch.tensor(sep_w)  # Ensure tensor for math if needed
                if not isinstance(coh_w, torch.Tensor):
                    coh_w = torch.tensor(coh_w)
                if not isinstance(align_w, torch.Tensor):
                    align_w = torch.tensor(align_w)

                # --- Separation ---
                total_sep_force = torch.zeros_like(x_h)
                if sep_w > 1e-6:
                    diff_from_neighbors = x_h.unsqueeze(2) - neighbor_vecs  # (B, T, k, D)
                    dist_sq = diff_from_neighbors.abs().pow(2).sum(dim=-1).clamp(min=1e-6)  # (B, T, k)
                    sep_force_per = -diff_from_neighbors / dist_sq.unsqueeze(-1)  # Repulsion proportional to 1/r^2 * direction
                    # Average forces from valid neighbors
                    total_sep_force = (
                        torch.where(valid_neighbor_mask, sep_force_per, torch.zeros_like(sep_force_per)).sum(dim=2)
                        / num_neighbors
                    )
                    if force_analysis:
                        force_analysis["force_boids_sep_norm_avg"] = (
                            torch.linalg.vector_norm(total_sep_force, dim=-1).mean().item()
                        )

                # --- Cohesion ---
                total_cohesion_force = torch.zeros_like(x_h)
                if coh_w > 1e-6:
                    # Average position of valid neighbors
                    avg_neighbor_pos = (
                        torch.where(valid_neighbor_mask, neighbor_vecs, torch.zeros_like(neighbor_vecs)).sum(dim=2)
                        / num_neighbors
                    )
                    total_cohesion_force = avg_neighbor_pos - x_h  # Force towards avg pos
                    if force_analysis:
                        force_analysis["force_boids_coh_norm_avg"] = (
                            torch.linalg.vector_norm(total_cohesion_force, dim=-1).mean().item()
                        )

                # --- Alignment ---
                total_align_force = torch.zeros_like(x_h)
                if align_w > 1e-6 and last_update_state is not None:
                    logger.debug(f"[Head {head_idx}] Calculating Boids Alignment force...")
                    try:
                        avg_neighbor_update = torch.zeros_like(x_h)
                        state_shape = last_update_state.shape
                        if state_shape == x_h.shape:  # Full state (B, T, D) - External mode usually
                            # Need valid neighbor_indices (B, T, k)
                            if neighbor_indices.max() < T and neighbor_indices.min() >= -1:
                                valid_mask_k = neighbor_indices != -1  # (B, T, k)
                                num_valid_neighbors = valid_mask_k.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, T, 1)
                                # Create indices for gathering efficiently
                                B_idx = torch.arange(B, device=x_h.device).view(-1, 1, 1)
                                T_idx = torch.arange(T, device=x_h.device).view(1, -1, 1)
                                # Gather using valid indices, handle -1 safely
                                safe_indices = torch.where(valid_mask_k, neighbor_indices, torch.zeros_like(neighbor_indices))
                                gathered_updates = last_update_state[B_idx, safe_indices]  # (B, T, k, D)
                                # Zero out contributions from invalid neighbors before summing
                                summed_updates = torch.where(
                                    valid_mask_k.unsqueeze(-1), gathered_updates, torch.zeros_like(gathered_updates)
                                ).sum(dim=2)  # (B, T, D)
                                avg_neighbor_update = summed_updates / num_valid_neighbors
                        elif state_shape[0] == 1 and state_shape[2] == D:  # Buffer mode (average) (1, 1, D) or (1, D)
                            avg_neighbor_update = last_update_state.expand_as(x_h)  # Expand avg dir
                            logger.debug("Using averaged last update state for Boids Alignment.")
                        else:
                            logger.warning(f"Unexpected shape for Boids last_update_state: {state_shape}. Skipping Alignment.")

                        total_align_force = avg_neighbor_update  # Align towards average update direction
                        if force_analysis:
                            force_analysis["force_boids_align_norm_avg"] = (
                                torch.linalg.vector_norm(total_align_force, dim=-1).mean().item()
                            )
                    except Exception as e:
                        logger.error(f"Failed Alignment force calculation: {e}", exc_info=True)
                else:
                    logger.debug("Skipping Boids Alignment (no state/zero weight).")

                # Add Boids forces using dynamic weights
                combined_force = (
                    combined_force + sep_w * total_sep_force + coh_w * total_cohesion_force + align_w * total_align_force
                )
            else:
                logger.debug(f"[Head {head_idx}] Skipping Boids forces (no neighbors found/returned).")
        # --- End Boids ---

        # --- Density Modulation Factor Calculation ---
        density_mod_factor = torch.tensor(1.0, device=x_h.device, dtype=torch.float32)
        if self.config_force.get("use_density_modulation", False):
            density_mod_factor = self._calculate_density_mod_factor(A, T, analysis_data=force_analysis)
            if "influence_weights" in self.config_force.get("density_modulation_target", []):
                logger.debug("Applying density modulation to influence weights.")
                # Recalculate base force with modulated weights
                combined_force = (alpha * local_dir * density_mod_factor) + (beta * global_dir * density_mod_factor)
                # Add back Field/Boids forces which were added to the unmodulated base force
                if "total_field_force" in locals():
                    combined_force += total_field_force
                if "total_sep_force" in locals():
                    combined_force += sep_w * total_sep_force  # Use dynamic weights
                if "total_cohesion_force" in locals():
                    combined_force += coh_w * total_cohesion_force
                if "total_align_force" in locals():
                    combined_force += align_w * total_align_force
                if force_analysis:
                    force_analysis["density_mod_applied_to"] = "influence_weights"
            elif force_analysis:
                force_analysis["density_mod_applied_to"] = "integrator"

        # --- Update analysis data dict ---
        if analysis_data is not None and force_analysis is not None:
            analysis_data.update(force_analysis)
            analysis_data["force_combined_norm_avg"] = torch.linalg.vector_norm(combined_force, dim=-1).mean().item()

        logger.debug(f"[Head {head_idx}] Force calculation complete.")
        return combined_force, density_mod_factor, state_out

    def _calculate_density_mod_factor(self, A: torch.Tensor, T: int, analysis_data: Optional[Dict] = None) -> torch.Tensor:
        """Calculates the density modulation factor based on assignments."""
        factor = torch.tensor(1.0, device=A.device, dtype=torch.float32)
        measure = self.config_force.get("density_measure", "none")
        if measure == "cluster_size":
            B, _, K_max = A.shape
            cluster_usage = A.detach().sum(dim=1)  # Sum over T -> (B, K_max)
            # Normalize usage per batch item relative to uniform?
            relative_density = cluster_usage / max(1, T)  # Fraction of time steps assigned (approx)
            sensitivity = self.config_force.get("density_sensitivity", 0.1)
            # Modulation decreases effect in dense areas: exp(-sens * density)
            modulation_per_cluster = torch.exp(-sensitivity * relative_density)  # (B, K_max) real
            # Get modulation factor per token based on assignment A
            factor = (A.detach() * modulation_per_cluster.unsqueeze(1)).sum(dim=-1, keepdim=True)  # (B, T, 1) real
            if analysis_data:
                analysis_data["density_mod_factor_avg"] = factor.mean().item()
        # elif measure == 'kde_stub': logger.warning("KDE density measure stub.")
        return factor.float().clamp(min=0.1, max=10.0)  # Ensure float and clamp range

    def extra_repr(self) -> str:
        return f"formal={self.use_formal_force}, boids={self.config_force.get('use_boids_rules', False)}, fields={self.config_force.get('use_interaction_fields', False)}, density={self.config_force.get('use_density_modulation', False)}"
