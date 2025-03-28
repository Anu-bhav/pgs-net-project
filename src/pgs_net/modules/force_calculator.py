# src/pgs_net/modules/force_calculator.py
""" Computes the combined update force vector. """
import torch
import torch.nn as nn
import logging
import time
from typing import Dict, Optional, Tuple, Union
from ..config import DEFAULT_PGS_FFN_CONFIG
from .interfaces import FormalForceCalculator, NeighborSearch
from .complex_utils import ComplexLinear
from .neighbor_search import NaiveNeighborSearch, FaissNeighborSearch
from .formal_force import PotentialEnergyForceV2, PlaceholderFormalForce # Import implementations

logger = logging.getLogger(__name__)

# Helper for shared parameters
def get_shared_parameter(module, param_name, shared_params_dict, default_factory): # ... (implementation)

class UpdateForceCalculator(nn.Module):
    """
    Calculates the raw update force for each token by combining influences
    from local/global queens, optional physics fields, Boids rules, or formal potential gradients.
    """
    def __init__(self, d_head: int, num_heads: int, max_clusters: int, config: Dict, shared_params: Optional[Dict] = None):
        """
        Initializes the UpdateForceCalculator.

        Args:
            d_head (int): Dimension per head.
            num_heads (int): Total number of heads.
            max_clusters (int): Max physical clusters.
            config (Dict): Full PGS_FFN config.
            shared_params (Optional[Dict]): Dictionary for shared parameters.
        """
        super().__init__()
        self.config_force = config.get('update_forces', DEFAULT_PGS_FFN_CONFIG['update_forces'])
        self.arch_config = config.get('architecture', DEFAULT_PGS_FFN_CONFIG['architecture'])
        self.param_sharing_config = self.arch_config.get('param_sharing', {})
        self.shared_params = shared_params if shared_params is not None else {}

        self.is_complex = self.arch_config.get('use_complex_representation', False)
        self.dtype = torch.complex64 if self.is_complex else torch.float32
        self.d_head = d_head; self.num_heads = num_heads; self.max_clusters = max_clusters

        # --- Neighbor Search Module (for Boids) ---
        self.neighbor_search_module: Optional[NeighborSearch] = None
        if self.config_force.get('use_boids_rules', False):
             k = self.config_force.get('boids_neighbor_k', 5)
             use_faiss = self.config_force.get('boids_use_faiss', True) and faiss_available # Use Faiss if flag is True and library available
             if use_faiss:
                  try:
                      self.neighbor_search_module = FaissNeighborSearch(
                          k, self.d_head, self.is_complex,
                          index_type=self.config_force.get('boids_faiss_index_type', "IndexFlatL2"),
                          update_every=self.config_force.get('boids_update_index_every', 1),
                          use_gpu = True # Default to GPU Faiss if available
                      )
                  except Exception as e:
                      logger.error(f"Failed to init FaissNeighborSearch: {e}. Falling back to Naive.")
                      self.neighbor_search_module = NaiveNeighborSearch(k, self.is_complex) # Fallback
             else: # Use Naive
                 self.neighbor_search_module = NaiveNeighborSearch(k, self.is_complex)

        # --- Formal Force Calculator ---
        self.formal_force_calc: Optional[FormalForceCalculator] = None
        self.use_formal_force = self.config_force.get('use_formal_force', False)
        if self.use_formal_force:
             force_type = self.config_force.get('formal_force_type', 'none')
             if force_type == 'simple_potential': # Legacy name?
                 self.formal_force_calc = PotentialEnergyForceV2(config, self.is_complex, self.dtype) # Use V2 as simple
                 logger.warning("Formal force type 'simple_potential' used, mapping to PotentialEnergyForceV2.")
             elif force_type == 'potential_v2':
                 self.formal_force_calc = PotentialEnergyForceV2(config, self.is_complex, self.dtype)
             # Add elif for future types...
             else:
                  self.formal_force_calc = PlaceholderFormalForce(config, self.is_complex, self.dtype)
                  logger.warning(f"Using Placeholder Formal Force ({force_type} not impl).")
                  self.use_formal_force = False # Disable if using placeholder? Or let it return zero? Let it return zero.

        # --- Other Parameters / Layers ---
        # Fitness Layer
        if self.config_force.get('use_fitness_modulation', False):
            factory_fit = lambda: (ComplexLinear if self.is_complex else nn.Linear)(d_head, 1)
            share = self.param_sharing_config.get('share_force_params', False)
            self.fitness_layer = get_shared_parameter(self, "fitness_layer", self.shared_params, factory_fit) if share else factory_fit()
            logger.info(f"Fitness modulation enabled (Shared={share}).")

        # Interaction Field Charges
        if self.config_force.get('use_interaction_fields', False) and self.config_force.get('learnable_charges', False):
             factory_charge_head = lambda: nn.Parameter(torch.randn(num_heads, max_clusters))
             factory_charge_shared = lambda: nn.Parameter(torch.randn(max_clusters))
             share = self.param_sharing_config.get('share_force_params', False)
             if share:
                  self.centroid_charges = get_shared_parameter(self, "centroid_charges", self.shared_params, factory_charge_shared)
             else: self.centroid_charges = factory_charge_head()
             logger.info(f"Interaction fields enabled with learnable charges (Shared={share}).")
        elif self.config_force.get('use_interaction_fields', False):
            # Use fixed charges (buffer) - shape (H, K) to allow head-specific fixed charges if needed later
            self.register_buffer('centroid_charges_fixed', torch.ones(num_heads, max_clusters))
            logger.info("Interaction fields enabled with fixed charges (1.0).")


    def forward(self,
                x_h: torch.Tensor,
                A: torch.Tensor,
                local_queens: torch.Tensor,
                global_queen: torch.Tensor,
                centroids_h: torch.Tensor,
                head_idx: int,
                state_in: Optional[Dict] = None,
                dynamic_params: Optional[Dict] = None,
                analysis_data: Optional[Dict] = None,
                last_update_state: Optional[torch.Tensor] = None # State from Integrator (for Boids Alignment)
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Calculates the combined raw update force and density modulation factor.

        Args:
            x_h: Token embeddings for the head (B, T, D).
            A: Assignment matrix (B, T, K_max).
            local_queens: Local queens (B, K_max, D).
            global_queen: Global queen (B, D).
            centroids_h: Centroids for the head (K_max, D).
            head_idx: Current head index.
            state_in: Input state (rarely used here).
            dynamic_params: Dynamically learned hyperparameters.
            analysis_data: Dictionary to store analysis results.
            last_update_state: Output from previous step's integrator (B, T, D) or (1, D).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict]:
                - combined_force (B, T, D)
                - density_mod_factor (B, T, 1) or scalar tensor 1.0
                - state_out (typically empty for this module)
        """
        B, T, D = x_h.shape
        K_max = centroids_h.shape[0]
        force_analysis = {} if analysis_data is not None else None
        state_out: Dict[str, torch.Tensor] = {}
        logger.debug(f"[Head {head_idx}] Update Force Calculation started.")

        # --- Formal Force Calculation (Overrides others if enabled) ---
        if self.use_formal_force and self.formal_force_calc is not None:
             logger.debug("Using Formal Force Calculation.")
             # Pass necessary inputs
             combined_force = self.formal_force_calc.calculate_force(x_h, A, local_queens, global_queen, centroids_h)
             density_mod_factor = torch.tensor(1.0, device=x_h.device, dtype=torch.float32) # No density mod with formal force? Or apply after? Apply after.
             if force_analysis is not None:
                 norm_val = torch.linalg.vector_norm(combined_force, dim=-1).mean().item()
                 force_analysis["force_formal_norm_avg"] = norm_val
             # --- Calculate Density Mod Factor (Still useful?) ---
             if self.config_force.get('use_density_modulation', False):
                 density_mod_factor = self._calculate_density_mod_factor(A, T, analysis_data=force_analysis)
             return combined_force, density_mod_factor, state_out # Return early

        # --- Standard Force Pipeline ---
        # --- Get Effective Hyperparameters ---
        alpha = dynamic_params.get('local_influence_weight', self.config_force['local_influence_weight']) if dynamic_params else self.config_force['local_influence_weight']
        beta = dynamic_params.get('global_influence_weight', self.config_force['global_influence_weight']) if dynamic_params else self.config_force['global_influence_weight']
        if not isinstance(alpha, torch.Tensor): alpha = torch.tensor(alpha, device=x_h.device, dtype=torch.float32) # Use float for weights
        if not isinstance(beta, torch.Tensor): beta = torch.tensor(beta, device=x_h.device, dtype=torch.float32)

        # --- Base Local/Global Influence ---
        local_influence = torch.bmm(A, local_queens) # (B, T, D)
        local_dir = local_influence - x_h
        global_dir = global_queen.unsqueeze(1) - x_h # Broadcasts T dim

        # --- Fitness Modulation (P) ---
        if self.config_force.get('use_fitness_modulation', False) and hasattr(self, 'fitness_layer'):
             fit_token = self.fitness_layer(x_h); fit_local = self.fitness_layer(local_influence)
             if fit_token.is_complex(): fit_token = fit_token.real # Use real part for fitness scalar
             if fit_local.is_complex(): fit_local = fit_local.real
             P = torch.sigmoid(fit_local - fit_token) # (B, T, 1)
             local_dir = P * local_dir # Apply modulation
             if force_analysis is not None: force_analysis["fitness_mod_P_avg"] = P.mean().item()
        else: P = 1.0

        combined_force = (alpha * local_dir + beta * global_dir)
        if force_analysis is not None: # Log base component norms
             force_analysis["force_local_dir_norm_avg"] = torch.linalg.vector_norm(local_dir, dim=-1).mean().item()
             force_analysis["force_global_dir_norm_avg"] = torch.linalg.vector_norm(global_dir, dim=-1).mean().item()

        # --- Interaction Fields ---
        if self.config_force.get('use_interaction_fields', False):
             if hasattr(self, 'centroid_charges') or hasattr(self, 'centroid_charges_fixed'):
                  learnable = self.config_force.get('learnable_charges', False)
                  share = self.param_sharing_config.get('share_force_params', False)
                  if learnable: charges = self.centroid_charges if not share else self.centroid_charges # Access potentially shared param
                  else: charges = self.centroid_charges_fixed[head_idx] # Access buffer slice
                  token_charge = 1.0
                  strength = dynamic_params.get('interaction_field_strength', self.config_force['interaction_field_strength']) if dynamic_params else self.config_force['interaction_field_strength']

                  diff = x_h.unsqueeze(2) - centroids_h.detach().unsqueeze(0).unsqueeze(0) # (B, T, K, D) Treat centroids as fixed sources
                  dist = torch.linalg.vector_norm(diff, ord=2, dim=-1).clamp(min=1e-6) # (B, T, K) Real distance
                  field_force_contrib = torch.zeros_like(diff)
                  if self.config_force['interaction_field_type'] == 'coulomb':
                       # Force = Strength * q_tok * q_cen / dist^3 * (-diff)
                       force_mag_over_dist = (strength * token_charge * charges.view(1, 1, K)) / dist.pow(3) # (B, T, K)
                       field_force_contrib = -force_mag_over_dist.unsqueeze(-1) * diff # (B, T, K, D) complex if input is complex
                  # Add other field types later...

                  # Aggregate force based on assignment A (Treat A as fixed weights)
                  total_field_force = (A.detach().unsqueeze(-1) * field_force_contrib).sum(dim=2) # (B, T, D)
                  combined_force = combined_force + total_field_force
                  if force_analysis is not None: force_analysis["force_field_norm_avg"] = torch.linalg.vector_norm(total_field_force, dim=-1).mean().item()
             else: logger.warning("Interaction fields enabled but charges not found.")

        # --- Boids Rules ---
        if self.config_force.get('use_boids_rules', False) and self.neighbor_search_module is not None:
             logger.debug(f"[Head {head_idx}] Calculating Boids forces...")
             neighbor_indices, neighbor_vecs = self.neighbor_search_module.find_neighbors(x_h) # (B, T, k), (B, T, k, D)

             if neighbor_vecs is not None and neighbor_vecs.numel() > 0:
                  valid_neighbor_mask = (neighbor_indices != -1) # (B, T, k)
                  num_neighbors = valid_neighbor_mask.sum(dim=-1, keepdim=True).clamp(min=1) # (B, T, 1), avoid div by zero

                  # --- Separation ---
                  sep_w = dynamic_params.get('boids_separation_weight', self.config_force['boids_separation_weight']) if dynamic_params else self.config_force['boids_separation_weight']
                  total_sep_force = torch.zeros_like(x_h)
                  if sep_w > 1e-6:
                       diff_from_neighbors = x_h.unsqueeze(2) - neighbor_vecs # (B, T, k, D)
                       dist_sq = diff_from_neighbors.abs().pow(2).sum(dim=-1).clamp(min=1e-6) # (B, T, k)
                       # Force = - w * diff / dist^2 (mask invalid neighbors)
                       sep_force_per = - sep_w * diff_from_neighbors / dist_sq.unsqueeze(-1)
                       # Sum forces from valid neighbors only
                       total_sep_force = torch.where(valid_neighbor_mask.unsqueeze(-1), sep_force_per, torch.zeros_like(sep_force_per)).sum(dim=2) / num_neighbors # Average force
                       if force_analysis: force_analysis["force_boids_sep_norm_avg"] = torch.linalg.vector_norm(total_sep_force, dim=-1).mean().item()

                  # --- Cohesion ---
                  coh_w = dynamic_params.get('boids_cohesion_weight', self.config_force['boids_cohesion_weight']) if dynamic_params else self.config_force['boids_cohesion_weight']
                  total_cohesion_force = torch.zeros_like(x_h)
                  if coh_w > 1e-6:
                       # Average position of valid neighbors
                       avg_neighbor_pos = torch.where(valid_neighbor_mask.unsqueeze(-1), neighbor_vecs, torch.zeros_like(neighbor_vecs)).sum(dim=2) / num_neighbors
                       total_cohesion_force = coh_w * (avg_neighbor_pos - x_h) # Force towards avg pos
                       if force_analysis: force_analysis["force_boids_coh_norm_avg"] = torch.linalg.vector_norm(total_cohesion_force, dim=-1).mean().item()

                  # --- Alignment ---
                  align_w = dynamic_params.get('boids_alignment_weight', self.config_force['boids_alignment_weight']) if dynamic_params else self.config_force['boids_alignment_weight']
                  total_align_force = torch.zeros_like(x_h)
                  if align_w > 1e-6 and last_update_state is not None:
                      logger.debug("Calculating Boids Alignment force...")
                      try: # Gather neighbor's last updates
                          if last_update_state.shape == x_h.shape: # Full state (B, T, D) - External mode usually
                               B_idx = torch.arange(B, device=x_h.device).view(-1, 1, 1)
                               # Need to handle -1 indices safely
                               valid_indices = torch.where(valid_neighbor_mask, neighbor_indices, torch.zeros_like(neighbor_indices)) # Replace -1 with 0 temporarily
                               gathered_updates = last_update_state[B_idx, valid_indices] # Gather (B, T, k, D)
                               # Zero out contributions from invalid neighbors
                               avg_neighbor_update = torch.where(valid_neighbor_mask.unsqueeze(-1), gathered_updates, torch.zeros_like(gathered_updates)).sum(dim=2) / num_neighbors
                               total_align_force = align_w * avg_neighbor_update # Align towards average update direction
                               if force_analysis: force_analysis["force_boids_align_norm_avg"] = torch.linalg.vector_norm(total_align_force, dim=-1).mean().item()
                          elif (last_update_state.dim() == 3 and last_update_state.shape[0] == 1 and last_update_state.shape[2] == D): # Buffer mode (average) (1, 1, D)
                               avg_update_dir = last_update_state.expand_as(x_h) # Expand avg dir
                               total_align_force = align_w * avg_update_dir
                               if force_analysis: force_analysis["force_boids_align_norm_avg"] = torch.linalg.vector_norm(total_align_force, dim=-1).mean().item()
                               logger.debug("Using averaged last update state for Boids Alignment.")
                          else: logger.warning(f"Unexpected shape for Boids last_update_state: {last_update_state.shape}. Skipping Alignment.")
                      except Exception as e: logger.error(f"Failed Alignment force calculation: {e}", exc_info=True)
                  else: logger.debug("Skipping Boids Alignment (no state/zero weight).")

                  # Add Boids forces
                  combined_force = combined_force + total_sep_force + total_cohesion_force + total_align_force
             else: logger.debug("Skipping Boids forces (no neighbors found/returned).")
        # --- End Boids ---

        # --- Density Modulation Factor ---
        density_mod_factor = torch.tensor(1.0, device=x_h.device, dtype=torch.float32)
        if self.config_force.get('use_density_modulation', False):
            density_mod_factor = self._calculate_density_mod_factor(A, T, analysis_data=force_analysis)
            # Apply modulation to influence weights? Or return factor? Return factor.
            if 'influence_weights' in self.config_force.get('density_modulation_target', []):
                logger.debug("Applying density modulation to influence weights.")
                combined_force = (alpha * local_dir * density_mod_factor) + (beta * global_dir * density_mod_factor) # Recompute base force with modulation
                if force_analysis: force_analysis["density_mod_applied_to"] = 'influence_weights'
            elif force_analysis: force_analysis["density_mod_applied_to"] = 'integrator' # Default: integrator applies it

        # --- Update analysis data dict ---
        if analysis_data is not None and force_analysis is not None:
            analysis_data.update(force_analysis)
            analysis_data["force_combined_norm_avg"] = torch.linalg.vector_norm(combined_force, dim=-1).mean().item()

        logger.debug(f"[Head {head_idx}] Force calculation complete.")
        return combined_force, density_mod_factor, state_out


    def _calculate_density_mod_factor(self, A: torch.Tensor, T: int, analysis_data: Optional[Dict] = None) -> torch.Tensor:
        """ Calculates the density modulation factor based on assignments. """
        B, _, K_max = A.shape
        factor = torch.tensor(1.0, device=A.device, dtype=torch.float32)
        if self.config_force.get('density_measure') == 'cluster_size':
            cluster_usage = A.sum(dim=1) # Sum over T -> (B, K_max)
            # Normalize usage per batch item? Or globally? Per batch item.
            relative_density = cluster_usage / T # Fraction of time steps assigned (approx)
            # Could also use EMA usage from DynamicK? More stable but delayed. Let's use current batch usage.
            sensitivity = self.config_force.get('density_sensitivity', 0.1)
            # Modulation decreases effect in dense areas: exp(-sens * density)
            modulation_per_cluster = torch.exp(-sensitivity * relative_density) # (B, K_max) real
            # Get modulation factor per token based on assignment A
            # (B, T, K_max) * (B, 1, K_max) -> sum(K_max) -> (B, T)
            factor = (A * modulation_per_cluster.unsqueeze(1)).sum(dim=-1, keepdim=True) # (B, T, 1) real
            if analysis_data: analysis_data["density_mod_factor_avg"] = factor.mean().item()
        # elif self.config_force.get('density_measure') == 'kde_stub': logger.warning("KDE density measure stub.")
        return factor.float() # Ensure float