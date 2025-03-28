# src/pgs_net/modules/geometry_similarity.py
"""Computes adaptive multi-geometric similarity."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, Any, Optional, List, Union
from ..config import DEFAULT_PGS_FFN_CONFIG  # Import default for structured access
from .complex_utils import ComplexLinear  # Assuming complex_utils is at the same level

logger = logging.getLogger(__name__)


# Helper to get shared parameters
def get_shared_parameter(
    module: nn.Module, param_name: str, shared_params_dict: Dict[str, nn.Parameter], default_factory
) -> nn.Parameter:
    """Gets or creates a shared parameter and registers it locally."""
    if param_name not in shared_params_dict:
        shared_params_dict[param_name] = default_factory()
        logger.debug(f"Created shared parameter: {param_name} with shape {shared_params_dict[param_name].shape}")
    # Register the shared parameter in the current module so it's included in state_dict etc.
    # Use a unique name to avoid conflicts if multiple modules share the same base name
    shared_attr_name = f"_shared_{param_name}"
    if not hasattr(module, shared_attr_name):
        setattr(module, shared_attr_name, shared_params_dict[param_name])
    return shared_params_dict[param_name]


class GeometrySimilarity(nn.Module):
    """
    Computes similarity between tokens and centroids using multiple, potentially
    mixed or forced, geometric spaces (Euclidean, Hyperbolic, Fractal, Oscillator).
    Handles real and complex representations, parameter sharing, and dynamic parameters.
    """

    def __init__(self, d_head: int, config: Dict[str, Any], shared_params: Optional[Dict[str, nn.Parameter]] = None):
        """
        Initializes the GeometrySimilarity module.

        Args:
            d_head (int): The dimension per head.
            config (Dict[str, Any]): The main PGS_FFN configuration dictionary.
            shared_params (Optional[Dict[str, nn.Parameter]]): Dictionary holding shared parameters.
        """
        super().__init__()
        self.config = config.get("geometry", DEFAULT_PGS_FFN_CONFIG["geometry"])
        self.arch_config = config.get("architecture", DEFAULT_PGS_FFN_CONFIG["architecture"])
        self.cluster_config = config.get("clustering", DEFAULT_PGS_FFN_CONFIG["clustering"])
        self.param_sharing_config = self.arch_config.get("param_sharing", DEFAULT_PGS_FFN_CONFIG["architecture"]["param_sharing"])
        self.shared_params = shared_params if shared_params is not None else {}

        self.is_complex = self.arch_config.get("use_complex_representation", False)
        self.dtype = torch.complex64 if self.is_complex else torch.float32
        logger.info(
            f"Initializing GeometrySimilarity: Complex={self.is_complex}, ShareParams={self.param_sharing_config.get('share_geometry_params', False)}, Branches={self.config['branches']}, Force={self.config['force_branch']}"
        )

        self.d_head = d_head
        self.eps = self.config.get("hyperbolic_projection_eps", 1e-5)

        # --- Determine Active Branches ---
        self.active_branches: List[str] = []
        self.branch_indices: Dict[str, int] = {}
        idx = 0
        available_branches = self.config.get("branches", [])
        for branch in available_branches:
            compatible = True
            if branch == "fractal" and self.config.get("fractal_metric_type") == "advanced_stub":
                logger.warning("Advanced Fractal metric is a stub.")
            if branch == "hyperbolic" and self.is_complex and self.config.get("complex_hyperbolic_mode") == "siegel_uhp_stub":
                logger.warning("Siegel UHP complex hyperbolic is a stub.")
            # Add more checks if needed...
            if compatible:
                self.active_branches.append(branch)
                self.branch_indices[branch] = idx
                idx += 1
            else:
                logger.warning(f"Skipping incompatible/stub branch '{branch}' for complex={self.is_complex}")
        self.num_active_branches = len(self.active_branches)
        logger.info(f"Active geometry branches: {self.active_branches}")
        if not self.active_branches and self.config.get("force_branch") is None:
            raise ValueError("No active geometry branches compatible with configuration.")

        max_clusters = self.cluster_config.get("max_clusters", 4)

        # --- Oscillator Params ---
        self.osc_params = self.config.get("oscillator_params", {})
        if "oscillator" in self.active_branches and self.config.get("use_oscillator_similarity", False):
            logger.info("Initializing Oscillator parameters.")
            freq_dim = self.osc_params.get("frequency_dim", 1)
            phase_dim = self.osc_params.get("phase_dim", 1)
            factory_freq = lambda: nn.Parameter(torch.randn(max_clusters, freq_dim))
            factory_phase = lambda: nn.Parameter(torch.randn(max_clusters, phase_dim))
            factory_phase_w = lambda: nn.Parameter(torch.tensor(float(self.osc_params.get("phase_weight", 0.1))))
            learnable_freq = self.osc_params.get("learnable_frequency", False)
            learnable_phase = self.osc_params.get("learnable_phase", False)
            learnable_phase_w = self.osc_params.get("learnable_phase_weight", True)
            share_geom = self.param_sharing_config.get("share_geometry_params", False)

            if learnable_freq:
                freq_input_dim = 2 * self.d_head if self.is_complex else self.d_head
                self.token_freq_proj = nn.Linear(freq_input_dim, freq_dim)
                if share_geom:
                    self.centroid_freq = get_shared_parameter(self, "centroid_freq", self.shared_params, factory_freq)
                else:
                    self.centroid_freq = factory_freq()
                logger.info(f"Oscillator using learnable frequencies (Dim={freq_dim}), Shared={share_geom}")
            if learnable_phase:
                phase_input_dim = 2 * self.d_head if self.is_complex else self.d_head
                self.token_phase_proj = nn.Linear(phase_input_dim, phase_dim)
                if share_geom:
                    self.centroid_phase = get_shared_parameter(self, "centroid_phase", self.shared_params, factory_phase)
                else:
                    self.centroid_phase = factory_phase()
                if learnable_phase_w:
                    if share_geom:
                        self.osc_phase_weight = get_shared_parameter(
                            self, "osc_phase_weight", self.shared_params, factory_phase_w
                        )
                    else:
                        self.osc_phase_weight = factory_phase_w()
                else:
                    self.register_buffer("osc_phase_weight_fixed", torch.tensor(float(self.osc_params.get("phase_weight", 0.1))))
                logger.info(
                    f"Oscillator using learnable phases (Dim={phase_dim}) with {'learnable' if learnable_phase_w else 'fixed'} weight, Shared={share_geom and learnable_phase_w}"
                )

        # --- Fractal Alpha ---
        alpha_val = float(self.config.get("fractal_alpha", 1.0))
        learnable_alpha = self.config.get("learnable_fractal_alpha", False)
        share_geom = self.param_sharing_config.get("share_geometry_params", False)
        if learnable_alpha:
            factory_alpha = lambda: nn.Parameter(torch.tensor(alpha_val))
            if share_geom:
                self.fractal_alpha = get_shared_parameter(self, "fractal_alpha", self.shared_params, factory_alpha)
            else:
                self.fractal_alpha = factory_alpha()
        else:
            self.register_buffer("fractal_alpha_fixed", torch.tensor(alpha_val))
        # Add getter method for consistent access
        logger.info(f"Fractal alpha: {alpha_val}, Learnable={learnable_alpha}, Shared={share_geom and learnable_alpha}")

        # --- Temperature (tau) ---
        num_temp_params = (
            self.num_active_branches
            if self.config.get("use_geometry_switching", True) and self.config.get("force_branch") is None
            else 1
        )
        self.tau_param: Optional[nn.Parameter] = None
        self.tau_buffer: Optional[torch.Tensor] = None
        if num_temp_params > 0:
            base_temp_val = self.config.get("similarity_base_temp", 1.0)
            factory_tau = lambda: nn.Parameter(torch.ones(num_temp_params) * base_temp_val)
            learnable_temp = self.config.get("learnable_similarity_temp", True)
            share_geom = self.param_sharing_config.get("share_geometry_params", False)

            if learnable_temp:
                if share_geom:
                    self.tau_param = get_shared_parameter(self, "similarity_tau", self.shared_params, factory_tau)
                else:
                    self.tau_param = factory_tau()
            else:  # Fixed tau
                tau_buffer_val = torch.ones(num_temp_params) * base_temp_val
                # Register buffer only if not shared, otherwise shared param exists but requires_grad=False
                if not share_geom:
                    self.register_buffer("tau_buffer", tau_buffer_val)
                else:  # Access shared param but ensure it's not trained
                    self.tau_param = get_shared_parameter(self, "similarity_tau", self.shared_params, factory_tau)
                    if self.tau_param.requires_grad:
                        logger.warning(
                            "Shared similarity_tau parameter requires_grad=True but config learnable_similarity_temp=False. Setting requires_grad=False."
                        )
                        self.tau_param.requires_grad_(False)
            logger.info(
                f"Similarity temperature (tau) params: {num_temp_params}, learnable={learnable_temp}, shared={share_geom and learnable_temp}"
            )
        else:  # No active branches? Should not happen based on earlier check, but have fallback
            self.register_buffer("tau_buffer", torch.tensor([self.config.get("similarity_base_temp", 1.0)]))

        # --- Mixing Weights (gate_logits) ---
        self.gate_logits_param: Optional[nn.Parameter] = None
        if (
            self.config.get("use_geometry_switching", True)
            and self.num_active_branches > 1
            and self.config.get("force_branch") is None
        ):
            factory_logits = lambda: nn.Parameter(torch.ones(self.num_active_branches))
            share_geom = self.param_sharing_config.get("share_geometry_params", False)
            if share_geom:
                self.gate_logits_param = get_shared_parameter(self, "geometry_gate_logits", self.shared_params, factory_logits)
            else:
                self.gate_logits_param = factory_logits()
            logger.info(f"Using learnable mixing weights for {self.num_active_branches} branches (Shared={share_geom}).")

    # --- Property getters for consistent access ---
    @property
    def effective_fractal_alpha(self) -> torch.Tensor:
        if hasattr(self, "fractal_alpha"):
            return self.fractal_alpha
        elif hasattr(self, "fractal_alpha_fixed"):
            return self.fractal_alpha_fixed
        else:
            return torch.tensor(1.0)  # Fallback

    @property
    def effective_osc_phase_weight(self) -> torch.Tensor:
        if hasattr(self, "osc_phase_weight"):
            return self.osc_phase_weight.clamp(min=0)  # Ensure non-negative
        elif hasattr(self, "osc_phase_weight_fixed"):
            return self.osc_phase_weight_fixed
        else:
            return torch.tensor(0.0)  # Default if not configured

    def get_effective_tau(self, dynamic_params: Optional[Dict] = None) -> torch.Tensor:
        """Gets the current effective tau, potentially overridden by meta-learning."""
        current_tau = self.tau_param if hasattr(self, "tau_param") else self.tau_buffer
        base_temp_override = dynamic_params.get("similarity_base_temp") if dynamic_params else None

        if base_temp_override is not None and current_tau is not None:
            current_mean = current_tau.mean()
            scale_factor = base_temp_override / current_mean.clamp(min=1e-6)
            effective_tau = (current_tau * scale_factor).clamp(min=1e-6)
            # logger.debug(f"GeoSim: Using dynamic base_temp {base_temp_override.item():.3f} (Scale {scale_factor.item():.3f}) -> Eff.Tau Mean: {effective_tau.mean().item():.3f}")
            return effective_tau
        elif current_tau is not None:
            return current_tau.clamp(min=1e-6)
        else:  # Fallback
            return torch.tensor([1.0], device=next(self.parameters()).device)

    def get_effective_logits(self, dynamic_params: Optional[Dict] = None) -> Optional[torch.Tensor]:
        """Gets the current effective gate logits, potentially overridden."""
        # TODO: Implement dynamic override for logits if meta-learning targets them
        return self.gate_logits_param if hasattr(self, "gate_logits_param") else None

    # --- Distance/Similarity Implementations ---
    def _complex_poincare_distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Computes distance in the complex Poincaré disk model. Assumes inputs are projected."""
        # Formula: d(u,v) = arcosh(1 + 2 * |u-v|^2 / ((1-|u|^2)(1-|v|^2)))
        uv_diff_sq = (u.unsqueeze(2) - v.unsqueeze(0).unsqueeze(0)).abs().pow(2).sum(dim=-1)  # |u-v|^2 summed over D -> (B,T,K)
        u_norm_sq = u.abs().pow(2).sum(dim=-1, keepdim=True).clamp(max=1.0 - self.eps)  # |u|^2 (B,T,1)
        v_norm_sq = v.abs().pow(2).sum(dim=-1).clamp(max=1.0 - self.eps)  # |v|^2 (K,)

        denom = (1.0 - u_norm_sq) * (1.0 - v_norm_sq.unsqueeze(0).unsqueeze(0))  # (B, T, K)
        denom = denom.clamp(min=self.eps)  # Avoid division by zero or negative values

        cosh_arg = (1.0 + 2.0 * uv_diff_sq / denom).clamp(min=1.0 + self.eps)  # Ensure >= 1 + eps

        # arcosh(x) = log(x + sqrt(x^2 - 1)) - numerically stable version
        distance = torch.log(cosh_arg + torch.sqrt(cosh_arg.pow(2) - 1.0).clamp(min=self.eps))
        return distance  # Real distance

    def _real_poincare_dist_sq(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Returns Squared Euclidean distance ||u-v||^2 for use in the real arcosh formula."""
        # u: (B, T, D), v: (K, D)
        return (u.unsqueeze(2) - v.unsqueeze(0).unsqueeze(0)).pow(2).sum(dim=-1)  # (B, T, K)

    def _project_to_poincare_real(self, x: torch.Tensor) -> torch.Tensor:
        """Projects real points outside the ball onto the boundary."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        max_norm = 1.0 - self.eps
        # Calculate scale factor: < 1 only if norm > max_norm
        scale = torch.clamp(norm, max=max_norm) / (norm + self.eps)
        return x * scale

    def _project_to_poincare_complex(self, z: torch.Tensor) -> torch.Tensor:
        """Projects complex points outside the unit disk onto the boundary."""
        norm = z.abs()  # Magnitude |z| shape (..., D) or (...) if norm applied? No, magnitude element-wise? No, vector norm.
        vec_norm = torch.linalg.vector_norm(z, ord=2, dim=-1, keepdim=True)  # Norm over last dim -> (..., 1)
        max_norm = 1.0 - self.eps
        clipped_norm = torch.clamp(vec_norm, max=max_norm)
        # Scale factor: clipped_norm / norm. Is 1 if norm <= max_norm.
        scale = clipped_norm / (vec_norm + self.eps)  # Shape (..., 1)
        return z * scale  # Apply scale (broadcasts over D)

    def _euclidean_sim(self, X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        logger.debug("Calculating Euclidean similarity...")
        if self.is_complex:
            # Complex dot product: sum(x * y.conj())
            # Use einsum for clarity: 'btd,kd->btk' (b=batch, t=time, d=dim, k=clusters)
            sim = torch.einsum("btd,kd->btk", X, centroids.conj())
            return sim.real  # Use real part as similarity metric
        else:
            # Real dot product: X @ C.T
            return torch.matmul(X, centroids.t())

    def _hyperbolic_sim(self, X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        logger.debug(f"Calculating Hyperbolic similarity (Complex={self.is_complex})...")
        if self.is_complex:
            mode = self.config.get("complex_hyperbolic_mode", "formal")
            logger.debug(f"Complex hyperbolic mode: {mode}")
            if mode == "formal":
                X_proj = self._project_to_poincare_complex(X)
                centroids_proj = self._project_to_poincare_complex(centroids)
                distance = self._complex_poincare_distance(X_proj, centroids_proj)
                return -distance
            elif mode == "approx_magnitude":
                X_mag = X.abs()
                centroids_mag = centroids.abs()  # Element-wise magnitudes
                logger.warning("Using approximate complex hyperbolic distance based on magnitudes (Treating as real vectors).")
                # Use real Poincaré distance on magnitudes treated as real vectors
                X_mag_proj = self._project_to_poincare_real(X_mag)
                centroids_mag_proj = self._project_to_poincare_real(centroids_mag)
                dist_sq = self._real_poincare_dist_sq(X_mag_proj, centroids_mag_proj)
                # Calculate arcosh based on real formula using L2 norms of magnitudes
                u_norm_sq = X_mag_proj.pow(2).sum(-1, keepdim=True).clamp(max=1.0 - self.eps)
                v_norm_sq = centroids_mag_proj.pow(2).sum(-1).clamp(max=1.0 - self.eps)
                denom = (1.0 - u_norm_sq) * (1.0 - v_norm_sq.unsqueeze(0).unsqueeze(0))
                cosh_arg = (1.0 + 2.0 * dist_sq / denom.clamp(min=self.eps)).clamp(min=1.0 + self.eps)
                hyper_dist = torch.log(cosh_arg + torch.sqrt(cosh_arg.pow(2) - 1.0).clamp(min=self.eps))
                return -hyper_dist
            else:  # 'default_euclidean' or fallback
                logger.debug("Complex+Hyperbolic defaulting to Complex Euclidean distance.")
                diff = X.unsqueeze(2) - centroids.unsqueeze(0).unsqueeze(0)
                dist = torch.linalg.vector_norm(diff, ord=2, dim=-1)  # L2 norm magnitude
                return -dist
        else:  # Real Poincaré distance
            X_proj = self._project_to_poincare_real(X)
            centroids_proj = self._project_to_poincare_real(centroids)
            dist_sq = self._real_poincare_dist_sq(X_proj, centroids_proj)  # ||u-v||^2
            # Arcosh calculation consistent with complex formal definition structure
            u_norm_sq = X_proj.pow(2).sum(-1, keepdim=True).clamp(max=1.0 - self.eps)
            v_norm_sq = centroids_proj.pow(2).sum(-1).clamp(max=1.0 - self.eps)
            denom = (1.0 - u_norm_sq) * (1.0 - v_norm_sq.unsqueeze(0).unsqueeze(0))
            cosh_arg = (1.0 + 2.0 * dist_sq / denom.clamp(min=self.eps)).clamp(min=1.0 + self.eps)
            hyper_dist = torch.log(cosh_arg + torch.sqrt(cosh_arg.pow(2) - 1.0 + self.eps))  # Add eps inside sqrt
            return -hyper_dist

    def _fractal_sim_base(self, X: torch.Tensor, centroids: torch.Tensor, ord: Union[int, float, str]) -> torch.Tensor:
        """Base for fractal power law similarity using specified norm."""
        diff = X.unsqueeze(2) - centroids.unsqueeze(0).unsqueeze(0)  # (B, T, K, D)
        # Use torch.linalg.vector_norm for robust norm calculation
        dist = torch.linalg.vector_norm(diff, ord=ord, dim=-1).clamp(min=self.eps)  # Real distance (B, T, K)
        alpha = self.effective_fractal_alpha  # Use property getter
        return -(dist.pow(alpha))

    def _fractal_sim(self, X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        logger.debug("Calculating Fractal similarity (Power Euclidean)...")
        return self._fractal_sim_base(X, centroids, ord=2)  # L2 norm

    def _manhattan_power_sim(self, X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        logger.debug("Calculating Fractal similarity (Manhattan Power)...")
        return self._fractal_sim_base(X, centroids, ord=1)  # L1 norm

    def _box_counting_refined_sim(self, X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """Refined Box Counting approximation."""
        logger.debug("Calculating Fractal similarity (Box Counting Refined)...")
        B, T, D = X.shape
        K = centroids.shape[0]
        sim = torch.zeros(B, T, K, device=X.device, dtype=torch.float32)
        X_real = X.real if self.is_complex else X
        centroids_real = centroids.real if self.is_complex else centroids

        # Params from config
        f_params = self.config.get("fractal_params", {})
        num_scales = f_params.get("num_scales", 5)
        scale_factor = f_params.get("box_scale_factor", 1.5)
        bound = f_params.get("grid_bound", 2.0)
        max_range = 2 * bound
        regression_mode = f_params.get("regression_mode", "standard")

        if bound <= 0:
            logger.error("Box counting grid_bound must be positive.")
            return sim.float()

        # Precompute scales/box sizes
        scales, box_sizes = [], []
        current_box_size = max_range
        for _ in range(num_scales):
            if current_box_size < 1e-6:
                break
            scales.append(math.log(1.0 / current_box_size))
            box_sizes.append(current_box_size)
            current_box_size /= scale_factor
        if len(scales) < 2:
            logger.warning("Box counting needs at least 2 scales.")
            return sim.float()
        scales_t = torch.tensor(scales, device=X.device, dtype=torch.float32)
        box_sizes_t = torch.tensor(box_sizes, device=X.device, dtype=torch.float32)

        # Assign tokens to nearest centroid (hard assignment)
        diffs = X_real.unsqueeze(2) - centroids_real.unsqueeze(0).unsqueeze(0)
        dists_sq = diffs.pow(2).sum(-1)
        hard_assignments = torch.argmin(dists_sq, dim=-1)  # (B, T)

        estimated_dims = torch.ones(K, device=X.device, dtype=torch.float32)  # Default dim
        for k in range(K):
            mask_k = hard_assignments == k
            N_k = mask_k.sum()
            if N_k < 5:
                continue  # Skip if too few points
            tokens_k = X_real[mask_k]  # (N_k, D)

            # Box counting (using unique heuristic - potentially slow/memory intensive)
            tokens_shifted = tokens_k + bound  # Shift to [0, max_range]
            indices = (tokens_shifted.unsqueeze(0) / box_sizes_t.view(-1, 1, 1)).long()  # (num_scales, N_k, D)

            counts = []
            valid_scales_idx = []
            for s in range(len(scales)):
                try:
                    # Use unique on flattened coordinates for potentially better scaling than tuple conversion
                    # Combine D dimensions into a single large integer (requires bounded integer coords)
                    # This hash needs to be robust. Alternative: unique rows directly.
                    unique_coords = torch.unique(indices[s], dim=0)
                    count = unique_coords.shape[0]
                    if count > 0:
                        counts.append(math.log(count))
                        valid_scales_idx.append(s)
                except Exception as e:
                    logger.warning(f"Box counting unique failed: {e}")
                    break

            # Linear regression
            if len(valid_scales_idx) > 1:
                valid_scales_t = scales_t[valid_scales_idx]
                counts_t = torch.tensor(counts, device=X.device, dtype=torch.float32)
                X_reg = torch.stack([valid_scales_t, torch.ones_like(valid_scales_t)], dim=1)
                try:
                    if regression_mode == "robust_stub":
                        logger.warning("Robust regression stub.")
                    coeffs, _ = torch.linalg.lstsq(X_reg, counts_t.unsqueeze(1))
                    dimension = max(0.0, min(coeffs[0, 0].item(), D))  # Clamp dimension
                    estimated_dims[k] = dimension
                except Exception as e:
                    logger.warning(f"Fractal dim regression failed: {e}")

        # Assign negative dimension as similarity
        final_sim = torch.zeros(B, T, K, device=X.device, dtype=torch.float32)
        # Use advanced indexing: index into estimated_dims using hard_assignments, then scatter
        dims_for_tokens = estimated_dims[hard_assignments]  # (B, T)
        final_sim.scatter_(dim=-1, index=hard_assignments.unsqueeze(-1), src=-dims_for_tokens.unsqueeze(-1))

        return final_sim.float()

    def _advanced_fractal_sim(self, X, centroids):
        logger.warning("Advanced fractal similarity (e.g., Correlation Dimension) not implemented. Using power Euclidean.")
        return self._fractal_sim(X, centroids)

    def _siegel_uhp_stub_sim(self, X, centroids):
        """Placeholder for Siegel Upper Half-Plane hyperbolic similarity."""
        logger.warning("Siegel Upper Half-Plane similarity not implemented. Using fallback.")
        alt_mode = self.config.get("complex_hyperbolic_alt_mode", "default_euclidean")
        if self.is_complex:
            if alt_mode == "approx_magnitude":
                # Re-call _hyperbolic_sim which handles this mode
                return self._hyperbolic_sim(X, centroids)
            else:  # Default Euclidean
                return self._euclidean_sim(X, centroids)
        else:  # Real input, Siegel doesn't apply
            logger.error("Siegel UHP stub called with real inputs. Falling back to Euclidean.")
            return self._euclidean_sim(X, centroids)

    def _oscillator_sim(self, X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """Calculates similarity based on learned frequency and phase."""
        logger.debug("Calculating Oscillator similarity...")
        similarity = torch.zeros(*X.shape[:-1], centroids.shape[0], device=X.device, dtype=torch.float32)
        has_freq = hasattr(self, "centroid_freq") and hasattr(self, "token_freq_proj")
        has_phase = hasattr(self, "centroid_phase") and hasattr(self, "token_phase_proj")

        # Prepare input for projections (real/imag concatenated if complex)
        proj_input = torch.cat([X.real, X.imag], dim=-1) if self.is_complex else X

        # Frequency Component
        if has_freq:
            token_freq = self.token_freq_proj(proj_input)  # (B, T, freq_dim)
            centroid_freq = self.centroid_freq  # (K, freq_dim) (Handles shared via property access)
            freq_diff = token_freq.unsqueeze(2) - centroid_freq.unsqueeze(0).unsqueeze(0)
            freq_dist_sq = freq_diff.pow(2).sum(dim=-1)  # (B, T, K)
            similarity = similarity - freq_dist_sq
            logger.debug(f"Oscillator freq component calculated.")
        else:
            logger.debug("Oscillator skipping frequency component.")

        # Phase Component
        if has_phase:
            # Project to get phase value/vector, use tanh to keep bounded [-pi, pi]
            token_phase = torch.tanh(self.token_phase_proj(proj_input)) * math.pi
            centroid_phase = torch.tanh(self.centroid_phase) * math.pi  # Handles shared
            phase_diff = token_phase.unsqueeze(2) - centroid_phase.unsqueeze(0).unsqueeze(0)
            # Cosine similarity averaged over phase dimensions
            phase_sim = torch.cos(phase_diff).mean(dim=-1)
            # Combine using effective weight
            phase_weight = self.effective_osc_phase_weight  # Use property getter
            similarity = similarity + phase_weight * phase_sim
            logger.debug(f"Oscillator phase component calculated (Weight: {phase_weight.item():.3f}).")
        else:
            logger.debug("Oscillator skipping phase component.")

        if not has_freq and not has_phase:
            logger.warning("Oscillator branch active but no freq/phase configured. Returning zero.")

        return similarity.float()  # Ensure float output

    def forward(
        self,
        X: torch.Tensor,
        centroids: torch.Tensor,
        analysis_data: Optional[Dict] = None,
        dynamic_params: Optional[Dict] = None,
    ) -> torch.Tensor:
        """Computes the final similarity score based on configuration."""
        # --- Get Effective Parameters ---
        effective_tau = self.get_effective_tau(dynamic_params)
        effective_logits = self.get_effective_logits(dynamic_params)

        # --- Setup Branch Map ---
        branch_map = {
            "euclidean": self._euclidean_sim,
            "hyperbolic": self._hyperbolic_sim,
            "fractal": self._fractal_sim,  # Default fractal
            "oscillator": self._oscillator_sim,
        }
        # Overwrite fractal based on config
        fractal_type = self.config.get("fractal_metric_type", "power_euclidean")
        if fractal_type == "manhattan_power":
            branch_map["fractal"] = self._manhattan_power_sim
        elif fractal_type == "box_counting_refined":
            branch_map["fractal"] = self._box_counting_refined_sim
        elif fractal_type == "advanced_stub":
            branch_map["fractal"] = self._advanced_fractal_sim
        # Handle complex hyperbolic stub/fallback
        complex_mode = self.config.get("complex_hyperbolic_mode", "formal")
        alt_mode = self.config.get("complex_hyperbolic_alt_mode", "none")
        if (
            self.is_complex
            and complex_mode != "formal"
            and complex_mode != "approx_magnitude"
            and complex_mode != "default_euclidean"
        ):
            if complex_mode == "siegel_uhp_stub":
                branch_map["hyperbolic"] = self._siegel_uhp_stub_sim
            else:
                branch_map["hyperbolic"] = self._alternative_complex_hyperbolic_sim  # Generic fallback

        # --- Compute Similarities ---
        target_branches = self.active_branches if self.config["force_branch"] is None else [self.config["force_branch"]]
        logger.debug(f"Target geometry branches for computation: {target_branches}")
        computed_sims = {}
        raw_sims_analysis = {}
        active_branch_names_for_mix = []
        sim_compute_times = {}
        for branch_name in self.active_branches:  # Compute all active branches
            if branch_name in branch_map:
                t_start = time.time()
                try:
                    sim_val = branch_map[branch_name](X, centroids)
                    computed_sims[branch_name] = sim_val
                    sim_compute_times[branch_name] = time.time() - t_start
                    if analysis_data is not None:
                        raw_sims_analysis[f"sim_{branch_name}_raw_mean"] = sim_val.mean().item()  # Log mean raw sim
                    if branch_name in target_branches:
                        active_branch_names_for_mix.append(branch_name)
                except Exception as e:
                    logger.error(f"Error computing similarity for branch '{branch_name}': {e}", exc_info=True)
            elif branch_name in target_branches:
                logger.error(f"Target branch '{branch_name}' has no valid compute method.")

        if not computed_sims or not active_branch_names_for_mix:
            raise ValueError("No similarities were computed successfully for target branches.")
        if analysis_data:
            analysis_data["geometry_compute_times_sec"] = sim_compute_times

        # --- Combine or Select ---
        num_mix_branches = len(active_branch_names_for_mix)
        mix_weights_analysis = None
        tau_indices = list(range(effective_tau.shape[0])) if effective_tau.dim() > 0 else [0]

        if self.config["force_branch"] is not None:
            branch_name = self.config["force_branch"]
            if branch_name not in computed_sims:
                raise ValueError(f"Forced branch '{branch_name}' computation failed.")
            logger.debug(f"Using forced branch: {branch_name} with temp {effective_tau[tau_indices[0]].item():.3f}")
            combined_sim = computed_sims[branch_name] / effective_tau[tau_indices[0]]
            mix_weights_analysis = torch.tensor([1.0], device=X.device)
        elif self.config["use_geometry_switching"] and num_mix_branches > 1 and effective_logits is not None:
            logger.debug(f"Mixing {num_mix_branches} branches: {active_branch_names_for_mix}")
            sims_list, active_temps, active_logits_list = [], [], []
            active_branch_final_names = []
            temp_idx_map = {name: i for i, name in enumerate(self.active_branches)}  # Map original active branches to tau indices

            for branch_name in active_branch_names_for_mix:
                sims_list.append(computed_sims[branch_name])
                # Find correct temp index based on original active_branches order
                temp_idx = temp_idx_map.get(branch_name, 0) if len(tau_indices) > 1 else 0
                active_temps.append(effective_tau[tau_indices[temp_idx]])
                # Find correct logit index based on order in active_branch_names_for_mix
                logit_idx = active_branch_names_for_mix.index(branch_name)
                active_logits_list.append(effective_logits[logit_idx])  # Assumes logits match active_branch_names_for_mix order
                active_branch_final_names.append(branch_name)

            sims = torch.stack(sims_list, dim=-1)
            active_temps_tensor = torch.stack(active_temps)
            active_logits_tensor = torch.stack(active_logits_list)
            sims_scaled = sims / active_temps_tensor  # Apply temperature
            mix_weights = F.softmax(active_logits_tensor, dim=0)  # (num_mix_branches,)
            mix_weights_analysis = mix_weights.detach()
            logger.debug(f"Mixing weights ({active_branch_final_names}): {mix_weights_analysis.cpu().numpy()}")
            combined_sim = (sims_scaled * mix_weights.view(1, 1, 1, -1)).sum(dim=-1)
        elif num_mix_branches == 1:
            branch_name = active_branch_names_for_mix[0]
            logger.debug(f"Using single active branch: {branch_name} with temp {effective_tau[tau_indices[0]].item():.3f}")
            combined_sim = computed_sims[branch_name] / effective_tau[tau_indices[0]]
            mix_weights_analysis = torch.tensor([1.0], device=X.device)
        else:
            raise ValueError("Invalid configuration/state for geometry mixing/selection.")

        if combined_sim.is_complex():  # Ensure float output
            logger.warning("Similarity output was complex, taking real part.")
            combined_sim = combined_sim.real

        # --- Collect Analysis Data ---
        if analysis_data is not None:
            analysis_data.update(raw_sims_analysis)  # Store raw means
            analysis_data["sim_combined_mean"] = combined_sim.mean().item()
            analysis_data["similarity_temps_effective"] = effective_tau.detach()
            if mix_weights_analysis is not None:
                analysis_data["geometry_mix_weights"] = mix_weights_analysis
            # Store branch names corresponding to weights/temps
            analysis_data["active_geometry_branches"] = active_branch_names_for_mix if num_mix_branches > 1 else target_branches

        logger.debug("Geometry similarity calculation complete.")
        return combined_sim
