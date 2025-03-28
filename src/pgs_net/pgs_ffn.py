# src/pgs_net/pgs_ffn.py
"""Main PGS_FFN Orchestrator Module."""

import logging
import math
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEFAULT_PGS_FFN_CONFIG
from .modules.adapters import InputAdapter, OutputAdapter
from .modules.clustering_assignment import ClusteringAssignment
from .modules.complex_utils import ComplexLinear, ComplexRMSNorm
from .modules.cross_layer import CrossLayerHandler
from .modules.dynamic_k import SplitMergeDynamicKController, UsageBasedDynamicKController
from .modules.force_calculator import UpdateForceCalculator
from .modules.geometry_similarity import GeometrySimilarity
from .modules.integrator import UpdateIntegrator
from .modules.interfaces import DynamicKController, FormalForceCalculator, MetaConfigLearner, Regularization
from .modules.meta_config import AdvancedHyperNetworkMetaConfig
from .modules.non_locality import NonLocalityModule
from .modules.placeholders import PlaceholderDynamicKController, PlaceholderFormalForce, PlaceholderMetaConfig
from .modules.queen_computation import QueenComputation
from .modules.regularization import CentroidRepulsionRegularization, OrthogonalRegularization, SpectralNormConstraint

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


class PGS_FFN(nn.Module):
    """
    PolyGeometric Swarm Network Feed-Forward Layer.
    Replaces standard Transformer FFN with adaptive, multi-geometric, swarm-inspired dynamics.
    """

    def __init__(self, d_model: int, num_heads: int, config: Optional[Dict] = None):
        """
        Initializes the PGS_FFN layer.

        Args:
            d_model (int): Dimension of the input/output embeddings.
            num_heads (int): Number of parallel swarm simulation heads.
            config (Optional[Dict]): Configuration dictionary overriding defaults.
                                      Defaults to DEFAULT_PGS_FFN_CONFIG.

        """
        super().__init__()
        if config is None:
            config = DEFAULT_PGS_FFN_CONFIG
        # Deepcopy to prevent modification of the default config object
        self.config = deepcopy(config)

        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            logger.warning(f"d_model ({d_model}) not divisible by num_heads ({num_heads}). Adjusting num_heads.")
            # Find closest divisor or default to 1? Let's default to 1 head.
            # Find factors and pick closest? Simpler: Use 1 head if not divisible.
            self.num_heads = 1
            # Alternatively, adjust d_head and maybe add projection? Adjusting num_heads is simpler.
            # self.num_heads = next((h for h in range(num_heads, 0, -1) if d_model % h == 0), 1)
            logger.warning(f"Using num_heads = {self.num_heads}")

        self.d_head = d_model // self.num_heads

        # --- Core Architecture Settings ---
        self.arch_config = self.config.get("architecture", {})
        self.param_sharing_config = self.arch_config.get("param_sharing", {})
        self.is_complex = self.arch_config.get("use_complex_representation", False)
        self.dtype = torch.complex64 if self.is_complex else torch.float32
        self.state_mode = self.arch_config.get("state_management", {}).get("mode", "buffers")
        self.collect_analysis = self.arch_config.get("collect_analysis_data", False)
        self.log_gradients = self.arch_config.get("log_gradient_norms", False)
        self.use_amp = self.arch_config.get("use_amp", False)

        logger.info(
            f"Initializing PGS_FFN: Dm={d_model}, H={self.num_heads}, Dh={self.d_head}, Complex={self.is_complex}, State={self.state_mode}, AMP={self.use_amp}"
        )

        # Dictionary to hold shared parameters across heads/modules if needed
        self.shared_params: Dict[str, nn.Parameter] = {}

        # --- Initialize Parameters (Centroids - potentially shared) ---
        max_clusters = self.config.get("clustering", {}).get("max_clusters", 4)
        factory_centroids_head = lambda: nn.Parameter(
            torch.randn(self.num_heads, max_clusters, self.d_head, dtype=self.dtype) * 0.1
        )  # Smaller init?
        factory_centroids_shared = lambda: nn.Parameter(torch.randn(max_clusters, self.d_head, dtype=self.dtype) * 0.1)
        if self.param_sharing_config.get("share_centroids", False):
            logger.info("Sharing centroids across heads.")
            # Register the shared centroids directly in PGS_FFN
            self.centroids_shared = get_shared_parameter(self, "centroids", self.shared_params, factory_centroids_shared)
            # Individual modules will access self.centroids_shared via parent or passed arg
        else:
            self.centroids = factory_centroids_head()  # Head-specific (H, K, D)
        logger.info(
            f"Centroids initialized (Shared={self.param_sharing_config.get('share_centroids', False)}). Shape: {self.centroids_shared.shape if hasattr(self, 'centroids_shared') else self.centroids.shape}"
        )

        # --- Meta Learner Placeholder ---
        self.meta_learner: Optional[MetaConfigLearner] = None
        if self.config.get("meta_learning", {}).get("use_meta_config", False):
            meta_type = self.config.get("meta_learning", {}).get("meta_learner_type", "none")
            if meta_type == "advanced_hypernetwork":
                self.meta_learner = AdvancedHyperNetworkMetaConfig(d_model, self.config)
            elif meta_type != "none":
                self.meta_learner = PlaceholderMetaConfig(d_model, self.config)
                logger.warning(f"Using Placeholder Meta Config ({meta_type} not impl).")
            if self.meta_learner:
                self.meta_learner.set_global_config(self.config)  # Inject global config if needed by learner

        # --- Cross-Layer Handler ---
        self.cross_layer_handler: Optional[CrossLayerHandler] = None
        if self.config.get("cross_layer", {}).get("use_cross_layer_comm", False):
            self.cross_layer_handler = CrossLayerHandler(d_model, self.d_head, self.num_heads, self.config)

        # --- Initialize Sub-Modules (Pass shared_params dict and full config) ---
        device = (
            next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        )  # Get device AFTER params are created
        self.input_adapter = InputAdapter(self.d_head, self.config)
        self.geometry_similarity = GeometrySimilarity(self.d_head, self.config, self.shared_params)
        self.clustering_assignment = ClusteringAssignment(
            self.config, device, self.dtype
        )  # Needs device/dtype for dynamic K state
        self.queen_computation = QueenComputation(self.d_head, self.num_heads, max_clusters, self.config, self.shared_params)
        self.update_force_calculator = UpdateForceCalculator(
            self.d_head, self.num_heads, max_clusters, self.config, self.shared_params
        )
        self.update_integrator = UpdateIntegrator(self.d_head, self.num_heads, self.config, self.shared_params)
        self.output_adapter = OutputAdapter(self.d_head, self.config)

        # --- Non-Locality Module ---
        self.non_locality_module: Optional[NonLocalityModule] = None
        if self.arch_config.get("use_non_locality_module", False):
            # Fill in embed_dim/num_heads if needed based on d_model
            nl_params = self.arch_config.get("non_locality_params", {})
            if nl_params.get("embed_dim", -1) == -1:
                nl_params["embed_dim"] = d_model
            if nl_params.get("num_heads", -1) == -1:
                nl_params["num_heads"] = self.num_heads  # Match main heads? Or default? Use main heads.
            self.arch_config["non_locality_params"] = nl_params  # Update config dict
            self.non_locality_module = NonLocalityModule(d_model, self.config)

        # --- Final Output Projection ---
        self.out_proj = nn.Linear(d_model, d_model)
        # Apply spectral norm constraint if configured
        if self.arch_config.get("regularization", {}).get("use_spectral_norm_output", False):
            try:
                self.out_proj = torch.nn.utils.parametrizations.spectral_norm(self.out_proj)
                logger.info("Applied spectral norm to output projection.")
            except Exception as e:
                logger.error(f"Failed to apply spectral norm to out_proj: {e}")

        # --- Regularization Modules (Loss terms applied in forward) ---
        self.regularizers = nn.ModuleList()
        reg_config = self.arch_config.get("regularization", {})
        if reg_config.get("use_orthogonal_centroids", False):
            strength = reg_config.get("orthogonal_strength", 0.01)
            self.regularizers.append(OrthogonalRegularization(strength=strength))
        if reg_config.get("use_centroid_repulsion_loss", False):
            # Get strength from formal force params for consistency? Or separate? Separate.
            strength = reg_config.get("centroid_repulsion_strength", 0.001)  # Add new config key
            eps = self.config.get("update_forces", {}).get("formal_force_params", {}).get("repulsion_smooth_eps", 1e-3)
            self.regularizers.append(CentroidRepulsionRegularization(strength=strength, eps=eps))
        # Add other regularizers here...

        # --- Gradient Logging Hooks ---
        self.grad_hooks = []
        self._grad_norms_log: Dict[str, float] = {}  # Internal storage for logged norms
        if self.log_gradients:
            self._register_gradient_hooks()

        logger.info("PGS_FFN Sub-modules initialized.")

    def _register_gradient_hooks(self):
        """Registers backward hooks to log gradient norms."""
        # Clear previous hooks if any
        self.remove_gradient_hooks()
        self._grad_norms_log.clear()

        def hook_fn(grad: Optional[torch.Tensor], name: str):
            if grad is not None:
                norm = torch.linalg.vector_norm(grad.detach()).item()
                self._grad_norms_log[name] = norm
                # logger.debug(f"Grad Norm - {name}: {norm:.3g}") # Logging here is too verbose
            else:
                self._grad_norms_log[name] = 0.0

        # Register hooks for key parameters
        param_map = {}
        if hasattr(self, "centroids_shared"):
            param_map["centroids_shared"] = self.centroids_shared
        elif hasattr(self, "centroids"):
            param_map["centroids_all"] = self.centroids

        if hasattr(self.geometry_similarity, "gate_logits_param"):
            param_map["geo_mix_logits"] = self.geometry_similarity.gate_logits_param
        if hasattr(self.geometry_similarity, "tau_param"):
            param_map["geo_temps"] = self.geometry_similarity.tau_param
        if hasattr(self.geometry_similarity, "fractal_alpha") and isinstance(
            self.geometry_similarity.fractal_alpha, nn.Parameter
        ):
            param_map["geo_fractal_alpha"] = self.geometry_similarity.fractal_alpha
        # Add hooks for oscillator params if learnable...

        if (
            hasattr(self.queen_computation, "effective_global_weights_logits")
            and self.queen_computation.effective_global_weights_logits is not None
        ):
            param_map["queen_global_logits"] = self.queen_computation.effective_global_weights_logits

        if (
            hasattr(self.update_integrator, "effective_decay_logits")
            and self.update_integrator.effective_decay_logits is not None
        ):
            param_map["integrator_decay_logits"] = self.update_integrator.effective_decay_logits
        if hasattr(self.update_integrator, "effective_gate_layer") and self.update_integrator.effective_gate_layer is not None:
            param_map["integrator_gate_layer_w"] = self.update_integrator.effective_gate_layer.weight
            if self.update_integrator.effective_gate_layer.bias is not None:
                param_map["integrator_gate_layer_b"] = self.update_integrator.effective_gate_layer.bias

        # Add hooks for fitness layer, charges etc. if learnable...

        for name, param in param_map.items():
            if param is not None and param.requires_grad:
                logger.debug(f"Registering grad hook for: {name}")
                handle = param.register_hook(lambda grad, n=name: hook_fn(grad, n))
                self.grad_hooks.append(handle)

        logger.info(f"Registered {len(self.grad_hooks)} gradient logging hooks.")

    def get_gradient_log(self) -> Dict[str, float]:
        """Returns the collected gradient norms after backward pass."""
        # Returns a copy
        return getattr(self, "_grad_norms_log", {}).copy()

    def remove_gradient_hooks(self):
        """Removes registered hooks."""
        for handle in self.grad_hooks:
            handle.remove()
        self.grad_hooks = []
        if hasattr(self, "_grad_norms_log"):
            self._grad_norms_log.clear()
        # logger.info("Removed gradient logging hooks.")

    def get_dynamic_params_for_head(self, head_idx: int, context: Dict) -> Dict:
        """Calls meta-learner and extracts params relevant for this head."""
        if self.meta_learner is None:
            return {}
        all_dynamic_params = self.meta_learner.get_dynamic_config(context)
        # Currently, all meta-learned params apply globally or per-head logic is within module
        # TODO: Add logic here if meta-learner outputs per-head params directly
        return all_dynamic_params

    # --- Forward Pass ---
    def forward(
        self,
        x: torch.Tensor,
        state_in: Optional[Dict] = None,
        epoch: Optional[int] = None,
        loss_feedback: Optional[torch.Tensor] = None,
        cross_layer_input: Optional[Dict] = None,
        max_epochs: Optional[int] = None,  # Needed for epoch norm context
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict], Optional[Dict], Optional[Dict]]:
        """
        Performs the forward pass of the PGS_FFN layer.

        Args:
            x (torch.Tensor): Input tensor (B, T, D_model).
            state_in (Optional[Dict]): Input state for 'external' mode.
            epoch (Optional[int]): Current epoch number.
            loss_feedback (Optional[torch.Tensor]): Recent loss value for meta-learner context.
            cross_layer_input (Optional[Dict]): Information from the previous layer.
            max_epochs(Optional[int]): Total epochs for context normalization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[Dict], Optional[Dict], Optional[Dict]]:
                - final_output (B, T, D_model)
                - total_aux_loss (scalar float tensor)
                - state_out (dictionary containing state for 'external' mode, else None)
                - analysis_data (dictionary with detailed metrics if enabled, else None)
                - cross_layer_output (dictionary with info for next layer if enabled, else None)

        """
        fwd_start_time = time.time()
        B, T, D_model_in = x.shape
        if D_model_in != self.d_model:  # Check input dim
            logger.error(f"Input dimension {D_model_in} does not match model dimension {self.d_model}.")
            # Try to project? Or raise error? Raise error for clarity.
            raise ValueError(f"Input dimension mismatch: Expected {self.d_model}, Got {D_model_in}")

        # --- Setup ---
        logger.debug(
            f"PGS_FFN Forward: Input={x.shape}, StateMode={self.state_mode}, Train={self.training}, Complex={self.is_complex}"
        )
        analysis_data: Optional[Dict] = (
            {"global": {}, "heads": [{} for _ in range(self.num_heads)]} if self.collect_analysis else None
        )
        state_out: Optional[Dict] = {} if self.state_mode == "external" else None
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)

        # --- Meta Config Update ---
        dynamic_params_all = {}
        if self.meta_learner is not None:
            # Build context state - needs avg entropy from *previous* step? Tricky. Use current for now.
            avg_entropy_approx = (
                np.nanmean([h.get("assignment_entropy", np.nan) for h in analysis_data["heads"]]) if analysis_data else 0.0
            )
            context = {
                "epoch": epoch,
                "max_epochs": max_epochs,
                "loss_feedback": loss_feedback,
                "avg_input_norm": torch.linalg.vector_norm(x, dim=-1).mean().item() / math.sqrt(self.d_model),
                "avg_assignment_entropy": avg_entropy_approx,
            }
            dynamic_params_all = self.get_dynamic_params_for_head(-1, context)
            if analysis_data and dynamic_params_all:
                analysis_data["global"]["meta_params"] = {k: v.item() for k, v in dynamic_params_all.items()}

        # --- Non-Locality (Before) ---
        x_intermediate = x
        nl_placement = self.arch_config.get("non_locality_placement", "none")
        if self.non_locality_module is not None and nl_placement == "before":
            if self.is_complex:
                logger.warning("Applying Non-locality MHA before complex projection - operates on real input.")
            x_intermediate = self.non_locality_module(x)  # Assumes non_locality works on D_model
            if analysis_data:
                analysis_data["global"]["non_locality_output_norm_before"] = (
                    torch.linalg.vector_norm(x_intermediate, dim=-1).mean().item()
                )

        # --- Prepare Head Inputs ---
        x_heads_real = x_intermediate.view(B, T, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        x_heads = self.input_adapter(x_heads_real)  # Potentially complex (B, H, T, D_head)

        # --- Inject Cross-Layer Info ---
        x_heads_conditioned = x_heads
        if self.cross_layer_handler is not None and cross_layer_input is not None:
            # Handler expects (B, T, Dm) for attention/gating, but (B, H, T, Dh) for cond_input on heads.
            # Let's assume handler's inject method handles the required input shape based on its type.
            # Passing x_intermediate (B, T, Dm) for attention/gating, x_heads for cond_input? Complex.
            # Simplification: Assume cross-layer operates *before* head split for attn/gate, handled above.
            # Assume conditional_input operates *after* head split.
            if self.cross_layer_handler.method == "conditional_input":
                x_heads_conditioned = self.cross_layer_handler.inject_input_info(x_heads, cross_layer_input)
            # else: Injection handled before head split (requires moving non-locality call too)

        # --- Process Heads ---
        processed_heads: List[torch.Tensor] = []
        all_head_state_out: Dict[str, Any] = {}  # For external mode state collection
        head_compute_times: List[float] = []
        collected_global_queens: List[torch.Tensor] = []  # For cross-layer output

        # --- AMP Context for Head Loop ---
        amp_enabled = self.use_amp and x.is_cuda
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            for h in range(self.num_heads):
                head_start_time = time.time()
                x_h = x_heads_conditioned[:, h, :, :]  # Use potentially conditioned input
                centroids_h = (
                    self.centroids_shared if self.param_sharing_config.get("share_centroids", False) else self.centroids[h]
                )
                head_analysis = analysis_data["heads"][h] if analysis_data else None
                state_in_h = state_in.get(f"head_{h}") if state_in is not None and self.state_mode == "external" else None
                state_out_h = {} if self.state_mode == "external" else None  # Collect state per head

                # Apply head-specific dynamic params if meta-learner supports it
                dynamic_params_h = dynamic_params_all  # Assume global for now

                # --- Pipeline ---
                try:
                    sim = self.geometry_similarity(x_h, centroids_h, analysis_data=head_analysis, dynamic_params=dynamic_params_h)
                    # Pass x_h for dynamic K variance calculation
                    A = self.clustering_assignment(
                        sim,
                        head_idx=h,
                        epoch=epoch,
                        centroids=centroids_h,
                        x_h=x_h,
                        analysis_data=head_analysis,
                        dynamic_params=dynamic_params_h,
                    )
                    queen_state_in = state_in_h.get("queen_comp") if state_in_h else None
                    local_q, global_q, aux_q, state_q = self.queen_computation(
                        x_h, A, centroids_h, head_idx=h, state_in=queen_state_in, dynamic_params=dynamic_params_h
                    )
                    if state_q:
                        state_out_h["queen_comp"] = state_q  # Store module state
                    if self.cross_layer_handler is not None and "global_queen" in self.cross_layer_handler.required_info:
                        collected_global_queens.append(global_q)

                    force_state_in = state_in_h.get("force_calc") if state_in_h else None
                    integ_state_in = state_in_h.get("integrator") if state_in_h else None
                    # Retrieve last update state needed for Boids Alignment
                    last_update_state_for_boids = integ_state_in.get(f"boids_last_update_h{h}") if integ_state_in else None
                    base_force, density_mod, state_force = self.update_force_calculator(
                        x_h,
                        A,
                        local_q,
                        global_q,
                        centroids_h,
                        head_idx=h,
                        state_in=force_state_in,
                        dynamic_params=dynamic_params_h,
                        analysis_data=head_analysis,
                        last_update_state=last_update_state_for_boids,
                    )
                    if state_force:
                        state_out_h["force_calc"] = state_force

                    x_out_h, aux_int, state_int = self.update_integrator(
                        x_h,
                        base_force,
                        density_mod,
                        head_idx=h,
                        state_in=integ_state_in,
                        dynamic_params=dynamic_params_h,
                        analysis_data=head_analysis,
                    )
                    if state_int:
                        state_out_h["integrator"] = state_int

                    # Accumulate outputs
                    processed_heads.append(x_out_h)
                    total_aux_loss += aux_q.float() + aux_int.float()  # Ensure float aux losses
                    if state_out_h:
                        all_head_state_out[f"head_{h}"] = state_out_h

                except Exception as e:
                    logger.error(f"Error processing Head {h}: {e}", exc_info=True)
                    # Handle error: skip head? return zero output? Return input? Use input.
                    processed_heads.append(x_h)  # Pass input through on error for this head
                    total_aux_loss += 0.0  # No aux loss contribution

                head_compute_times.append(time.time() - head_start_time)
                # logger.debug(f"[Head {h}] Processing complete ({head_compute_times[-1]:.4f} sec).")

        # --- End AMP context ---

        # --- Combine Heads ---
        if len(processed_heads) != self.num_heads:  # Check if any heads failed
            logger.error("Number of processed heads does not match expected number. Output may be invalid.")
            # Fallback: return input x? Requires careful handling downstream.
            # Let's try to proceed with potentially fewer heads combined.
            if not processed_heads:
                return x, total_aux_loss, state_out, analysis_data, None  # Return original input if all fail

        try:
            output_combined = torch.stack(processed_heads, dim=1).permute(0, 2, 1, 3).reshape(B, T, self.d_model)
        except Exception as e:
            logger.error(f"Failed to stack/reshape processed heads: {e}. Returning original input.", exc_info=True)
            return x, total_aux_loss, state_out, analysis_data, None

        # --- Adapt Output & Non-Locality (Parallel/After) ---
        output_adapted = self.output_adapter(output_combined)  # Back to real if needed
        final_output = output_adapted
        if self.non_locality_module is not None and nl_placement != "before":
            if self.is_complex:
                logger.warning("Applying Non-locality MHA after complex processing - operates on real output.")
            non_local_out = self.non_locality_module(output_adapted)  # Assumes MHA operates on D_model
            if nl_placement == "parallel":
                final_output = output_adapted + non_local_out
            elif nl_placement == "after":
                final_output = non_local_out
            if analysis_data:
                analysis_data["global"]["non_locality_output_norm_after"] = (
                    torch.linalg.vector_norm(final_output, dim=-1).mean().item()
                )

        # --- Final Projection ---
        try:
            final_output = self.out_proj(final_output)
        except Exception as e:
            logger.error(f"Output projection failed: {e}")
            return x  # Fallback

        # --- Add Explicit Regularization Losses ---
        if self.training and hasattr(self, "regularizers") and self.regularizers:
            reg_loss = torch.tensor(0.0, device=final_output.device, dtype=torch.float32)
            logger.debug("Applying explicit regularization losses...")
            centroids_param = self.centroids_shared if self.param_sharing_config.get("share_centroids", False) else self.centroids
            for reg_module in self.regularizers:
                term_loss = 0.0
                try:
                    if isinstance(reg_module, (OrthogonalRegularization, CentroidRepulsionRegularization)):
                        if centroids_param.dim() == 3:
                            term_loss = (
                                sum(reg_module(centroids_param[h].detach()) for h in range(self.num_heads)) / self.num_heads
                            )  # Detach centroids for loss calc? No, need gradient w.r.t them.
                        elif centroids_param.dim() == 2:
                            term_loss = reg_module(centroids_param)
                    # Add other regularizer types here
                    reg_loss += term_loss
                    logger.debug(f"Regularization Term ({type(reg_module).__name__}): {term_loss.item():.4g}")
                except Exception as e:
                    logger.error(f"Regularization module {type(reg_module).__name__} failed: {e}")

            total_aux_loss += reg_loss
            if analysis_data:
                analysis_data["global"]["regularization_loss"] = reg_loss.item()

        # --- Cross-Layer Output Generation ---
        cross_layer_output = None
        if self.cross_layer_handler is not None:
            cross_layer_output = self.cross_layer_handler.collect_output_info(
                analysis_data, collected_global_queens, layer_output_tokens=final_output
            )

        # --- Logging & Return ---
        total_forward_time = time.time() - fwd_start_time
        logger.info(
            f"PGS_FFN Forward Completed ({total_forward_time:.4f}s). Out={final_output.shape}, AuxLoss={total_aux_loss.item():.4f}"
        )
        if analysis_data:
            analysis_data["global"]["total_forward_time_sec"] = total_forward_time
            analysis_data["global"]["avg_head_time_sec"] = (
                sum(head_compute_times) / len(head_compute_times) if head_compute_times else 0
            )
            analysis_data["global"]["total_aux_loss"] = total_aux_loss.item()
            if self.log_gradients:
                analysis_data["global"]["gradients"] = self.get_gradient_log()  # Store logged gradients

        # Ensure aux loss is float scalar
        final_aux_loss = total_aux_loss.float().squeeze()
        if final_aux_loss.dim() != 0:
            final_aux_loss = final_aux_loss.mean()  # Ensure scalar

        return final_output, final_aux_loss, state_out, analysis_data, cross_layer_output

    @torch.no_grad()
    def apply_constraints(self):
        """Apply constraints (e.g., from regularization modules) after optimizer step."""
        if hasattr(self, "regularizers") and self.regularizers:
            logger.debug("Applying model constraints...")
            for reg_module in self.regularizers:
                try:
                    # Apply constraints registered within the module (e.g., SpectralNormConstraint)
                    reg_module.apply_constraints(self)  # Pass self? Or iterate modules? Iterate.
                    for submodule in self.modules():  # Apply to relevant submodules if needed
                        reg_module.apply_constraints(submodule)
                except Exception as e:
                    logger.error(f"Failed to apply constraint {type(reg_module).__name__}: {e}")

    def __del__(self):
        """Ensure gradient hooks are removed when object is deleted."""
        self.remove_gradient_hooks()

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_heads={self.num_heads}, complex={self.is_complex}, state={self.state_mode}"
