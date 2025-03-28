# src/pgs_net/config.py
"""
Defines the default configuration dictionary for the PGS-Net FFN module
and potentially helper functions for loading/merging configurations.
"""

import logging
import math
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

# Default Configuration Dictionary (Checkpoint 3.3)
DEFAULT_PGS_FFN_CONFIG: Dict[str, Any] = {
    # --- Architecture Settings ---
    "architecture": {
        "state_management": {
            "mode": "buffers",  # Options: 'none', 'buffers', 'external'
            "boids_alignment_state_type": "avg_update",  # 'avg_update', 'full_update' (only if mode='external')
        },
        "use_gradient_clipping": False,  # Enable clipping update vector norm within Integrator
        "gradient_clip_threshold": 1.0,  # Norm threshold for clipping
        "use_amp": False,  # Use Automatic Mixed Precision (torch.cuda.amp.autocast)
        "use_complex_representation": False,  # Use complex numbers for embeddings/operations
        "complex_projection_method": "linear",  # How to project real input to complex: 'linear', 'real_as_complex'
        "complex_output_method": "linear",  # How to project complex output to real: 'linear', 'real_part', 'magnitude'
        # Non-Locality Module
        "use_non_locality_module": False,
        "non_locality_type": "mha",  # 'mha', 'linear_attn_stub', 'gated_attn_stub'
        "non_locality_placement": "before",  # 'before', 'parallel', 'after' PGS FFN logic
        "non_locality_params": {"embed_dim": -1, "num_heads": -1, "dropout": 0.1},  # embed_dim/num_heads filled later if -1
        # Analysis / Debug
        "collect_analysis_data": False,  # Set to True to return detailed internal states dictionary
        "log_gradient_norms": False,  # Enable backward hooks to log gradient norms
        # Efficiency
        "use_conditional_computation": False,  # Enable basic token update gating
        "conditional_comp_threshold": 0.01,  # Threshold for force norm below which update is skipped
        # Parameter Sharing (Flags control sharing across heads)
        "param_sharing": {
            "share_centroids": False,
            "share_geometry_params": False,  # Shares tau, gate_logits, osc params (freq/phase/weight)
            "share_queen_weights": False,  # Shares global_weights_logits
            "share_force_params": False,  # Shares fitness_layer, charges
            "share_integrator_params": False,  # Shares decay_logits, gate_layer
        },
        # Regularization
        "regularization": {
            "use_orthogonal_centroids": False,  # Apply orthogonal regularization loss to centroids
            "orthogonal_strength": 0.001,
            "use_centroid_repulsion_loss": False,  # Apply centroid repulsion loss (alternative to formal force term)
            "use_spectral_norm_output": False,  # Apply spectral norm constraint to final output projection
        },
        # Normalization Type (applied in Integrator)
        "normalization_type": "rmsnorm",  # 'rmsnorm', 'adagroupnorm'
        "adagroupnorm_groups": 4,  # Number of groups if using AdaGroupNorm
    },
    # --- Geometry Settings ---
    "geometry": {
        "use_geometry_switching": True,  # Enable mixing multiple geometries
        "branches": ["euclidean", "hyperbolic", "fractal"],  # List of geometries to potentially use
        "force_branch": None,  # Force a single branch (e.g., "euclidean"), overrides switching
        # Fractal Settings
        "fractal_metric_type": "power_euclidean",  # 'power_euclidean', 'manhattan_power', 'box_counting_refined', 'advanced_stub'
        "fractal_alpha": 1.0,  # Exponent for power law distances
        "learnable_fractal_alpha": False,
        "fractal_params": {
            "num_scales": 5,
            "box_scale_factor": 1.5,
            "grid_bound": 2.0,
            "regression_mode": "standard",
        },  # Params for box counting
        # Similarity Temperature (Tau)
        "similarity_base_temp": 1.0,  # Initial/fixed base temperature 'tau' for similarities
        "learnable_similarity_temp": True,  # Learn tau per branch/globally
        # Hyperbolic Settings
        "hyperbolic_projection_eps": 1e-5,  # Epsilon for Poincaré ball projection
        "complex_hyperbolic_mode": "formal",  # How to handle complex+hyperbolic: 'formal' (Poincaré), 'approx_magnitude', 'default_euclidean', 'siegel_uhp_stub'
        "complex_hyperbolic_alt_mode": "approx_magnitude",  # Fallback if formal/selected fails or is stub
        # Oscillator Settings
        "use_oscillator_similarity": False,  # Add oscillator-based similarity branch
        "oscillator_params": {
            "learnable_frequency": True,
            "frequency_dim": 1,
            "learnable_phase": True,
            "phase_dim": 1,
            "phase_weight": 0.1,  # Initial/fixed weight for phase component
            "learnable_phase_weight": True,  # Make phase weight learnable
        },
    },
    # --- Clustering & Assignment Settings ---
    "clustering": {
        "max_clusters": 4,  # Maximum physical centroids per head
        # Assignment Temperature (softmax sharpness)
        "assignment_temp": 1.0,  # Initial/fixed temperature (can be meta-learned)
        "use_dynamic_assignment_temp": False,  # Enable annealing/scheduling
        "dynamic_temp_schedule": "annealing",  # 'annealing', 'loss_based_stub'
        "dynamic_temp_params": {"start": 1.0, "end": 0.1, "rate": 0.999, "max_epochs_for_schedule": 100},  # Params for schedule
        # Tunneling (Stochastic Jumps)
        "use_tunneling_assignment": False,  # Use STE to allow random jumps
        "tunneling_prob": 0.001,  # Probability per token
        # Dynamic K (Cluster Count Adaptation)
        "use_dynamic_k": False,  # Enable dynamic cluster count
        "dynamic_k_type": "split_merge",  # 'none', 'usage_based', 'split_merge'
        "dynamic_k_params": {
            "min_k": 2,
            "max_k_buffer": 0,  # Allow K to slightly exceed max_clusters temporarily
            "update_interval": 20,  # Update K logic every N steps
            "usage_ema_decay": 0.98,
            "prune_threshold_factor": 0.1,  # Prune if usage < factor * (1/CurrentK)
            "reactivation_threshold_factor": 0.5,  # Reactivate if usage > factor * (1/CurrentK)
            "split_variance_threshold_factor": 2.0,  # Split if variance > factor * AvgVariance
            "split_method": "perturb",  # 'perturb', 'pca_stub'
            "split_init_factor": 0.05,  # Perturbation factor for new centroids
            "merge_distance_threshold_factor": 0.15,  # Merge if dist < factor * AvgInterCentroidDist
            "merge_method": "average",  # 'average', 'keep_dominant'
        },
    },
    # --- Queen Computation Settings ---
    "queens": {
        "use_global_queen_momentum": True,  # Use EMA for global queen
        "momentum_decay": 0.9,  # Decay factor for EMA (can be meta-learned)
        "use_learnable_global_weights": True,  # Learn weights for combining local queens -> global queen
        # Auxiliary Losses (Only applied during training)
        "use_entropy_regularization": False,  # Penalize/encourage entropy of assignments or global weights
        "entropy_target": "assignments",  # 'assignments', 'global_weights'
        "entropy_weight": 0.001,
        "entropy_type": "maximize",  # 'maximize' (encourage exploration) or 'minimize' (encourage specialization)
        "use_diversity_aux_loss": True,  # Penalize cosine similarity between local queens
        "diversity_aux_loss_weight": 0.1,
    },
    # --- Token Update Force Settings ---
    "update_forces": {
        # Base Influence Weights (can be meta-learned)
        "local_influence_weight": 0.5,  # Alpha: Weight for local queen/influence direction
        "global_influence_weight": 0.3,  # Beta: Weight for global queen direction
        # Fitness Modulation (P factor)
        "use_fitness_modulation": False,  # Scale local influence by P = sigmoid(fit_local - fit_token)
        # Physics-Inspired Interaction Fields
        "use_interaction_fields": False,  # Add Coulomb/Gravity-like forces between tokens/centroids
        "interaction_field_type": "coulomb",  # 'coulomb'
        "interaction_field_strength": 0.01,
        "learnable_charges": False,  # Learn centroid charges (token charge assumed 1)
        # Density-Dependent Modulation (Quorum Sensing)
        "use_density_modulation": False,  # Modulate params based on local cluster density
        "density_modulation_target": ["decay"],  # List of params to modulate: 'decay', 'influence_weights'
        "density_measure": "cluster_size",  # 'cluster_size' (from assignments), 'kde_stub'
        "density_sensitivity": 0.1,  # How strongly density affects modulation (e.g., exp(-sens * density))
        # Boids Flocking Rules
        "use_boids_rules": False,  # Add Separation, Cohesion, Alignment forces
        "boids_separation_weight": 0.01,
        "boids_alignment_weight": 0.01,  # Alignment towards average *neighbor update direction*
        "boids_cohesion_weight": 0.005,  # Cohesion towards average *neighbor position*
        "boids_neighbor_k": 5,  # Number of neighbors to consider
        "boids_use_faiss": True,  # Use Faiss for neighbor search if available
        "boids_faiss_index_type": "IndexFlatL2",  # Faiss index type
        "boids_update_index_every": 1,  # Rebuild Faiss index frequency (0=once, 1=always, N=every N steps)
        # Formal Force Calculation (Overrides other forces if enabled)
        "use_formal_force": False,
        "formal_force_type": "potential_v2",  # 'none', 'simple_potential', 'potential_v2', 'lennard_jones_stub'
        "formal_force_params": {
            "attraction_centroid_weight": 1.0,  # Attraction to assigned centroid
            "attraction_global_weight": 0.1,  # Attraction to global queen
            # Centroid repulsion moved to architecture.regularization
            "repulsion_token_weight": 0.0001,  # Repulsion between tokens
            "repulsion_token_radius": 1.0,  # Radius for token repulsion
            "repulsion_potential_type": "smoothed_inverse_sq",  # 'smoothed_inverse_sq', 'lennard_jones_stub'
            "repulsion_smooth_eps": 1e-3,
            "alignment_weight": 0.0,  # Weight for alignment term (requires velocity state) - KEEP 0
        },
    },
    # --- Update Integration Settings ---
    "integration": {
        "use_token_momentum": False,  # Apply EMA momentum to the integrated update force
        "token_momentum_decay": 0.9,  # Can be meta-learned
        "use_learnable_decay": True,  # Master decay scaling (sigmoid(logit)) per head
        "fixed_decay_value": 0.9,  # Value if learnable_decay is False
        "use_gate": True,  # Sigmoid gate on the final update step, conditioned on token state
        "use_rms_norm": True,  # Apply RMSNorm (or selected norm) to the final update step
        "rms_norm_eps": 1e-8,
        "use_dropout": True,  # Apply Dropout to the final update step
        "dropout_rate": 0.1,
    },
    # --- Meta-Learning (Learning Hyperparameters) ---
    "meta_learning": {
        "use_meta_config": False,  # Enable meta-learning of specified hyperparameters
        "meta_learner_type": "advanced_hypernetwork",  # 'none', 'hypernetwork', 'advanced_hypernetwork', 'rl_stub'
        "meta_config_target_params": [
            "local_influence_weight",
            "global_influence_weight",
            "similarity_base_temp",
            "assignment_temp",
            "token_momentum_decay",
        ],  # List of keys in config to make dynamic
        "meta_context_sources": [
            "epoch_norm",
            "avg_input_norm",
            "loss_feedback",
            "avg_assignment_entropy",
        ],  # Info fed to meta-learner
        "hypernetwork_params": {
            "hidden_dims": [64, 32],
            "use_layer_norm": True,
            "use_context_embedding": True,
            "context_embedding_dim": 16,
            "context_feature_dims": {},
            "output_activation": "scaled_sigmoid_softplus",
        },
        "meta_target_param_ranges": {  # Target ranges for 'scaled_sigmoid_softplus' output activation
            "local_influence_weight": [0.0, 2.0],
            "global_influence_weight": [0.0, 2.0],
            "similarity_base_temp": [0.05, 5.0],
            "assignment_temp": [0.05, 5.0],
            "token_momentum_decay": [0.5, 0.999],
        },
    },
    # --- Cross-Layer Communication ---
    "cross_layer": {
        "use_cross_layer_comm": False,  # Enable passing info between PGS layers
        "cross_layer_type": "attention",  # 'none', 'conditional_input', 'attention', 'gating'
        "cross_layer_info": ["layer_output_tokens", "avg_global_queen"],  # List of keys to collect and pass
        "cross_layer_projection_dim": 64,  # Dimension for projected Keys/Values in attention, or projected info for gating/cond_input
        "cross_layer_params": {
            "num_attn_heads": 4,
            "kv_projection_method": "shared",
            "attention_variant": "standard_mha",
            "combine_method": "add_norm",
        },  # Specific params for method
    },
}


# --- Helper Functions (Optional: Load/Merge YAML/Dict configs) ---
def load_config(path: str) -> Dict[str, Any]:
    # Implementation using yaml.safe_load or json.load
    import yaml  # Requires PyYAML

    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {path}")
        # Could potentially merge with defaults here
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {path}: {e}")
        return {}


def merge_configs(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merges update dict into base dict."""
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
