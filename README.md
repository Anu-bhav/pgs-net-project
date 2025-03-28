# PGS-Net: PolyGeometric Swarm Network

## Overview

PGS-Net is a research project implementing a novel, highly configurable Feed-Forward Network (FFN) layer designed to replace the standard MLP sub-layer within Transformer architectures. Drawing inspiration from Swarm Intelligence and Geometric Deep Learning, PGS-Net models token interactions dynamically within adaptive geometric spaces.

Instead of static matrix multiplications, PGS-Net simulates a swarm where tokens (agents) cluster around learned centroids based on similarity computed using Euclidean, Hyperbolic (Real & Complex Poincaré), Fractal (Power-Law, Manhattan, Box-Counting), or Oscillator metrics. Emergent "queen" representations guide token updates through a rich suite of configurable, physics-inspired dynamics, including momentum, interaction fields, density modulation, Boids rules, and potential energy gradients.

This repository contains the full implementation (v3.3), including advanced adaptability features like dynamic cluster counts, meta-learned hyperparameters, cross-layer communication, parameter sharing, and conditional computation. It supports complex-valued representations, offers extensive analysis data collection capabilities, and includes visualization utilities.

**Current Status:** Feature implementation complete (Checkpoint 3.3). Ready for rigorous testing, hyperparameter optimization, and empirical evaluation across various tasks.

## Key Features

- **PolyGeometric Similarity:** Adaptively mixes or forces different geometric metrics for token-centroid similarity calculation.
  - Euclidean
  - Hyperbolic (Real Poincaré, Complex Poincaré, Magnitude Approximation)
  - Fractal (Power-Law L2/L1, Box-Counting Approximation, Advanced Stub)
  - Oscillator (Learnable Frequency & Phase)
- **Swarm Dynamics:**
  - Local & Global Queen computation with optional EMA momentum.
  - Configurable update forces: Local/Global influence, Fitness modulation, Interaction Fields (Coulomb), Boids rules (Separation, Cohesion, Alignment w/ Faiss option), Formal Potential Energy gradients.
- **Adaptability & Efficiency:**
  - **Dynamic K:** Cluster count adapts via Usage-Based Pruning/Reactivation or Split/Merge algorithms.
  - **Meta-Learning:** HyperNetwork learns key hyperparameters based on training context.
  - **Cross-Layer Communication:** Handler enables information flow (Avg Queen, Geo Weights, Tokens) between layers via Conditional Input or Attention.
  - **Parameter Sharing:** Configurable sharing of parameters (centroids, geometry, queen, force, integrator) across heads.
  - **Conditional Computation:** Skips updates for tokens with low force magnitude.
  - **Faiss Acceleration:** Optional GPU acceleration for Boids neighbor search.
  - **AMP Support:** Optional Automatic Mixed Precision via `torch.cuda.amp.autocast`.
- **Stability & Robustness:**
  - Integrated Update Vector Clipping.
  - Normalization Options (RMSNorm, AdaGroupNorm stub).
  - Regularization Options (Orthogonal Centroids, Centroid Repulsion).
  - Gradient Norm Logging via hooks.
- **Complex Number Support:** Option to perform computations using `torch.complex64`.
- **State Management:** Flexible options ('none', 'buffers', 'external') for handling recurrent state (momentum, Boids alignment).
- **Configuration Driven:** Highly modular architecture controlled via a detailed YAML/Python configuration dictionary.
- **Analysis & Visualization:** Extensive logging and collection of internal metrics into structured data, with utilities for processing and plotting using Pandas, Matplotlib, and Seaborn.

## Project Structure

pgs-net-project/
├── configs/ # Configuration files (base, experiments, ablations)
├── data/ # Data loading scripts (task-specific)
├── experiments/ # Main training & evaluation scripts (train.py, analyze_run.py)
├── notebooks/ # Jupyter notebooks for analysis & visualization
├── results/ # Output directory (logs, checkpoints, analysis data, plots) - gitignored
├── src/ # Source code for the pgs_net library
│ └── pgs_net/
│ ├── config.py
│ ├── pgs_ffn.py (Main Orchestrator)
│ ├── transformer_block.py
│ ├── modules/ # Core sub-modules (geometry, clustering, dynamics, etc.)
│ └── utils/ # Logging, analysis processing, visualization helpers
├── tests/ # Unit and integration tests
├── .gitignore
├── requirements.txt # Dependencies
└── README.md # This file

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd pgs-net-project
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    _Note: `requirements.txt` should include:_
    - `torch>=1.10` (adjust based on your CUDA version)
    - `numpy`
    - `pandas`
    - `matplotlib`
    - `seaborn`
    - `pyyaml` (for config loading)
    - _(Optional but Recommended)_ `umap-learn`, `scikit-learn` (for `projection_utils.py`)
    - _(Optional)_ `faiss-gpu` or `faiss-cpu` (for efficient Boids neighbor search) - Installation can be complex, see [Faiss documentation](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).
    - _(Optional)_ `pytest` (for running tests)
    - _(Optional)_ `jupyterlab` or `notebook` (for running analysis notebooks)
    - _(Optional)_ `hydra-core` (for advanced configuration management)

## Configuration

PGS-Net's behavior is controlled via a configuration dictionary, typically loaded from a YAML file (see `configs/`). The default structure and options are defined in `src/pgs_net/config.py`.

Key sections include:

- `architecture`: Core settings like complex numbers, state management, parameter sharing, AMP, conditional computation, regularization, normalization.
- `geometry`: Geometry branches, switching/forcing, parameters for each geometry type.
- `clustering`: Max clusters, assignment temperature, tunneling, dynamic K settings.
- `queens`: Momentum, weights, auxiliary losses.
- `update_forces`: Influence weights, enabling/params for fitness, fields, density, Boids, formal force.
- `integration`: Token momentum, decay, gate, norm, dropout, clipping.
- `meta_learning`: Enable meta-learner, specify type, targets, context, hypernetwork structure.
- `cross_layer`: Enable cross-layer comm, specify type, info sources, attention params.

Merge experiment-specific configurations with the `base_config.yaml`.

## Usage

**1. Training:**

- Modify or create a configuration YAML file in `configs/experiments/`.
- Update the `experiments/train.py` script:
  - Load the desired configuration.
  - Set up your specific dataset loader and model architecture (e.g., embedding layers, number of `TransformerBlock`s, final task head).
  - Instantiate the model, potentially passing the loaded `pgs_ffn_config` to each `TransformerBlock`.
  - Implement the training loop:
    - Handle data loading.
    - Pass necessary context (epoch, loss) to the model's forward pass if using meta-learning or dynamic temperature.
    - Manage state dictionaries if `state_management.mode == 'external'`.
    - Manage cross-layer information flow if `use_cross_layer_comm == True`.
    - Collect the `total_aux_loss` and add it (weighted) to the main loss before backpropagation.
    - Perform optimization step.
    - Optionally call `model.apply_constraints()` after optimizer step if needed by regularization.
    - Collect and save main metrics and `analysis_data` dictionary per step/epoch.
    - Log gradient norms using `model.get_gradient_log()` after `loss.backward()` if enabled.
- Run the training script:
  ```bash
  python experiments/train.py --config_path configs/experiments/my_experiment.yaml --output_dir results/my_experiment_run1
  ```

**2. Analysis & Visualization:**

- Use the `experiments/analyze_run.py` script or Jupyter notebooks in `notebooks/`.
- Load the saved `analysis_data.pkl` (or `analysis_summary.csv`) from a results directory.
- Use functions from `src/pgs_net/utils/analysis_processing.py` to process the data into a Pandas DataFrame.
- Use functions from `src/pgs_net/utils/viz_utils.py` (and potentially `projection_utils.py`) to generate plots:
  - `plot_training_curves`
  - `plot_geometry_analysis`
  - `plot_dynamics_analysis`
  - `plot_correlation_heatmap`
  - Project embeddings using `project_embeddings`, then plot with `plot_embedding_visualization`.

## Testing

Run tests using `pytest`:

```bash
pytest tests/
```
