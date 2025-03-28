# Create the main project directory and navigate into it
mkdir pgs-net-project
cd pgs-net-project

# Create configuration directories and placeholder files
mkdir -p configs/experiments configs/ablations
touch configs/base_config.yaml
touch configs/experiments/nlp_complex_splitmerge_meta.yaml
touch configs/experiments/vision_real_boids_faiss.yaml
touch configs/ablations/no_dynamic_k.yaml
touch configs/ablations/no_formal_force.yaml

# Create data directory
mkdir data
touch data/.gitkeep # Keep directory in git even if empty initially

# Create experiments directory and scripts
mkdir experiments
touch experiments/train.py
touch experiments/analyze_run.py
touch experiments/evaluate.py

# Create notebooks directory
mkdir notebooks
touch notebooks/01_Config_Exploration.ipynb
touch notebooks/02_Geometry_Analysis.ipynb
touch notebooks/03_Swarm_Dynamics_Viz.ipynb
touch notebooks/04_Embedding_Projection.ipynb
touch notebooks/05_Ablation_Comparison.ipynb

# Create results directory (often gitignored, but create structure)
mkdir results
touch results/.gitkeep

# Create main source directory structure
mkdir -p src/pgs_net/modules src/pgs_net/utils
touch src/pgs_net/__init__.py
touch src/pgs_net/config.py
touch src/pgs_net/pgs_ffn.py
touch src/pgs_net/transformer_block.py

# Create files within src/pgs_net/modules/
touch src/pgs_net/modules/__init__.py
touch src/pgs_net/modules/interfaces.py
touch src/pgs_net/modules/placeholders.py
touch src/pgs_net/modules/complex_utils.py
touch src/pgs_net/modules/adapters.py
touch src/pgs_net/modules/geometry_similarity.py
touch src/pgs_net/modules/clustering_assignment.py
touch src/pgs_net/modules/dynamic_k.py
touch src/pgs_net/modules/queen_computation.py
touch src/pgs_net/modules/neighbor_search.py
touch src/pgs_net/modules/formal_force.py
touch src/pgs_net/modules/force_calculator.py
touch src/pgs_net/modules/integrator.py
touch src/pgs_net/modules/normalization.py
touch src/pgs_net/modules/regularization.py
touch src/pgs_net/modules/meta_config.py
touch src/pgs_net/modules/cross_layer.py
touch src/pgs_net/modules/non_locality.py

# Create files within src/pgs_net/utils/
touch src/pgs_net/utils/__init__.py
touch src/pgs_net/utils/logging_utils.py
touch src/pgs_net/utils/analysis_processing.py
touch src/pgs_net/utils/viz_utils.py
touch src/pgs_net/utils/projection_utils.py

# Create tests directory structure
mkdir -p tests/modules
touch tests/__init__.py
touch tests/conftest.py
touch tests/modules/__init__.py
touch tests/modules/test_geometry_similarity.py # Example test file
touch tests/modules/test_dynamic_k.py       # Example test file
touch tests/test_pgs_ffn.py
touch tests/test_transformer_block.py

# Create root project files
touch .gitignore
touch requirements.txt
touch README.md

echo "PGS-Net project structure created successfully."