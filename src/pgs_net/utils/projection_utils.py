# src/pgs_net/utils/projection_utils.py
"""Utilities for dimensionality reduction (UMAP, t-SNE)."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Check optional dependencies
try:
    import umap  # umap-learn package

    umap_available = True
except ImportError:
    umap = None
    umap_available = False
    logger.info("UMAP library not found (`pip install umap-learn`). UMAP projection disabled.")

try:
    from sklearn.decomposition import PCA  # Often useful for pre-processing TSNE
    from sklearn.manifold import TSNE

    sklearn_available = True
except ImportError:
    TSNE = None
    PCA = None
    sklearn_available = False
    logger.info("scikit-learn not found (`pip install scikit-learn`). t-SNE/PCA projection disabled.")


def project_embeddings(
    embeddings: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    use_pca_preprocessing: bool = True,
    pca_components: int = 50,
    random_state: int = 42,
    umap_kwargs: Optional[dict] = None,
    tsne_kwargs: Optional[dict] = None,
) -> Optional[np.ndarray]:
    """
    Projects high-dim embeddings to 2D or 3D using UMAP or t-SNE.

    Args:
        embeddings (np.ndarray): Numpy array of embeddings, shape (N, D).
        method (str): Projection method ('umap' or 'tsne').
        n_components (int): Target dimension (2 or 3).
        use_pca_preprocessing (bool): If True and D > pca_components, apply PCA before t-SNE/UMAP.
        pca_components (int): Number of components for PCA preprocessing.
        random_state (int): Random seed for reproducibility.
        umap_kwargs (Optional[dict]): Additional kwargs for umap.UMAP.
        tsne_kwargs (Optional[dict]): Additional kwargs for sklearn.manifold.TSNE.

    Returns:
        Optional[np.ndarray]: Projected embeddings, shape (N, n_components), or None if failed.

    """
    if embeddings is None or not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        logger.error("Invalid input embeddings for projection.")
        return None
    if n_components not in [2, 3]:
        logger.error("n_components must be 2 or 3 for visualization.")
        return None

    N, D = embeddings.shape
    logger.info(f"Starting projection: Method={method.upper()}, N={N}, D={D}, TargetDim={n_components}")

    # Ensure float32 for most algorithms
    embeddings_float = embeddings.astype(np.float32)

    # Optional PCA Preprocessing (especially recommended for t-SNE)
    if use_pca_preprocessing and D > pca_components and PCA is not None:
        logger.info(f"Applying PCA preprocessing ({pca_components} components)...")
        try:
            pca = PCA(n_components=pca_components, random_state=random_state)
            embeddings_processed = pca.fit_transform(embeddings_float)
            logger.info(f"PCA complete. Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
        except Exception as e:
            logger.error(f"PCA preprocessing failed: {e}. Using original data.", exc_info=True)
            embeddings_processed = embeddings_float
    else:
        embeddings_processed = embeddings_float

    projected: Optional[np.ndarray] = None
    # --- Projection ---
    if method.lower() == "umap":
        if not umap_available:
            logger.error("UMAP selected but library not available.")
            return None
        try:
            logger.info(f"Running UMAP (n_components={n_components})...")
            umap_params = {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean"}
            if umap_kwargs:
                umap_params.update(umap_kwargs)
            reducer = umap.UMAP(n_components=n_components, random_state=random_state, **umap_params)
            projected = reducer.fit_transform(embeddings_processed)
        except Exception as e:
            logger.error(f"UMAP projection failed: {e}", exc_info=True)
            return None

    elif method.lower() == "tsne":
        if not sklearn_available:
            logger.error("t-SNE selected but scikit-learn not available.")
            return None
        try:
            logger.info(f"Running t-SNE (n_components={n_components})...")
            # Default perplexity depends on N, typical range 5-50
            perplexity_val = min(30.0, max(5.0, (N - 1) / 3.0 - 1))  # Heuristic
            tsne_params = {"perplexity": perplexity_val, "n_iter": 300, "learning_rate": "auto", "init": "pca"}
            if tsne_kwargs:
                tsne_params.update(tsne_kwargs)
            reducer = TSNE(n_components=n_components, random_state=random_state, **tsne_params)
            projected = reducer.fit_transform(embeddings_processed)
        except Exception as e:
            logger.error(f"t-SNE projection failed: {e}", exc_info=True)
            return None
    else:
        logger.error(f"Unknown projection method: {method}")
        return None

    logger.info(f"Projection complete. Output shape: {projected.shape}")
    return projected
