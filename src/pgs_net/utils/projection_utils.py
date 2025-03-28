# src/pgs_net/utils/projection_utils.py
"""Utilities for dimensionality reduction (UMAP, t-SNE, PCA)."""

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Check optional dependencies
try:
    import umap  # umap-learn package

    umap_available = True
    logger.debug("UMAP library loaded.")
except ImportError:
    umap = None
    umap_available = False
    logger.info("UMAP library not found (`pip install umap-learn`). UMAP projection disabled.")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    sklearn_available = True
    logger.debug("scikit-learn library loaded.")
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
    umap_kwargs: Optional[Dict[str, Any]] = None,
    tsne_kwargs: Optional[Dict[str, Any]] = None,
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
        logger.error("Invalid input embeddings for projection (must be 2D numpy array).")
        return None
    if n_components not in [2, 3]:
        logger.error("n_components must be 2 or 3 for visualization.")
        return None
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        logger.warning(
            f"Embeddings contain NaN/Inf ({np.isnan(embeddings).sum()} NaN, {np.isinf(embeddings).sum()} Inf). Projection may fail or be inaccurate."
        )
        # Option: Remove or impute NaNs/Infs
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)  # Replace with zeros or large numbers

    N, D = embeddings.shape
    if N == 0:
        logger.warning("Cannot project empty embeddings array.")
        return None
    logger.info(f"Starting projection: Method={method.upper()}, N={N}, D={D}, TargetDim={n_components}")

    # Ensure float32 for most algorithms
    embeddings_float = embeddings.astype(np.float32)

    # Optional PCA Preprocessing
    embeddings_processed = embeddings_float
    if use_pca_preprocessing and D > pca_components and N > pca_components and PCA is not None:
        logger.info(f"Applying PCA preprocessing ({pca_components} components)...")
        try:
            # Ensure pca_components is not larger than min(N, D)
            n_pca = min(pca_components, N, D)
            if n_pca < 2:
                raise ValueError("Not enough samples/features for PCA.")
            pca = PCA(n_components=n_pca, random_state=random_state, svd_solver="auto")
            embeddings_processed = pca.fit_transform(embeddings_float)
            logger.info(
                f"PCA complete. Output shape: {embeddings_processed.shape}, Explained variance: {pca.explained_variance_ratio_.sum():.3f}"
            )
        except Exception as e:
            logger.error(f"PCA preprocessing failed: {e}. Using original data.", exc_info=True)
            embeddings_processed = embeddings_float  # Fallback to original
    elif use_pca_preprocessing:
        logger.info("Skipping PCA preprocessing (D <= pca_components or N <= pca_components or PCA unavailable).")

    projected: Optional[np.ndarray] = None
    start_time = time.time()
    # --- Projection ---
    if method.lower() == "umap":
        if not umap_available:
            logger.error("UMAP selected but library not available.")
            return None
        try:
            logger.info(f"Running UMAP (n_components={n_components})...")
            umap_params = {
                "n_neighbors": min(15, N - 1),
                "min_dist": 0.1,
                "metric": "euclidean",
            }  # Adjust n_neighbors if N is small
            if N <= umap_params["n_neighbors"]:
                umap_params["n_neighbors"] = max(2, N - 1)  # Ensure n_neighbors < N
            if umap_kwargs:
                umap_params.update(umap_kwargs)
            logger.debug(f"UMAP params: {umap_params}")
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
            perplexity_val = min(30.0, max(5.0, (N - 1) / 3.0 - 1)) if N > 5 else 5.0  # Heuristic, ensure N > perplexity
            tsne_params = {
                "perplexity": perplexity_val,
                "n_iter": max(250, N // 10),
                "learning_rate": "auto",
                "init": "pca" if D > n_components else "random",
            }
            if tsne_kwargs:
                tsne_params.update(tsne_kwargs)
            # Ensure perplexity < N
            tsne_params["perplexity"] = min(tsne_params["perplexity"], N - 1)
            logger.debug(f"t-SNE params: {tsne_params}")
            reducer = TSNE(n_components=n_components, random_state=random_state, **tsne_params)
            projected = reducer.fit_transform(embeddings_processed)
        except Exception as e:
            logger.error(f"t-SNE projection failed: {e}", exc_info=True)
            return None
    else:
        logger.error(f"Unknown projection method: {method}")
        return None

    proj_time = time.time() - start_time
    logger.info(f"Projection complete ({proj_time:.2f} sec). Output shape: {projected.shape}")
    return projected
