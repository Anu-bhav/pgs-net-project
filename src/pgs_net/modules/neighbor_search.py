# src/pgs_net/modules/neighbor_search.py
"""Neighbor Search Algorithms (Naive, Faiss)."""

import torch
import torch.nn as nn
import logging
import time
from typing import Optional, Tuple
from .interfaces import NeighborSearch
import numpy as np  # Faiss often works with numpy

# Check for Faiss
try:
    import faiss

    # Check if GPU faiss is available
    try:
        res_test = faiss.StandardGpuResources()
        faiss_gpu_available = True
        del res_test
        logger.info("Faiss GPU resources available.")
    except AttributeError:  # Older Faiss or no GPU support compiled
        faiss_gpu_available = False
        logger.info("Faiss GPU resources unavailable (AttributeError). Checking CPU.")
        try:  # Check CPU faiss exists
            index = faiss.IndexFlatL2(10)
            faiss_cpu_available = True
            logger.info("Faiss CPU available.")
        except Exception:
            faiss_cpu_available = False
            logger.warning("Faiss CPU index creation failed.")
    except Exception as e:  # Other Faiss GPU init errors
        faiss_gpu_available = False
        logger.warning(f"Faiss GPU resources unavailable ({e}). Checking CPU.")
        faiss_cpu_available = hasattr(faiss, "IndexFlatL2")
        if faiss_cpu_available:
            logger.info("Faiss CPU available.")
        else:
            logger.warning("Faiss CPU not available.")
    faiss_available = faiss_gpu_available or faiss_cpu_available
except ImportError:
    faiss = None
    faiss_available = False
    faiss_gpu_available = False
    faiss_cpu_available = False
    logger.warning("Faiss library not found. Efficient Boids neighbor search disabled.")

logger = logging.getLogger(__name__)


class NaiveNeighborSearch(NeighborSearch):
    """Finds k nearest neighbors using pairwise distances (O(T^2))."""

    def __init__(self, k: int, is_complex: bool):
        """
        Initializes NaiveNeighborSearch.

        Args:
            k (int): Number of neighbors to find.
            is_complex (bool): Whether input tensors are complex.
        """
        super().__init__()
        if k <= 0:
            logger.warning("NaiveNeighborSearch initialized with k<=0. It will return no neighbors.")
        self.k = k
        self.is_complex = is_complex
        logger.info(f"Initialized Naive Neighbor Search (k={k}).")

    @torch.no_grad()
    def find_neighbors(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): Input tensor, shape (B, T, D).
        Returns: Tuple of indices (B, T, k) and vecs (B, T, k, D), or (None, None).
        """
        start_time = time.time()
        B, T, D = x.shape
        k = min(self.k, T - 1)  # Cannot have more neighbors than other tokens
        if k <= 0:
            logger.debug("k <= 0, returning no neighbors.")
            return None, None

        logger.debug(f"Naive search starting: B={B}, T={T}, k={k}")

        # Calculate pairwise squared distances (complex-aware)
        try:
            # Ensure float32 for stability if complex or otherwise needed
            x_float = x.to(torch.float32) if not self.is_complex else x.to(torch.complex64)

            # (B, T, 1, D) - (B, 1, T, D) -> (B, T, T, D)
            diffs = x_float.unsqueeze(2) - x_float.unsqueeze(1)
            # Use linalg.vector_norm for robust magnitude calculation
            dists_sq = torch.linalg.vector_norm(diffs, ord=2, dim=-1).pow(2)  # (B, T, T), real distances sq
            # Clamp to avoid potential numerical issues becoming NaN/inf before topk
            dists_sq = torch.nan_to_num(dists_sq, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
        except Exception as e:
            logger.error(f"Naive search distance calculation failed: {e}", exc_info=True)
            return None, None

        # Avoid selecting self by setting diagonal to infinity
        dists_sq.diagonal(dim1=1, dim2=2).fill_(float("inf"))

        # Find top k smallest distances (indices)
        try:
            # Use float for distances to handle inf correctly in topk
            _, neighbor_indices = torch.topk(dists_sq.float(), k=k, dim=-1, largest=False, sorted=False)  # (B, T, k)
        except RuntimeError as e:
            logger.warning(f"Naive search topk failed (k={k}, T={T}): {e}. No neighbors returned.")
            return None, None  # Fail gracefully if topk fails (e.g., k > T-1 after masking)

        # Gather neighbor vectors using advanced indexing
        batch_idx = torch.arange(B, device=x.device).view(-1, 1, 1)
        # neighbor_indices contains indices within the T dimension [0...T-1]
        # We want x[batch_idx, neighbor_indices] -> output shape (B, T, k, D)
        try:
            # Perform gather on original tensor x (potentially complex)
            neighbor_vecs = x[batch_idx, neighbor_indices]
        except IndexError as e:
            logger.error(
                f"Naive search gathering failed (Indices shape: {neighbor_indices.shape}, Max Index: {neighbor_indices.max()}): {e}"
            )
            return None, None  # Index out of bounds likely

        logger.debug(f"Naive neighbor search took {time.time() - start_time:.4f} seconds.")
        return neighbor_indices.long(), neighbor_vecs  # Ensure indices are long


class FaissNeighborSearch(NeighborSearch):
    """Finds k nearest neighbors using Faiss (GPU or CPU). Handles batch items independently."""

    def __init__(
        self, k: int, d_head: int, is_complex: bool, index_type: str = "IndexFlatL2", update_every: int = 1, use_gpu: bool = True
    ):
        """
        Initializes FaissNeighborSearch.

        Args:
            k (int): Number of neighbors to find.
            d_head (int): Dimension of the embeddings.
            is_complex (bool): Whether inputs are complex.
            index_type (str): Faiss index type string (e.g., "IndexFlatL2", "IndexIVFFlat").
            update_every (int): Rebuild index every N steps (0 = only once if possible). Not fully implemented.
            use_gpu (bool): Attempt to use GPU Faiss if available.
        """
        super().__init__()
        if not faiss_available:
            raise ImportError("Faiss library not found/loaded.")
        if k <= 0:
            logger.warning("FaissNeighborSearch initialized with k<=0. It will return no neighbors.")

        self.k = k
        self.d_head = d_head
        self.is_complex = is_complex
        self.index_type_str = index_type
        # update_every not fully supported yet, index rebuilt each time for simplicity/correctness
        self.update_every = 1  # Force rebuild for now
        self.steps_since_update = 0
        self.use_gpu = use_gpu and faiss_gpu_available

        if self.use_gpu and not faiss_gpu_available:
            logger.warning("Faiss GPU requested but unavailable. Falling back to Faiss CPU.")
            self.use_gpu = False
        if not self.use_gpu and not faiss_cpu_available:
            raise RuntimeError("Faiss CPU not available, cannot use FaissNeighborSearch without GPU or CPU.")

        logger.info(
            f"Initialized Faiss Neighbor Search (k={k}, D={d_head}, GPU={self.use_gpu}, Index={index_type}, UpdateEvery={self.update_every})."
        )

        self.faiss_res: Optional[faiss.GpuResources] = None
        if self.use_gpu:
            try:
                self.faiss_res = faiss.StandardGpuResources()
                logger.info("Faiss GPU resources initialized.")
            except Exception as e:
                logger.error(f"Failed to init Faiss GPU resources: {e}. Falling back to CPU.")
                self.use_gpu = False  # Fallback to CPU if GPU resources fail

    @torch.no_grad()
    def find_neighbors(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): Input tensor, shape (B, T, D).
        Returns: Tuple of indices (B, T, k) and vecs (B, T, k, D), or (None, None).
        """
        start_time = time.time()
        B, T, D_in = x.shape
        k = self.k
        k_search = min(k + 1, T)  # Search for k+1 to exclude self
        if k <= 0 or k_search <= 1:
            logger.debug("k <= 0 or T <= 1, returning no neighbors.")
            return None, None

        # Prepare data (real, float32, contiguous)
        x_real = (x.real if self.is_complex else x).detach().to(torch.float32)
        if x_real.shape[-1] != self.d_head:
            logger.error(f"Input dim {x_real.shape[-1]} != d_head {self.d_head}. Skipping Faiss.")
            return None, None

        # Allocate result tensors (initialize with -1 for indices)
        all_neighbor_indices = torch.full((B, T, k), -1, dtype=torch.long, device=x.device)
        all_neighbor_vecs = torch.zeros((B, T, k, self.d_head), dtype=x.dtype, device=x.device)
        found_any = False

        # --- Batch-by-Batch Search ---
        for b in range(B):
            data_b = x_real[b].contiguous()  # (T, D)
            index_b: Optional[faiss.Index] = None
            try:
                # --- Build Index (CPU or GPU) ---
                if self.use_gpu and self.faiss_res is not None:
                    # Direct GPU index (e.g., FlatL2 or IVF_Flat if trained)
                    # Note: IVF needs training phase. FlatL2 is simplest.
                    if self.index_type_str != "IndexFlatL2":
                        logger.warning(
                            f"Faiss GPU currently tested with IndexFlatL2, requested {self.index_type_str}. Using FlatL2."
                        )
                    index_b = faiss.GpuIndexFlatL2(self.faiss_res, self.d_head, faiss.METRIC_L2)
                    index_b.add(data_b)  # Add on GPU
                elif faiss_cpu_available:
                    # Build and search on CPU
                    data_b_cpu = data_b.cpu().numpy()
                    # Check if index type exists
                    if not hasattr(faiss, self.index_type_str):
                        logger.error(f"Faiss index type '{self.index_type_str}' not found. Using IndexFlatL2 CPU fallback.")
                        self.index_type_str = "IndexFlatL2"

                    index_b = getattr(faiss, self.index_type_str)(self.d_head)
                    # Handle potential training requirement for index types like IVF
                    if index_b.is_trained is False:
                        # Need enough data to train, might fail for small T
                        if T >= getattr(index_b, "nlist", 1) * 39:  # Faiss training heuristic
                            logger.debug(f"Training Faiss CPU index {self.index_type_str}...")
                            index_b.train(data_b_cpu)
                        else:
                            logger.warning(
                                f"Not enough data (T={T}) to train Faiss index {self.index_type_str}. Using FlatL2 CPU fallback."
                            )
                            del index_b
                            index_b = faiss.IndexFlatL2(self.d_head)
                    index_b.add(data_b_cpu)
                else:
                    logger.error("No usable Faiss index could be built.")
                    continue  # Skip this batch item

                # --- Search ---
                if self.use_gpu:
                    distances_b_gpu, indices_b_gpu = index_b.search(data_b, k_search)
                    distances_b, indices_b = distances_b_gpu, indices_b_gpu
                else:  # CPU Search
                    distances_b_np, indices_b_np = index_b.search(data_b_cpu, k_search)
                    distances_b = torch.from_numpy(distances_b_np).to(x.device)
                    indices_b = torch.from_numpy(indices_b_np).to(x.device)

            except Exception as e:
                logger.error(f"Faiss index/search failed for batch item {b}: {e}", exc_info=True)
                if index_b is not None:
                    del index_b
                continue

            # --- Process Results for item b ---
            # Indices_b: (T, k_search), Distances_b: (T, k_search)
            self_indices = torch.arange(T, device=x.device).unsqueeze(1)
            # Mask out self AND invalid indices (-1 returned by Faiss on insufficient neighbors)
            valid_neighbors_mask = (indices_b != self_indices) & (indices_b != -1)

            # Efficiently get top-k valid neighbors using masked distances
            masked_distances = torch.where(valid_neighbors_mask, distances_b, torch.tensor(float("inf"), device=x.device))
            # Sort distances to find order, take top-k non-inf distances/indices
            sorted_dist, sorted_orig_idx_in_search = torch.topk(masked_distances, k=min(k_search, T), dim=-1, largest=False)

            # Indices of the k nearest valid neighbors from the original search results
            neighbor_indices_b = indices_b.gather(dim=1, index=sorted_orig_idx_in_search)  # (T, k_search) sorted by distance

            # Select the top k that are valid (not self, not -1, not inf distance)
            valid_k_mask = (neighbor_indices_b[:, :k] != -1) & (sorted_dist[:, :k] != float("inf"))  # (T, k)
            num_found_per_token = valid_k_mask.sum(dim=-1)  # (T,)

            if valid_k_mask.any():
                # Get the indices [0..T-1] of the valid neighbors
                valid_indices_t = neighbor_indices_b[:, :k][valid_k_mask]  # Flat tensor of valid neighbor indices

                # Create scattering indices
                scatter_t_idx, scatter_k_idx = torch.nonzero(valid_k_mask, as_tuple=True)  # Indices within (T, k) space

                if valid_indices_t.numel() > 0:
                    # Gather vectors using original x (potentially complex) for item b
                    gathered_vecs = x[b][valid_indices_t]  # (NumValid, D)
                    # Scatter back into result tensors for this batch item
                    all_neighbor_vecs[b, scatter_t_idx, scatter_k_idx] = gathered_vecs
                    all_neighbor_indices[b, scatter_t_idx, scatter_k_idx] = valid_indices_t
                    found_any = True

            if index_b is not None:
                del index_b  # Cleanup index for this item

        logger.debug(f"Faiss neighbor search took {time.time() - start_time:.4f} seconds for batch size {B}.")
        if not found_any:
            logger.warning("Faiss search completed but found no valid neighbors.")
            return None, None
        return all_neighbor_indices, all_neighbor_vecs

    def extra_repr(self) -> str:
        return f"k={self.k}, d_head={self.d_head}, index={self.index_type_str}, gpu={self.use_gpu}"
