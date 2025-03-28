# src/pgs_net/modules/neighbor_search.py
"""Neighbor Search Algorithms (Naive, Faiss)."""

import torch
import torch.nn as nn
import logging
import time
from typing import Optional, Tuple
from .interfaces import NeighborSearch

# Check for Faiss
try:
    import faiss

    # Check if GPU faiss is available
    try:
        res_test = faiss.StandardGpuResources()
        faiss_gpu_available = True
        del res_test
    except AttributeError:
        # Older Faiss without GpuResources or GPU support not compiled
        faiss_gpu_available = False
        try:  # Check CPU faiss exists
            index = faiss.IndexFlatL2(10)
            faiss_cpu_available = True
        except Exception:
            faiss_cpu_available = False
    except Exception:  # Other Faiss GPU init errors
        faiss_gpu_available = False
        faiss_cpu_available = hasattr(faiss, "IndexFlatL2")  # Check basic CPU index exists
    faiss_available = faiss_gpu_available or faiss_cpu_available
except ImportError:
    faiss = None
    faiss_available = False
    faiss_gpu_available = False
    faiss_cpu_available = False

logger = logging.getLogger(__name__)
if faiss_available:
    logger.info(f"Faiss found (GPU Available: {faiss_gpu_available}, CPU Available: {faiss_cpu_available}).")
else:
    logger.warning("Faiss library not found. Efficient Boids neighbor search disabled.")


class NaiveNeighborSearch(NeighborSearch):
    """Finds k nearest neighbors using pairwise distances (O(T^2))."""

    def __init__(self, k: int, is_complex: bool):
        super().__init__()
        self.k = k
        self.is_complex = is_complex
        logger.info(f"Initialized Naive Neighbor Search (k={k}).")

    def find_neighbors(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): Input tensor, shape (B, T, D).
        Returns: Tuple of indices (B, T, k) and vecs (B, T, k, D).
        """
        start_time = time.time()
        B, T, D = x.shape
        k = min(self.k, T - 1)
        if k <= 0:
            return None, None

        # Calculate pairwise squared distances (complex-aware)
        # (B, T, 1, D) - (B, 1, T, D) -> (B, T, T, D)
        diffs = x.unsqueeze(2) - x.unsqueeze(1)
        dists_sq = diffs.abs().pow(2).sum(dim=-1)  # (B, T, T), real distances sq

        # Avoid selecting self by setting diagonal to infinity
        dists_sq.diagonal(dim1=1, dim2=2).fill_(float("inf"))

        # Find top k smallest distances (indices)
        try:
            _, neighbor_indices = torch.topk(dists_sq, k=k, dim=-1, largest=False, sorted=False)  # (B, T, k)
        except RuntimeError as e:  # Handle k > valid neighbors
            logger.warning(f"Naive search topk failed (k={k}, T={T}): {e}. Trying smaller k.")
            k_adjusted = min(k, T - 1)
            if k_adjusted <= 0:
                return None, None
            _, neighbor_indices = torch.topk(dists_sq, k=k_adjusted, dim=-1, largest=False, sorted=False)
            # Pad if k_adjusted < k
            if k_adjusted < k:
                padding = torch.full((B, T, k - k_adjusted), -1, dtype=torch.long, device=x.device)
                neighbor_indices = torch.cat([neighbor_indices, padding], dim=-1)

        # Gather neighbor vectors
        # Use advanced indexing: x[batch_indices, neighbor_indices]
        batch_idx = torch.arange(B, device=x.device).view(-1, 1, 1)
        # Handle potential -1 indices from padding
        valid_mask = neighbor_indices != -1
        neighbor_vecs = torch.zeros(B, T, k, D, dtype=x.dtype, device=x.device)

        # Efficient gather using valid mask (can be complex)
        # Create flat indices for gathering
        batch_flat = batch_idx.expand(-1, T, k)[valid_mask]
        token_flat = torch.arange(T, device=x.device).view(1, -1, 1).expand(B, -1, k)[valid_mask]  # Not needed for gather source
        neighbor_idx_flat = neighbor_indices[valid_mask]  # Valid neighbor indices (flat)

        if batch_flat.numel() > 0:  # Check if any valid neighbors exist
            # Gather from source x using batch_flat and neighbor_idx_flat
            gathered_vecs = x[batch_flat, neighbor_idx_flat]  # (NumValid, D)
            # Scatter back into neighbor_vecs
            # Create index tuple for scattering
            scatter_idx_tuple = torch.nonzero(valid_mask, as_tuple=True)  # Returns (batch_idxs, token_idxs, k_idxs)
            neighbor_vecs[scatter_idx_tuple] = gathered_vecs

        logger.debug(f"Naive neighbor search took {time.time() - start_time:.4f} seconds.")
        return neighbor_indices, neighbor_vecs


class FaissNeighborSearch(NeighborSearch):
    """Finds k nearest neighbors using Faiss (GPU or CPU)."""

    def __init__(
        self, k: int, d_head: int, is_complex: bool, index_type: str = "IndexFlatL2", update_every: int = 1, use_gpu: bool = True
    ):
        super().__init__()
        self.k = k
        self.d_head = d_head
        self.is_complex = is_complex
        self.index_type_str = index_type
        self.update_every = update_every
        self.steps_since_update = 0
        self.use_gpu = use_gpu and faiss_gpu_available  # Use GPU only if requested and available

        if not faiss_available:
            raise ImportError("Faiss library not found/loaded.")
        if self.use_gpu and not faiss_gpu_available:
            logger.warning("Faiss GPU requested but unavailable. Falling back to Faiss CPU.")
            self.use_gpu = False
        if not self.use_gpu and not faiss_cpu_available:
            raise RuntimeError("Faiss CPU not available, cannot use FaissNeighborSearch without GPU or CPU.")

        logger.info(
            f"Initialized Faiss Neighbor Search (k={k}, D={d_head}, GPU={self.use_gpu}, Index={index_type}, UpdateEvery={update_every})."
        )

        self.faiss_res: Optional[faiss.GpuResources] = None
        if self.use_gpu:
            try:
                self.faiss_res = faiss.StandardGpuResources()
            except Exception as e:
                logger.error(f"Failed to init Faiss GPU resources: {e}")
                self.use_gpu = False  # Fallback to CPU

        self.current_index: Optional[faiss.Index] = None  # Index built dynamically

    def find_neighbors(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): Input tensor, shape (B, T, D).
        Returns: Tuple of indices (B, T, k) and vecs (B, T, k, D).
        """
        start_time = time.time()
        B, T, D_in = x.shape
        k = self.k
        k_search = min(k + 1, T)  # Search for k+1 to exclude self
        if k_search <= 1:
            return None, None

        # Prepare data for Faiss (needs float32, contiguous, CPU for CPU index)
        x_real = (x.real if self.is_complex else x).detach().to(torch.float32)
        if x_real.shape[-1] != self.d_head:
            logger.error(f"Input dim {x_real.shape[-1]} != d_head {self.d_head}. Skipping Faiss.")
            return None, None

        all_neighbor_indices = -torch.ones(B, T, k, dtype=torch.long, device=x.device)
        all_neighbor_vecs = torch.zeros(B, T, k, self.d_head, dtype=self.dtype, device=x.device)
        found_any = False

        # --- Batch-by-Batch Search ---
        for b in range(B):
            data_b = x_real[b].contiguous()  # (T, D)
            index_b: Optional[faiss.Index] = None
            try:
                # --- Build Index (CPU or GPU) ---
                if self.use_gpu and self.faiss_res is not None:
                    index_b = faiss.GpuIndexFlatL2(self.faiss_res, self.d_head)  # Direct GPU index (only supports FlatL2 easily)
                    # For other index types on GPU: build CPU then transfer
                    # index_cpu_b = getattr(faiss, self.index_type_str)(self.d_head) # Build on CPU
                    # index_b = faiss.index_cpu_to_gpu(self.faiss_res, 0, index_cpu_b)
                    # if needs training: index_b.train(data_b) # Train on GPU data
                    index_b.add(data_b)  # Add on GPU
                elif faiss_cpu_available:
                    # Build and search on CPU
                    data_b_cpu = data_b.cpu().numpy()
                    index_b = getattr(faiss, self.index_type_str)(self.d_head)
                    if index_b.is_trained is False:
                        index_b.train(data_b_cpu)
                    index_b.add(data_b_cpu)
                else:
                    raise RuntimeError("No usable Faiss index found (GPU/CPU).")  # Should not happen based on init check

                # --- Search ---
                if self.use_gpu:
                    distances_b_gpu, indices_b_gpu = index_b.search(data_b, k_search)  # Search on GPU
                    # Move results to CPU for processing? Or process on GPU? Process on GPU.
                    distances_b, indices_b = distances_b_gpu, indices_b_gpu
                else:  # CPU Search
                    distances_b_np, indices_b_np = index_b.search(data_b_cpu, k_search)
                    # Move results to correct device
                    distances_b = torch.from_numpy(distances_b_np).to(x.device)
                    indices_b = torch.from_numpy(indices_b_np).to(x.device)  # Indices are Long

            except Exception as e:
                logger.error(f"Faiss index/search failed for batch item {b}: {e}", exc_info=True)
                if index_b is not None:
                    del index_b  # Cleanup
                continue  # Skip this batch item

            # --- Process Results for item b ---
            self_indices = torch.arange(T, device=x.device).unsqueeze(1)
            valid_neighbors_mask = (indices_b != self_indices) & (indices_b != -1)  # Exclude self and invalid index (-1)

            # Efficiently get top-k valid neighbors using masked distances
            masked_distances = torch.where(valid_neighbors_mask, distances_b, torch.tensor(float("inf"), device=x.device))
            # Sort distances to find order, take top-k non-inf distances/indices
            sorted_dist, sorted_orig_idx_in_search = torch.topk(masked_distances, k=min(k_search, T), dim=-1, largest=False)

            # Indices of the k nearest valid neighbors
            neighbor_indices_b = indices_b.gather(dim=1, index=sorted_orig_idx_in_search)[:, :k]  # Take top k valid
            # Create mask for indices that are actually valid (not padding -1 and not inf distance)
            valid_found_mask = (neighbor_indices_b != -1) & (sorted_dist[:, :k] != float("inf"))
            num_found_per_token = valid_found_mask.sum(dim=-1)

            # Gather vectors only for valid found neighbors
            if valid_found_mask.any():
                valid_indices_t = neighbor_indices_b[valid_found_mask]  # Flat tensor of valid indices [0..T-1]
                # Need to map flat indices back to (t, k_idx) structure for scattering
                scatter_b_idx, scatter_t_idx, scatter_k_idx = torch.nonzero(valid_found_mask, as_tuple=True)

                if valid_indices_t.numel() > 0:  # Check if any valid indices remain
                    gathered_vecs = x[b][valid_indices_t]  # Gather vectors using flat valid indices
                    # Scatter back using the computed scatter indices
                    all_neighbor_vecs[b, scatter_t_idx, scatter_k_idx] = gathered_vecs
                    all_neighbor_indices[b, scatter_t_idx, scatter_k_idx] = valid_indices_t  # Store the valid indices
                    found_any = True

            del index_b  # Cleanup index for this item

        logger.debug(f"Faiss neighbor search took {time.time() - start_time:.4f} seconds.")
        if not found_any:
            return None, None
        return all_neighbor_indices, all_neighbor_vecs
