# src/pgs_net/utils/analysis_processing.py
"""Utilities for processing the analysis_data dictionary collected during PGS_FFN runs."""

import ast  # For safely evaluating string representations of lists/dicts
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Constants for standard keys (enhances consistency)
STEP = "step"
LAYER_IDX = "layer_idx"
PGS_FFN_DATA = "pgs_ffn"  # Top level key for PGS data in block analysis
GLOBAL_DATA = "global"
HEAD_DATA = "heads"
META_PARAMS = "meta_params"
GRADIENTS = "gradients"  # Key within global data for gradient dict
GEOM_MIX_WEIGHTS = "geometry_mix_weights"
ACTIVE_GEOM_BRANCHES = "active_geometry_branches"
ASSIGNMENT_ENTROPY = "assignment_entropy"
DYNK_ACTIVE_COUNT = "dynamic_k_active_count"
# Add more constants as needed...


def safe_literal_eval(val: Any) -> Any:
    """Safely evaluate string representations of Python literals (lists, dicts)."""
    if isinstance(val, (str)) and val.startswith(("[", "{", "(")) and val.endswith(("]", "}", ")")):
        try:
            # Use literal_eval for safety over eval()
            return ast.literal_eval(val)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            # logger.warning(f"Could not parse string with literal_eval: '{str(val)[:100]}...'") # Can be verbose
            return val  # Return original string if eval fails
    return val


def parse_tensor_list_column(series: pd.Series) -> Union[np.ndarray, List[Any]]:
    """
    Robustly parses DataFrame columns containing string representations
    or actual lists/tuples of tensors/numbers into a stacked numpy array if possible.

    Args:
        series (pd.Series): The DataFrame column to parse.

    Returns:
        Union[np.ndarray, List[Any]]: A 2D numpy array where each row corresponds
                                      to an item in the series (padded with NaN),
                                      or the original list of parsed items if stacking fails.

    """
    parsed = []
    max_len = 0
    has_vectors = False

    for item in series.dropna():  # Process non-null items
        val = np.nan  # Default if parsing fails
        try:
            item_eval = safe_literal_eval(item)  # Try evaluating string reps first

            if isinstance(item_eval, (list, tuple)):
                # Convert numeric items, handle potential tensors within list
                num_list = [v.item() if isinstance(v, torch.Tensor) else float(v) for v in item_eval]
                val = np.array(num_list, dtype=np.float32)
            elif isinstance(item_eval, torch.Tensor):
                num_list = item_eval.detach().cpu().numpy().flatten().astype(np.float32).tolist()
                val = np.array(num_list, dtype=np.float32)
            elif isinstance(item_eval, (int, float, np.number)):
                val = np.array([float(item_eval)], dtype=np.float32)  # Wrap scalar
            elif isinstance(item_eval, np.ndarray):  # Already numpy array
                val = item_eval.astype(np.float32)

            # Update max_len and track if we found any vectors
            if isinstance(val, np.ndarray):
                if val.ndim == 0:  # Handle 0-dim arrays from .item() on single element tensor lists
                    val = np.array([val.item()], dtype=np.float32)
                if val.ndim == 1:  # Ensure it's 1D vector
                    max_len = max(max_len, len(val))
                    has_vectors = True
                else:  # If > 1D array, don't try to stack, store as object? Or flatten? Flatten for now.
                    val = val.flatten()
                    max_len = max(max_len, len(val))
                    has_vectors = True
            else:
                val = np.nan  # Mark as NaN if not converted to array

        except Exception as e:
            # logger.warning(f"Parsing error for item '{str(item)[:100]}...': {e}", exc_info=True)
            val = np.nan  # Assign NaN on any parsing error

        parsed.append(val)

    if not has_vectors:  # Return list of NaNs if no vectors found
        logger.debug(f"Column {series.name}: No vector data found during parsing.")
        # Reconstruct series with NaNs matching original index
        nan_series = pd.Series([np.nan] * len(series), index=series.index)
        return nan_series  # Return series of NaNs

    # Pad arrays to max length and stack
    padded = []
    for x in parsed:
        if isinstance(x, np.ndarray):
            pad_width = max_len - len(x)
            padded.append(np.pad(x, (0, pad_width), mode="constant", constant_values=np.nan))
        else:  # Handle NaNs or non-array items resulting from errors
            padded.append(np.full(max_len, np.nan))

    try:
        # Stack arrays - result shape (len(parsed), max_len)
        stacked = np.stack(padded, axis=0)
        # Reconstruct series with stacked data, aligning with original non-null index
        result_series = pd.Series([row for row in stacked], index=series.dropna().index)
        # Reindex to match original series index, filling missing with NaN or appropriate value
        return result_series.reindex(series.index)
    except ValueError:  # Handle case where arrays have incompatible shapes despite padding attempt
        logger.error(f"Could not stack parsed arrays for column {series.name}. Returning list.")
        # Return list aligned with original index
        result_list_aligned = pd.Series(parsed, index=series.dropna().index).reindex(series.index)
        return result_list_aligned
    except Exception as e:
        logger.error(f"Error stacking parsed tensor/list column {series.name}: {e}")
        result_list_aligned = pd.Series(parsed, index=series.dropna().index).reindex(series.index)
        return result_list_aligned  # Return list as fallback


def flatten_analysis_dict(data_dict: Dict, parent_key: str = "", sep: str = "_", max_list_len: int = 10) -> Dict:
    """
    Flattens a nested dictionary, handling tensors and long lists.

    Args:
        data_dict (Dict): The dictionary to flatten.
        parent_key (str): Internal use for recursion.
        sep (str): Separator for nested keys.
        max_list_len (int): Maximum length of lists/tensors to store directly. Longer ones stored as summary string.

    Returns:
        Dict: Flattened dictionary.

    """
    items = []
    if data_dict is None:
        return {}

    for k, v in data_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)  # Ensure key is string
        try:
            if isinstance(v, dict):
                items.extend(flatten_analysis_dict(v, new_key, sep=sep, max_list_len=max_list_len).items())
            elif isinstance(v, torch.Tensor):
                tensor = v.detach().cpu()
                if tensor.numel() == 1:
                    v_item = tensor.item()
                elif tensor.numel() <= max_list_len:
                    v_item = tensor.numpy().tolist()  # Store short tensors/lists directly
                else:  # Summarize long tensors
                    v_item = f"Tensor(shape={tuple(tensor.shape)}, mean={tensor.float().mean():.3g})"
                items.append((new_key, v_item))
            elif isinstance(v, list):
                if all(isinstance(i, (int, float, bool, str)) for i in v):  # List of primitives
                    if len(v) <= max_list_len:
                        items.append((new_key, v))  # Store short list
                    else:
                        items.append((new_key, f"List(len={len(v)}, first={v[0]})"))  # Summarize
                elif all(isinstance(i, torch.Tensor) for i in v):  # List of tensors
                    if len(v) <= max_list_len and all(t.numel() == 1 for t in v):
                        items.append((new_key, [t.item() for t in v]))  # List of scalars
                    else:
                        items.append((new_key, f"List[Tensor(len={len(v)})]"))
                else:  # Mixed list or list of complex objects
                    items.append((new_key, f"List(len={len(v)}, type={type(v[0]).__name__ if v else 'empty'})"))
            elif isinstance(v, (int, float, bool, str, np.number)):
                items.append((new_key, v))  # Store primitives directly
            elif pd.isna(v):
                items.append((new_key, np.nan))  # Store NaN
            else:  # Handle other types as string representation
                items.append((new_key, str(v)))

        except Exception as e:
            logger.warning(f"Error flattening key '{new_key}': {e}")
            items.append((new_key, "ERROR_FLATTENING"))

    return dict(items)


def process_run_analysis_data(all_step_data_list: List[Dict], head_agg_method: str = "mean") -> pd.DataFrame:
    """
    Processes analysis data collected over multiple steps/batches for a full run.
    Creates a DataFrame where each row corresponds to a layer within a step.

    Args:
        all_step_data_list (list[dict]): List where each dict contains 'step', 'layers', etc.
        head_agg_method (str): How to aggregate metrics across heads ('mean', 'std', 'median', 'none').

    Returns:
        pd.DataFrame: Processed and flattened analysis data. Returns empty DF on failure.

    """
    if not all_step_data_list or not isinstance(all_step_data_list, list):
        logger.warning("No valid analysis data list provided for processing.")
        return pd.DataFrame()

    processed_rows = []
    logger.info(f"Processing analysis data for {len(all_step_data_list)} steps...")

    # Define aggregation function based on method
    if head_agg_method == "mean":
        agg_func = np.nanmean
    elif head_agg_method == "median":
        agg_func = np.nanmedian
    elif head_agg_method == "std":
        agg_func = np.nanstd
    else:
        agg_func = None
        logger.info("No head aggregation specified.")

    # Define keys that definitely shouldn't be aggregated numerically
    non_numeric_agg_keys = {ACTIVE_GEOM_BRANCHES, "gradients"}

    for step_data in all_step_data_list:
        if not isinstance(step_data, dict):
            continue
        step = step_data.get(STEP, -1)
        main_loss = step_data.get("main_loss", np.nan)
        step_total_aux_loss = step_data.get("aux_loss", np.nan)
        layer_data_list = step_data.get("layers", [])

        for layer_idx, layer_data in enumerate(layer_data_list):
            if layer_data is None or not isinstance(layer_data, dict):
                continue

            flat_layer_data = {}
            try:
                # Flatten global PGS data
                pgs_global_data = layer_data.get(PGS_FFN_DATA, {}).get(GLOBAL_DATA, {})
                flat_layer_data.update(flatten_analysis_dict(pgs_global_data, parent_key="pgs_global"))

                # Flatten non-PGS data (e.g., self_attn)
                for key, value in layer_data.items():
                    if key != PGS_FFN_DATA:
                        flat_layer_data.update(flatten_analysis_dict({key: value}, parent_key=key))

                # Aggregate or Flatten Head Data
                pgs_head_data_list = layer_data.get(PGS_FFN_DATA, {}).get(HEAD_DATA, [])
                num_heads = len(pgs_head_data_list)
                if num_heads > 0 and pgs_head_data_list[0]:
                    head_keys = list(pgs_head_data_list[0].keys())  # Get keys from first head
                    for key in head_keys:
                        all_head_values = [h_data.get(key) for h_data in pgs_head_data_list]

                        # Try aggregation if requested and key seems numeric
                        if agg_func is not None and key not in non_numeric_agg_keys:
                            numeric_values = []
                            for v in all_head_values:
                                try:
                                    numeric_values.append(float(v))  # Try converting to float
                                except (ValueError, TypeError, Exception):
                                    numeric_values.append(np.nan)

                            if not all(np.isnan(nv) for nv in numeric_values):
                                agg_val = agg_func(numeric_values)
                                flat_layer_data[f"pgs_head_agg_{key}"] = agg_val
                            elif key in [
                                GEOM_MIX_WEIGHTS,
                                ACTIVE_GEOM_BRANCHES,
                            ]:  # Store head 0 for these complex types if agg failed
                                flat_layer_data[f"pgs_head_0_{key}"] = all_head_values[0]
                            # else: logger.debug(f"Could not aggregate key '{key}' numerically.")

                        else:  # No aggregation - flatten per head (can create many columns)
                            # Limit this for now to avoid excessive columns, maybe only for specific keys?
                            if key in [
                                GEOM_MIX_WEIGHTS,
                                ACTIVE_GEOM_BRANCHES,
                                ASSIGNMENT_ENTROPY,
                            ]:  # Example keys to keep per-head
                                for h_idx, h_value in enumerate(all_head_values):
                                    flat_layer_data[f"pgs_head{h_idx}_{key}"] = h_value

                flat_layer_data[STEP] = step
                flat_layer_data[LAYER_IDX] = layer_idx
                flat_layer_data["step_main_loss"] = main_loss
                flat_layer_data["step_total_aux_loss"] = step_total_aux_loss
                processed_rows.append(flat_layer_data)

            except Exception as e:
                logger.error(f"Error processing layer {layer_idx} at step {step}: {e}", exc_info=True)

    logger.info(f"Flattened into {len(processed_rows)} analysis rows.")
    if not processed_rows:
        return pd.DataFrame()

    # --- Create DataFrame ---
    try:
        df = pd.DataFrame(processed_rows)
        # Convert numeric columns after creation
        for col in df.select_dtypes(include=["object"]).columns:
            # Avoid converting columns known to be non-numeric (like lists of branches)
            if ACTIVE_GEOM_BRANCHES not in col and GRADIENTS not in col:
                try:
                    converted = pd.to_numeric(df[col], errors="coerce")
                    if not converted.isna().all():
                        df[col] = converted
                except (ValueError, TypeError):
                    pass
        logger.info(f"DataFrame created with shape {df.shape}")
    except Exception as e:
        logger.error(f"Failed to create DataFrame: {e}", exc_info=True)
        return pd.DataFrame()  # Return empty on failure

    # --- Post-processing Specific Columns ---
    logger.info("Post-processing specific analysis columns...")
    try:
        # Geometry weights (using head 0 data as representative if aggregated failed)
        weight_col = f"pgs_head_0_{GEOM_MIX_WEIGHTS}"
        branches_col = f"pgs_head_0_{ACTIVE_GEOM_BRANCHES}"
        if weight_col in df.columns and branches_col in df.columns:
            if not any(col.startswith("pgs_geom_weight_") for col in df.columns):  # Check if not already processed
                logger.info("Attempting post-processing for geometry weights...")
                extracted_weights = False
                try:
                    branches_lists = df[branches_col].apply(safe_literal_eval)
                    weights_arrays = parse_tensor_list_column(df[weight_col])
                    if isinstance(weights_arrays, np.ndarray):  # Check if parsing worked
                        # Determine unique branches across all steps/layers if needed
                        all_branches = set()
                        for sublist in branches_lists:
                            if isinstance(sublist, list):
                                all_branches.update(sublist)
                        all_branches = sorted(list(all_branches))

                        if all_branches:
                            for branch in all_branches:
                                df[f"pgs_geom_weight_{branch}"] = np.nan  # Initialize
                            for i, row in df.iterrows():
                                branches = branches_lists.iloc[i]
                                weights = weights_arrays[i]
                                if (
                                    isinstance(branches, list)
                                    and isinstance(weights, np.ndarray)
                                    and len(branches) == len(weights)
                                ):
                                    for branch_name, weight_val in zip(branches, weights):
                                        if branch_name in all_branches:
                                            df.loc[i, f"pgs_geom_weight_{branch_name}"] = weight_val
                            extracted_weights = True
                except Exception as e:
                    logger.error(f"Error post-processing geometry weights: {e}", exc_info=True)
                if extracted_weights:
                    logger.info(f"Extracted geometry weights for branches: {all_branches}")
                else:
                    logger.warning("Could not extract geometry weights.")
                # Optionally drop original columns: df = df.drop(columns=[weight_col, branches_col], errors='ignore')

        # Gradient norms
        grad_log_cols = [c for c in df.columns if c.endswith(GRADIENTS)]  # Find gradient dict columns
        if grad_log_cols:
            logger.info("Attempting post-processing for gradient norms...")
        for grad_col in grad_log_cols:
            try:
                prefix = grad_col.replace(f"_{GRADIENTS}", "") + "_grad_"  # Create prefix like 'pgs_global_grad_'
                parsed_dicts = df[grad_col].apply(safe_literal_eval)
                valid_dicts = parsed_dicts[parsed_dicts.apply(lambda x: isinstance(x, dict))]
                if not valid_dicts.empty:
                    grad_df = pd.json_normalize(valid_dicts).reindex(df.index)  # Expand keys
                    grad_df.columns = [f"{prefix}{key}" for key in grad_df.columns]  # Add prefix
                    # Drop original before concat to avoid column conflicts if run multiple times
                    df = df.drop(columns=[grad_col], errors="ignore")
                    df = pd.concat([df, grad_df], axis=1)
                    logger.info(f"Extracted gradient norms from {grad_col} into columns with prefix {prefix}")
                else:
                    df = df.drop(columns=[grad_col], errors="ignore")  # Drop if empty/invalid
            except Exception as e:
                logger.error(f"Error post-processing gradient norms for {grad_col}: {e}")

    except Exception as e:
        logger.error(f"Error during general post-processing of analysis data: {e}", exc_info=True)

    logger.info("Analysis data processing complete.")
    return df
