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
GRADIENTS = "gradients"
GEOM_MIX_WEIGHTS = "geometry_mix_weights"
ACTIVE_GEOM_BRANCHES = "active_geometry_branches"
ASSIGNMENT_ENTROPY = "assignment_entropy"
DYNK_ACTIVE_COUNT = "dynamic_k_active_count"
# Add more constants as needed...


def safe_literal_eval(val: Any) -> Any:
    """Safely evaluate string representations of Python literals (lists, dicts)."""
    if isinstance(val, (str)):
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

    for item in series:
        val = np.nan  # Default if parsing fails
        try:
            item_eval = safe_literal_eval(item)  # Try evaluating string reps first

            if isinstance(item_eval, (list, tuple)):
                num_list = [v.item() if isinstance(v, torch.Tensor) else float(v) for v in item_eval]
                val = np.array(num_list, dtype=np.float32)
            elif isinstance(item_eval, torch.Tensor):
                # Ensure tensor is detached and on CPU before converting
                num_list = item_eval.detach().cpu().numpy().flatten().astype(np.float32).tolist()
                val = np.array(num_list, dtype=np.float32)
            elif isinstance(item_eval, (int, float, np.number)):
                val = np.array([float(item_eval)], dtype=np.float32)  # Wrap scalar
            elif isinstance(item_eval, np.ndarray):  # Already numpy array
                val = item_eval.astype(np.float32)

            # Update max_len and track if we found any vectors
            if isinstance(val, np.ndarray):
                max_len = max(max_len, len(val))
                has_vectors = True

        except Exception as e:
            # logger.warning(f"Parsing error for item '{str(item)[:100]}...': {e}", exc_info=True)
            val = np.nan  # Assign NaN on any parsing error

        parsed.append(val)

    if not has_vectors:  # Return list of NaNs if no vectors found
        return [np.nan] * len(series)

    # Pad arrays to max length and stack
    padded = []
    for x in parsed:
        if isinstance(x, np.ndarray):
            pad_width = max_len - len(x)
            # Pad with NaN
            padded.append(np.pad(x, (0, pad_width), mode="constant", constant_values=np.nan))
        else:  # Handle NaNs or non-array items resulting from errors
            padded.append(np.full(max_len, np.nan))

    try:
        stacked = np.stack(padded, axis=0)
        return stacked
    except ValueError:  # Handle case where arrays have incompatible shapes despite padding attempt
        logger.error(f"Could not stack parsed arrays. Returning list of arrays/NaNs.")
        return parsed
    except Exception as e:
        logger.error(f"Error stacking parsed tensor/list column: {e}")
        return parsed  # Return list of arrays/NaNs as fallback


def flatten_analysis_dict(data_dict: Dict, parent_key: str = "", sep: str = "_") -> Dict:
    """Flattens a nested dictionary, handling tensors."""
    items = []
    if data_dict is None:
        return {}

    for k, v in data_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # Recursively flatten nested dictionaries
            items.extend(flatten_analysis_dict(v, new_key, sep=sep).items())
        elif isinstance(v, torch.Tensor):
            # Detach tensor, move to CPU, convert to scalar or list representation
            try:
                v_item = v.detach().cpu().item()  # Try converting to scalar first
            except ValueError:  # If tensor has more than one element
                try:
                    # Convert to numpy array then list for broader compatibility
                    v_item = v.detach().cpu().numpy().tolist()
                    # Optional: Convert list to string representation if storing in CSV causes issues
                    # v_item = str(v_item)
                except Exception as e:
                    logger.warning(f"Could not convert tensor '{new_key}' to list: {e}")
                    v_item = f"Tensor(shape={v.shape}, dtype={v.dtype})"  # Fallback string representation
            items.append((new_key, v_item))
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            # Handle lists of tensors (e.g., from multiple heads before aggregation)
            # Option 1: Store as string representation (safer for CSV)
            # items.append((new_key, str([f"Tensor(shape={t.shape})" for t in v])))
            # Option 2: Try to compute mean/std if numeric
            try:
                numeric_vals = [t.detach().cpu().item() for t in v]
                items.append((f"{new_key}_mean", np.nanmean(numeric_vals)))
                items.append((f"{new_key}_std", np.nanstd(numeric_vals)))
            except (ValueError, TypeError):  # If tensors are not scalar
                items.append((new_key, f"List[Tensor(len={len(v)})]"))  # Store placeholder description
        else:  # Store other types directly
            items.append((new_key, v))
    return dict(items)


def process_run_analysis_data(all_step_data_list: List[Dict], head_agg_method: str = "mean") -> pd.DataFrame:
    """
    Processes analysis data collected over multiple steps/batches for a full run.
    Creates a DataFrame where each row corresponds to a layer within a step.

    Args:
        all_step_data_list (list[dict]): List where each dict contains 'step', 'layers', etc.
        head_agg_method (str): How to aggregate metrics across heads ('mean', 'std', 'median', 'none').
                               If 'none', per-head data is not aggregated.

    Returns:
        pd.DataFrame: Processed and flattened analysis data.

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
        agg_func = None  # No aggregation

    for step_data in all_step_data_list:
        if not isinstance(step_data, dict):
            continue  # Skip invalid entries
        step = step_data.get(STEP, -1)
        main_loss = step_data.get("main_loss", np.nan)
        step_total_aux_loss = step_data.get("aux_loss", np.nan)
        layer_data_list = step_data.get("layers", [])  # List of dicts per layer

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
                    head_keys = pgs_head_data_list[0].keys()
                    for key in head_keys:
                        all_head_values = [h_data.get(key) for h_data in pgs_head_data_list]

                        # Try aggregation if requested and possible
                        if agg_func is not None:
                            numeric_values = []
                            valid_count = 0
                            for v in all_head_values:
                                try:
                                    numeric_values.append(float(v))
                                    valid_count += 1
                                except (ValueError, TypeError, Exception):
                                    numeric_values.append(np.nan)

                            if valid_count > 0 and not all(np.isnan(nv) for nv in numeric_values):
                                agg_val = agg_func(numeric_values)
                                flat_layer_data[f"pgs_head_agg_{key}"] = agg_val
                            elif key in [GEOM_MIX_WEIGHTS, ACTIVE_GEOM_BRANCHES]:  # Store head 0 for these complex types
                                flat_layer_data[f"pgs_head_0_{key}"] = all_head_values[0]
                        else:  # No aggregation - flatten per head
                            for h_idx, h_data in enumerate(pgs_head_data_list):
                                flat_layer_data.update(
                                    flatten_analysis_dict({key: h_data.get(key)}, parent_key=f"pgs_head{h_idx}_{key}")
                                )

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

    # Create DataFrame
    try:
        df = pd.DataFrame(processed_rows)
        # Convert numeric columns after creation
        for col in df.select_dtypes(include=["object"]).columns:  # Only check object columns
            try:
                # Attempt conversion, coercing errors
                converted = pd.to_numeric(df[col], errors="coerce")
                # Only overwrite if *some* conversion was successful (avoid converting pure string cols)
                if not converted.isna().all():
                    df[col] = converted
            except (ValueError, TypeError):
                pass  # Ignore errors

        logger.info(f"DataFrame created with shape {df.shape}")

        # --- Post-processing Specific Columns ---
        logger.info("Post-processing specific analysis columns...")
        # Geometry weights
        weight_col = f"pgs_head_0_{GEOM_MIX_WEIGHTS}"  # Using head 0 as representative
        branches_col = f"pgs_head_0_{ACTIVE_GEOM_BRANCHES}"
        if weight_col in df.columns and branches_col in df.columns:
            extracted_weights = False
            try:
                branches_lists = df[branches_col].apply(safe_literal_eval)
                weights_arrays = parse_tensor_list_column(df[weight_col])
                if isinstance(weights_arrays, np.ndarray):  # Check if parsing worked
                    all_branches = sorted(
                        list(set(b for sublist in branches_lists if isinstance(sublist, list) for b in sublist))
                    )
                    for branch in all_branches:
                        df[f"pgs_geom_weight_{branch}"] = np.nan  # Initialize
                    for i, row in df.iterrows():
                        branches = branches_lists.iloc[i]
                        weights = weights_arrays[i]
                        if isinstance(branches, list) and isinstance(weights, np.ndarray) and len(branches) == weights.shape[0]:
                            for branch_name, weight_val in zip(branches, weights):
                                if branch_name in all_branches:  # Check if branch name is expected
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
        grad_cols = [c for c in df.columns if GRADIENTS in c]  # Find gradient dict columns
        for grad_col in grad_cols:
            try:
                parsed_dicts = df[grad_col].apply(safe_literal_eval)
                valid_dicts = parsed_dicts[parsed_dicts.apply(lambda x: isinstance(x, dict))]
                if not valid_dicts.empty:
                    grad_df = pd.json_normalize(valid_dicts).reindex(df.index)
                    grad_df.columns = [
                        f"{grad_col.replace(GLOBAL_DATA + '_', '')}_{key}" for key in grad_df.columns
                    ]  # Add prefix
                    df = pd.concat([df.drop(columns=[grad_col]), grad_df], axis=1)
                    logger.info(f"Extracted gradient norms from {grad_col}")
            except Exception as e:
                logger.error(f"Error post-processing gradient norms for {grad_col}: {e}")

        logger.info("Analysis data processing complete.")
        return df
    except Exception as e:
        logger.error(f"Failed to create DataFrame: {e}", exc_info=True)
        return pd.DataFrame()
