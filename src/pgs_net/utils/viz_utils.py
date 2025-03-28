# src/pgs_net/utils/viz_utils.py
"""Plotting utilities for PGS-Net analysis data using Matplotlib and Seaborn."""

import logging
import math
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Check optional dependencies
plot_libs_available = False
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted")  # Set Seaborn style
    plot_libs_available = True
    logger.debug("Matplotlib and Seaborn loaded.")
except ImportError:
    logger.warning("Matplotlib or Seaborn not found. Plotting functions will be disabled.")

    # Define dummy functions
    def plot_training_curves(*args, **kwargs):
        logger.warning("Plotting disabled.")
        pass

    def plot_geometry_analysis(*args, **kwargs):
        logger.warning("Plotting disabled.")
        pass

    def plot_dynamics_analysis(*args, **kwargs):
        logger.warning("Plotting disabled.")
        pass

    def plot_embedding_visualization(*args, **kwargs):
        logger.warning("Plotting disabled.")
        pass

    def plot_correlation_heatmap(*args, **kwargs):
        logger.warning("Plotting disabled.")
        pass


if plot_libs_available:
    # --- Constants from analysis_processing ---
    STEP = "step"
    LAYER_IDX = "layer_idx"

    # --- Helper Functions ---
    def _get_agg_key(base_key: str, df: pd.DataFrame) -> Optional[str]:
        """Finds the aggregated column name for a base metric key."""
        agg_key = f"pgs_head_agg_{base_key}"
        if agg_key in df.columns:
            return agg_key
        # Fallback: check if head 0 data exists directly (e.g., for non-numeric)
        head0_key = f"pgs_head_0_{base_key}"
        if head0_key in df.columns:
            return head0_key
        # Fallback: check global scope if not found in heads
        global_key = f"pgs_global_{base_key}"
        if global_key in df.columns:
            return global_key
        # Fallback: check for raw key if flattening was simple
        if base_key in df.columns:
            return base_key
        return None

    def _save_or_show(fig, output_path: Optional[str] = None):
        """Helper to save or show plot."""
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                fig.savefig(output_path, dpi=150, bbox_inches="tight")
                logger.info(f"Plot saved to {output_path}")
                plt.close(fig)  # Close figure after saving
            except Exception as e:
                logger.error(f"Failed to save plot to {output_path}: {e}")
                try:
                    plt.show()  # Try showing if saving failed
                except Exception:
                    pass  # Avoid crashing if show fails too (e.g., no GUI)
        else:
            try:
                plt.show()
            except Exception as e:
                logger.error(f"Failed to show plot: {e}")
            plt.close(fig)  # Close after showing

    # --- Plotting Functions ---

    def plot_training_curves(df: pd.DataFrame, output_dir: Optional[str] = None):
        """Plots main loss, total aux loss vs. step."""
        if df is None or df.empty:
            logger.warning("Cannot plot curves: No data.")
            return
        logger.info("Plotting training curves...")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
        fig.suptitle("Training Curves", fontsize=14)
        has_data = False

        # Aggregate step losses (use first occurrence per step)
        step_agg = df.groupby(STEP)[["step_main_loss", "step_total_aux_loss"]].first().reset_index()

        # Plot Main Loss
        if "step_main_loss" in step_agg.columns and not step_agg["step_main_loss"].isna().all():
            sns.lineplot(data=step_agg, x=STEP, y="step_main_loss", label="Main Loss", ax=axes[0], errorbar="sd")
            axes[0].set_title("Main Task Loss")
            axes[0].set_ylabel("Loss")
            axes[0].grid(True)
            axes[0].legend()
            has_data = True
        else:
            axes[0].text(0.5, 0.5, "Main loss data missing", ha="center", va="center")
            axes[0].set_title("Main Task Loss")

        # Plot Aux Loss
        aux_loss_col = "step_total_aux_loss"
        reg_loss_col = _get_agg_key("regularization_loss", df)  # Find regularization loss column
        if aux_loss_col in step_agg.columns and not step_agg[aux_loss_col].isna().all():
            sns.lineplot(data=step_agg, x=STEP, y=aux_loss_col, label="Total Aux Loss", ax=axes[1], errorbar="sd")
            # Plot regularization loss if available
            if reg_loss_col and reg_loss_col in df.columns:
                reg_loss_agg = df.groupby(STEP)[reg_loss_col].first().reset_index()  # Use first (should be global)
                if not reg_loss_agg[reg_loss_col].isna().all():
                    sns.lineplot(data=reg_loss_agg, x=STEP, y=reg_loss_col, label="Reg Loss", ax=axes[1], linestyle=":")
            axes[1].set_title("Auxiliary Loss")
            axes[1].set_ylabel("Loss")
            axes[1].grid(True)
            axes[1].legend()
            axes[1].set_yscale("symlog", linthresh=1e-5)
            has_data = True
        else:
            axes[1].text(0.5, 0.5, "Aux loss data missing", ha="center", va="center")
            axes[1].set_title("Auxiliary Loss")

        if not has_data:
            logger.warning("No data found for training curves plot.")
            plt.close(fig)
            return
        axes[0].set_xlabel("Step")
        axes[1].set_xlabel("Step")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        _save_or_show(fig, os.path.join(output_dir, "training_curves.png") if output_dir else None)

    def plot_geometry_analysis(df: pd.DataFrame, output_dir: Optional[str] = None):
        """Plots geometry mixing weights and similarity temperatures."""
        if df is None or df.empty:
            logger.warning("Cannot plot geometry: No data.")
            return
        logger.info("Plotting geometry analysis...")
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
        fig.suptitle("Geometry Analysis", fontsize=14)
        plot_idx = 0

        # --- Geometry Mixing Weights (Line Plot) ---
        ax = axes[plot_idx]
        plot_idx += 1
        weight_cols = sorted([col for col in df.columns if col.startswith("pgs_geom_weight_")])
        if weight_cols:
            df_weights_agg = df.groupby(STEP)[weight_cols].mean().reset_index()  # Average over layers per step
            if not df_weights_agg.empty:
                for col in weight_cols:
                    branch_name = col.split("pgs_geom_weight_")[-1]
                    sns.lineplot(data=df_weights_agg, x=STEP, y=col, label=branch_name, ax=ax)
                ax.set_title("Avg Geometry Mixing Weights")
                ax.set_xlabel("Step")
                ax.set_ylabel("Avg Weight")
                ax.legend(fontsize="small")
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, "Aggregated weight data empty", ha="center")
                ax.set_title("Avg Geometry Mixing Weights")
        else:
            ax.text(0.5, 0.5, "Mixing weight data missing", ha="center")
            ax.set_title("Avg Geometry Mixing Weights")

        # --- Geometry Weights (Heatmap Layer vs Branch) ---
        ax = axes[plot_idx]
        plot_idx += 1
        if weight_cols and LAYER_IDX in df.columns:
            try:
                # Average weights over all steps for each layer/branch
                pivot_df = df.pivot_table(index=LAYER_IDX, columns=None, values=weight_cols, aggfunc=np.nanmean)
                if not pivot_df.empty and not pivot_df.isna().all().all():  # Check if pivot is valid
                    pivot_df.columns = [c.split("pgs_geom_weight_")[-1] for c in pivot_df.columns]
                    sns.heatmap(
                        pivot_df, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar=True, linewidths=0.5, linecolor="lightgray"
                    )
                    ax.set_title("Avg Weights (Layer vs. Branch)")
                    ax.set_xlabel("Geometry")
                    ax.set_ylabel("Layer Index")
                    ax.tick_params(axis="y", rotation=0)
                else:
                    logger.warning("Pivot table for geometry weights heatmap is empty or all NaN.")
                    ax.text(0.5, 0.5, "Heatmap data invalid", ha="center")
                    ax.set_title("Avg Weights (Layer vs. Branch)")
            except Exception as e:
                logger.warning(f"Could not create geometry weights heatmap: {e}")
                ax.text(0.5, 0.5, "Heatmap Error", ha="center")
                ax.set_title("Avg Weights (Layer vs. Branch)")
        else:
            ax.text(0.5, 0.5, "Weight/Layer data missing", ha="center")
            ax.set_title("Avg Weights (Layer vs. Branch)")

        # --- Similarity Temperatures ---
        ax = axes[plot_idx]
        plot_idx += 1
        temp_col_agg = _get_agg_key("similarity_temps_effective", df)  # Find aggregated temp column
        branch_names_col = f"pgs_head_0_{ACTIVE_GEOM_BRANCHES}"  # Get branches from head 0 data

        if temp_col_agg and temp_col_agg in df.columns:
            try:
                temp_data = parse_tensor_list_column(df[temp_col_agg])  # Parse the column
                if isinstance(temp_data, np.ndarray):
                    temp_df = pd.DataFrame(temp_data, index=df[STEP])  # Use step as index
                    temp_df = temp_df.groupby(level=0).mean()  # Average over layers/duplicates per step

                    branch_names = None
                    if branch_names_col in df.columns:
                        # Try getting branch names from first valid entry
                        first_valid_branches = df[branches_col].dropna().apply(safe_literal_eval).iloc[0]
                        if isinstance(first_valid_branches, list):
                            branch_names = first_valid_branches

                    num_branches_found = temp_df.shape[1]
                    for i in range(num_branches_found):
                        label = branch_names[i] if branch_names and i < len(branch_names) else f"Branch {i}"
                        sns.lineplot(data=temp_df, x=temp_df.index, y=i, label=label, ax=ax)
                    ax.legend(fontsize="small")
                else:
                    ax.text(0.5, 0.5, "Temp data parse/stack failed", ha="center")
            except Exception as e:
                logger.error(f"Error plotting temps: {e}")
                ax.text(0.5, 0.5, "Plotting Error", ha="center")
        else:
            ax.text(0.5, 0.5, "Aggregated Temp data missing", ha="center")
        ax.set_title("Avg Similarity Temps (Tau)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Temperature")
        ax.set_yscale("log")
        ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        _save_or_show(fig, os.path.join(output_dir, "geometry_analysis.png") if output_dir else None)

    def plot_dynamics_analysis(df: pd.DataFrame, output_dir: Optional[str] = None):
        """Plots force norms, gate values, entropy, DynK count, update norms, clipping, gradients etc."""
        if plt is None or df is None or df.empty:
            logger.warning("Cannot plot dynamics: No data.")
            return
        logger.info("Plotting swarm dynamics analysis...")

        # Find available metrics to plot
        plot_functions = {
            "force_norms": lambda ax: _plot_force_norms(df, ax),
            "gate_dist": lambda ax: _plot_gate_dist(df, ax),
            "gate_boxplot": lambda ax: _plot_gate_boxplot(df, ax),
            "entropy_line": lambda ax: _plot_entropy(df, ax),
            "dynk_line": lambda ax: _plot_dynk(df, ax),
            "update_norms_line": lambda ax: _plot_update_norms(df, ax),
            "clip_scale_line": lambda ax: _plot_clip_scale(df, ax),
            "condcomp_line": lambda ax: _plot_condcomp(df, ax),
            "grad_norms_line": lambda ax: _plot_grad_norms(df, ax),
            "meta_params_line": lambda ax: _plot_meta_params(df, ax),
        }
        available_plots = [key for key, func in plot_functions.items() if func(None)]  # Check if data exists via helper funcs

        num_plots = len(available_plots)
        if num_plots == 0:
            logger.warning("No dynamics data found to plot.")
            return
        grid_cols = 3
        grid_rows = math.ceil(num_plots / grid_cols)
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 5.5, grid_rows * 5))
        axes = axes.flatten()
        plot_idx = 0
        fig.suptitle("PGS Dynamics Analysis", fontsize=14)

        for plot_key in available_plots:
            if plot_idx < len(axes):
                logger.debug(f"Generating dynamics plot: {plot_key}")
                ax = axes[plot_idx]
                try:
                    plot_functions[plot_key](ax)  # Call helper plotting function
                    plot_idx += 1
                except Exception as e:
                    logger.error(f"Failed to generate plot '{plot_key}': {e}", exc_info=True)
                    ax.text(0.5, 0.5, f"Plot Error:\n{plot_key}", ha="center", va="center")
                    plot_idx += 1  # Move to next subplot even on error

        # Hide unused axes
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        _save_or_show(fig, os.path.join(output_dir, "dynamics_analysis.png") if output_dir else None)

    # --- Helper functions for plot_dynamics_analysis ---
    def _plot_force_norms(df, ax):
        force_cols = sorted([c for c in df.columns if c.startswith("pgs_head_agg_force_") and "_norm_avg" in c])
        if not force_cols or LAYER_IDX not in df.columns:
            return False  # Data not available
        df_melt = df.melt(id_vars=[STEP, LAYER_IDX], value_vars=force_cols, var_name="Force", value_name="Norm")
        df_melt["Force"] = df_melt["Force"].str.replace("pgs_head_agg_force_", "").str.replace("_norm_avg", "")
        sns.violinplot(data=df_melt, x=LAYER_IDX, y="Norm", hue="Force", ax=ax, cut=0, scale="width", inner="quartile")
        ax.set_title("Force Component Norms by Layer")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Average Norm")
        ax.legend(fontsize="xx-small", loc="upper right")
        ax.set_yscale("log")
        ax.grid(True)
        return True

    def _plot_gate_dist(df, ax):
        gate_col = _get_agg_key("gate_value_avg", df)
        if gate_col and not df[gate_col].isna().all():
            sns.histplot(df[gate_col].dropna(), kde=True, bins=30, ax=ax, stat="density")
            ax.set_title("Dist of Avg Gate Values")
            ax.set_xlabel("Gate Value")
            ax.set_xlim(-0.05, 1.05)
            return True
        return False

    def _plot_gate_boxplot(df, ax):
        gate_col = _get_agg_key("gate_value_avg", df)
        if gate_col and LAYER_IDX in df.columns and not df[gate_col].isna().all():
            sns.boxplot(data=df, x=LAYER_IDX, y=gate_col, ax=ax, showfliers=False)
            ax.set_title("Gate Values by Layer")
            ax.set_xlabel("Layer Index")
            ax.set_ylabel("Avg Gate Value")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(axis="y")
            return True
        return False

    def _plot_entropy(df, ax):
        entropy_col = _get_agg_key(ASSIGNMENT_ENTROPY, df)
        if entropy_col and not df[entropy_col].isna().all():
            hue = LAYER_IDX if LAYER_IDX in df.columns else None
            sns.lineplot(data=df, x=STEP, y=entropy_col, hue=hue, palette="coolwarm", legend=False, ax=ax, errorbar=("ci", 95))
            ax.set_title("Avg Assignment Entropy")
            ax.set_xlabel("Step")
            ax.set_ylabel("Entropy")
            ax.grid(True)
            return True
        return False

    def _plot_dynk(df, ax):
        dynk_col = _get_agg_key(DYNK_ACTIVE_COUNT, df)
        if dynk_col and not df[dynk_col].isna().all():
            hue = LAYER_IDX if LAYER_IDX in df.columns else None
            sns.lineplot(
                data=df, x=STEP, y=dynk_col, hue=hue, palette="coolwarm", legend=False, ax=ax, drawstyle="steps-post"
            )  # Use steps
            ax.set_title("Active Clusters (Dynamic K)")
            ax.set_xlabel("Step")
            ax.set_ylabel("Num Active Clusters")
            max_k = df[dynk_col].max()
            min_k = df[dynk_col].min()
            if not pd.isna(max_k) and not pd.isna(min_k):
                ax.set_ylim(max(0, min_k - 1), max_k + 1)
                ax.yaxis.get_major_locator().set_params(integer=True)
            ax.grid(True)
            return True
        return False

    def _plot_update_norms(df, ax):
        before_col = _get_agg_key("update_norm_before_norm", df) or _get_agg_key(
            "update_norm_after_clip", df
        )  # Find best "before" norm
        after_col = _get_agg_key("update_norm_after_norm", df)
        plotted = False
        if before_col and not df[before_col].isna().all():
            sns.lineplot(x=df[STEP], y=df[before_col], label="Before Norm", ax=ax, errorbar="sd")
            plotted = True
        if after_col and not df[after_col].isna().all():
            sns.lineplot(x=df[STEP], y=df[after_col], label="After Norm", ax=ax, errorbar="sd")
            plotted = True
        if plotted:
            ax.set_title("Avg Update Vector Norm")
            ax.set_xlabel("Step")
            ax.set_ylabel("Average Norm")
            ax.legend()
            ax.set_yscale("log")
            ax.grid(True)
        return plotted

    def _plot_clip_scale(df, ax):
        clip_scale_col = _get_agg_key("clip_scale_avg", df)
        if clip_scale_col and not df[clip_scale_col].isna().all():
            hue = LAYER_IDX if LAYER_IDX in df.columns else None
            sns.lineplot(data=df, x=STEP, y=clip_scale_col, hue=hue, palette="coolwarm", legend=False, ax=ax, errorbar="sd")
            ax.set_title("Avg Update Clipping Scale")
            ax.set_xlabel("Step")
            ax.set_ylabel("Scale (<=1)")
            ax.set_ylim(0, 1.1)
            ax.grid(True)
            return True
        return False

    def _plot_condcomp(df, ax):
        skip_col = _get_agg_key("conditional_skip_ratio", df)
        if skip_col and not df[skip_col].isna().all():
            hue = LAYER_IDX if LAYER_IDX in df.columns else None
            sns.lineplot(data=df, x=STEP, y=skip_col, hue=hue, palette="coolwarm", legend=False, ax=ax, errorbar="sd")
            ax.set_title("Cond. Comp Skip Ratio")
            ax.set_xlabel("Step")
            ax.set_ylabel("Skip Ratio")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True)
            return True
        return False

    def _plot_grad_norms(df, ax):
        grad_cols = sorted([c for c in df.columns if "_grad_" in c])  # Find gradient columns extracted by processing
        if grad_cols:
            df_melt = df.melt(id_vars=[STEP], value_vars=grad_cols, var_name="Parameter", value_name="Gradient Norm")
            # Clean names: remove prefix like 'pgs_global_gradients_'
            df_melt["Parameter"] = df_melt["Parameter"].str.replace(r"^.*_grad_", "", regex=True)
            sns.lineplot(data=df_melt, x=STEP, y="Gradient Norm", hue="Parameter", ax=ax, errorbar="sd")
            ax.set_title("Gradient Norms")
            ax.set_xlabel("Step")
            ax.set_ylabel("Norm")
            ax.legend(fontsize="xx-small", loc="center left", bbox_to_anchor=(1, 0.5))
            ax.set_yscale("log")
            ax.grid(True)
            return True
        return False

    def _plot_meta_params(df, ax):
        meta_cols = sorted([c for c in df.columns if c.startswith("pgs_global_meta_params_")])
        if meta_cols:
            for col in meta_cols:
                param_name = col.split("pgs_global_meta_params_")[-1]
                sns.lineplot(x=df[STEP], y=df[col], label=param_name, ax=ax)
            ax.set_title("Meta-Learned Parameters")
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.legend(fontsize="small")
            ax.grid(True)
            return True
        return False

    # --- End Helper functions ---

    def plot_correlation_heatmap(
        df: pd.DataFrame, metrics: Optional[List[str]] = None, layer: Optional[int] = None, output_dir: Optional[str] = None
    ):
        """Plots a heatmap of correlations between selected metrics, optionally for a specific layer."""
        if plt is None or df is None or df.empty:
            logger.warning("Cannot plot heatmap: No data.")
            return
        logger.info(f"Plotting correlation heatmap (Layer: {'All' if layer is None else layer})...")

        df_plot = df if layer is None else df[df[LAYER_IDX] == layer]
        if df_plot.empty:
            logger.warning(f"No data found for layer {layer} in correlation plot.")
            return

        if metrics is None:  # Select some default numeric metrics
            metrics = sorted([
                c
                for c in df_plot.select_dtypes(include=np.number).columns
                if c not in [STEP, LAYER_IDX]
                and not c.startswith("pgs_geom_weight_")
                and not c.endswith("_id")
                and not df_plot[c].isna().all()
            ])
            # Limit number of metrics for readability
            max_metrics = 30
            if len(metrics) > max_metrics:
                logger.warning(f"Too many metrics ({len(metrics)}), selecting subset for correlation heatmap.")
                metrics = metrics[:max_metrics]
            if not metrics:
                logger.warning("No suitable metrics found for correlation heatmap.")
                return

        corr_df = df_plot[metrics].corr()
        plt.figure(figsize=(min(30, max(10, len(metrics) * 0.8)), min(25, max(8, len(metrics) * 0.7))))
        sns.heatmap(
            corr_df,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            annot_kws={"size": 8 if len(metrics) < 20 else 6},
            vmin=-1,
            vmax=1,
        )
        plt.title(f"Metric Correlation Heatmap (Layer: {'Avg' if layer is None else layer})")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        fname = f"correlation_heatmap{'_layer' + str(layer) if layer is not None else '_avg'}.png"
        _save_or_show(fig, os.path.join(output_dir, fname) if output_dir else None)

    def plot_embedding_visualization(
        embeddings_proj: Optional[np.ndarray],
        labels: Optional[np.ndarray] = None,
        title: str = "Embedding Visualization",
        output_dir: Optional[str] = None,
        **kwargs,
    ):
        """Plots 2D/3D projected embeddings using scatter plot."""
        if plt is None or embeddings_proj is None:
            logger.warning("Cannot plot embeddings.")
            return
        if isinstance(embeddings_proj, torch.Tensor):
            embeddings_proj = embeddings_proj.detach().cpu().numpy()
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if not isinstance(embeddings_proj, np.ndarray) or embeddings_proj.ndim != 2:
            logger.error(f"Invalid embedding shape for plotting: {embeddings_proj.shape}")
            return

        logger.info(f"Plotting embedding visualization: {title}")
        dims = embeddings_proj.shape[1]
        fig = plt.figure(figsize=(9, 8) if dims == 2 else (10, 9))

        if dims == 2:
            ax = fig.add_subplot(111)
            sns.scatterplot(
                x=embeddings_proj[:, 0],
                y=embeddings_proj[:, 1],
                hue=labels,
                palette=kwargs.get("palette", "viridis"),
                s=kwargs.get("s", 10),
                alpha=kwargs.get("alpha", 0.6),
                legend="auto" if labels is not None else None,
                ax=ax,
            )
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
        elif dims == 3:
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(
                embeddings_proj[:, 0],
                embeddings_proj[:, 1],
                embeddings_proj[:, 2],
                c=labels,
                cmap=kwargs.get("cmap", "viridis"),
                s=kwargs.get("s", 10),
                alpha=kwargs.get("alpha", 0.5),
                depthshade=True,
            )
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.set_zlabel("Dim 3")
            if labels is not None and len(np.unique(labels)) < 20:  # Add legend only if reasonable number of classes
                try:
                    legend1 = ax.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.1, 1))
                    ax.add_artist(legend1)
                except Exception as e:
                    logger.warning(f"Could not create 3D scatter legend: {e}")
        else:
            logger.warning("Can only plot 2D or 3D embeddings.")
            plt.close(fig)
            return

        ax.set_title(title)
        ax.grid(True)
        plt.tight_layout()
        safe_title = title.replace(" ", "_").replace("/", "_").replace(":", "").replace("\\", "")
        _save_or_show(fig, os.path.join(output_dir, f"{safe_title}.png") if output_dir else None)
