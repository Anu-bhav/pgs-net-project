# src/pgs_net/utils/viz_utils.py
"""Plotting utilities for PGS-Net analysis data."""

import logging
import math
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Check availability
if plt is None or sns is None:
    logger.warning("Matplotlib or Seaborn not found. Plotting functions will be disabled.")

    # Define dummy functions if libraries are missing
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

else:
    # Set Seaborn style only if available
    sns.set_theme(style="whitegrid", palette="muted")

    # --- Plotting Functions ---

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
                plt.show()  # Show if saving failed
        else:
            plt.show()

    def plot_training_curves(df: pd.DataFrame, output_dir: Optional[str] = None):
        """Plots main loss, total aux loss vs. step."""
        if plt is None or df is None or df.empty:
            return
        logger.info("Plotting training curves...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        fig.suptitle("Training Curves", fontsize=14)

        # Aggregate step losses (use first occurrence per step)
        step_agg = df.groupby("step")[["step_main_loss", "step_total_aux_loss"]].first().reset_index()

        # Plot Main Loss
        if "step_main_loss" in step_agg.columns:
            sns.lineplot(data=step_agg, x="step", y="step_main_loss", label="Main Loss", ax=axes[0])
            # Add Val Loss here if available and aggregated similarly
            axes[0].set_title("Main Task Loss")
            axes[0].set_ylabel("Loss")
            axes[0].grid(True)
            axes[0].legend()
        else:
            axes[0].text(0.5, 0.5, "Main loss data missing", ha="center", va="center")

        # Plot Aux Loss
        aux_loss_col = "step_total_aux_loss"  # Key from processing
        if aux_loss_col in step_agg.columns:
            sns.lineplot(data=step_agg, x=aux_loss_col, y=aux_loss_col, label="Total Aux Loss", ax=axes[1])
            # Optional: Plot components like reg loss if globally available
            if "pgs_global_regularization_loss" in df.columns:
                reg_loss_agg = df.groupby("step")["pgs_global_regularization_loss"].first().reset_index()
                sns.lineplot(
                    data=reg_loss_agg, x="step", y="pgs_global_regularization_loss", label="Reg Loss", ax=axes[1], linestyle=":"
                )
            axes[1].set_title("Auxiliary Loss")
            axes[1].set_ylabel("Loss")
            axes[1].grid(True)
            axes[1].legend()
            axes[1].set_yscale("symlog", linthresh=1e-5)
        else:
            axes[1].text(0.5, 0.5, "Aux loss data missing", ha="center", va="center")

        axes[0].set_xlabel("Step")
        axes[1].set_xlabel("Step")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle
        _save_or_show(fig, os.path.join(output_dir, "training_curves.png") if output_dir else None)

    def plot_geometry_analysis(df: pd.DataFrame, output_dir: Optional[str] = None):
        """Plots geometry mixing weights and similarity temperatures."""
        if plt is None or df is None or df.empty:
            return
        logger.info("Plotting geometry analysis...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        fig.suptitle("Geometry Analysis", fontsize=14)

        # --- Geometry Mixing Weights (Line Plot) ---
        weight_cols = sorted([col for col in df.columns if col.startswith("pgs_geom_weight_")])
        ax = axes[0]
        if weight_cols:
            df_weights_agg = df.groupby("step")[weight_cols].mean().reset_index()  # Average over layers
            for col in weight_cols:
                branch_name = col.split("pgs_geom_weight_")[-1]
                sns.lineplot(data=df_weights_agg, x="step", y=col, label=branch_name, ax=ax)
            ax.set_title("Avg Geometry Mixing Weights")
            ax.set_xlabel("Step")
            ax.set_ylabel("Avg Weight")
            ax.legend(fontsize="small")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "Mixing weight data missing", ha="center", va="center")
            ax.set_title("Avg Geometry Mixing Weights")

        # --- Geometry Weights (Heatmap Layer vs Branch) ---
        ax = axes[1]
        if weight_cols and "layer_idx" in df.columns:
            try:
                # Average weights over all steps for each layer/branch
                pivot_df = df.pivot_table(index="layer_idx", columns=None, values=weight_cols, aggfunc=np.nanmean)
                pivot_df.columns = [c.split("pgs_geom_weight_")[-1] for c in pivot_df.columns]
                sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar=False, linewidths=0.5)
                ax.set_title("Avg Weights (Layer vs. Branch)")
                ax.set_xlabel("Geometry")
                ax.set_ylabel("Layer Index")
                ax.tick_params(axis="y", rotation=0)
            except Exception as e:
                logger.warning(f"Could not create geometry weights heatmap: {e}")
                ax.text(0.5, 0.5, "Heatmap Error", ha="center")
                ax.set_title("Avg Weights (Layer vs. Branch)")
        else:
            ax.text(0.5, 0.5, "Weight/Layer data missing", ha="center")
            ax.set_title("Avg Weights (Layer vs. Branch)")

        # --- Similarity Temperatures ---
        temp_cols = sorted([c for c in df.columns if "similarity_temps_effective" in c and "_agg" in c])  # Use aggregated temps
        ax = axes[2]
        if temp_cols:
            branch_names_col = f"pgs_head_0_{ACTIVE_GEOM_BRANCHES}"  # Need branch names
            branch_names = (
                safe_literal_eval(df[branch_names_col].iloc[0])
                if branch_names_col in df.columns
                else [f"Branch {i}" for i in range(len(temp_cols))]
            )

            for i, col in enumerate(temp_cols):
                step_data = df.groupby("step")[col].mean()
                sns.lineplot(
                    x=step_data.index,
                    y=step_data.values,
                    label=branch_names[i] if i < len(branch_names) else f"Branch {i}",
                    ax=ax,
                )
            ax.legend(fontsize="small")
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
            return
        logger.info("Plotting swarm dynamics analysis...")
        # Estimate number of plots needed based on available columns
        plot_keys = [
            "force_norms",
            "gate_dist",
            "gate_boxplot",
            "entropy_line",
            "dynk_line",
            "update_norms_line",
            "clip_scale_line",
            "condcomp_line",
            "grad_norms_line",
            "meta_params_line",
        ]
        available_plots = []
        # Check which data is present to decide plots
        if any(col.startswith("pgs_head_agg_force_") and "_norm_avg" in col for col in df.columns):
            available_plots.append("force_norms")
        if any(col.startswith("pgs_head_agg_gate_value_avg") for col in df.columns):
            available_plots.extend(["gate_dist", "gate_boxplot"])
        if any(col.startswith("pgs_head_agg_assignment_entropy") for col in df.columns):
            available_plots.append("entropy_line")
        if any(col.startswith("pgs_head_agg_dynamic_k_active_count") for col in df.columns):
            available_plots.append("dynk_line")
        if any(col.startswith("pgs_head_agg_update_norm_") for col in df.columns):
            available_plots.append("update_norms_line")
        if any(col.startswith("pgs_head_agg_clip_scale_avg") for col in df.columns):
            available_plots.append("clip_scale_line")
        if any(col.startswith("pgs_head_agg_conditional_skip_ratio") for col in df.columns):
            available_plots.append("condcomp_line")
        if any("_grads_" in col for col in df.columns):
            available_plots.append("grad_norms_line")
        if any(col.startswith("pgs_global_meta_params_") for col in df.columns):
            available_plots.append("meta_params_line")

        num_plots = len(available_plots)
        if num_plots == 0:
            logger.warning("No dynamics data found to plot.")
            return
        grid_cols = 3
        grid_rows = math.ceil(num_plots / grid_cols)
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 6, grid_rows * 5))
        axes = axes.flatten()  # Flatten grid for easy indexing
        plot_idx = 0
        fig.suptitle("PGS Dynamics Analysis", fontsize=14)

        def get_ax():
            nonlocal plot_idx
            if plot_idx < len(axes):
                ax = axes[plot_idx]
                plot_idx += 1
                return ax
            else:
                return None  # Should not happen if grid size is correct

        # --- Plotting Logic ---
        if "force_norms" in available_plots:
            ax = get_ax()
            force_cols = sorted([c for c in df.columns if c.startswith("pgs_head_agg_force_") and "_norm_avg" in c])
            df_melt = df.melt(id_vars=[STEP, LAYER_IDX], value_vars=force_cols, var_name="Force", value_name="Norm")
            df_melt["Force"] = df_melt["Force"].str.replace("pgs_head_agg_force_", "").str.replace("_norm_avg", "")
            sns.violinplot(data=df_melt, x=LAYER_IDX, y="Norm", hue="Force", ax=ax, cut=0, scale="width", inner="quartile")
            ax.set_title("Force Norms by Layer")
            ax.set_xlabel("Layer Index")
            ax.set_ylabel("Avg Norm")
            ax.legend(fontsize="x-small")
            ax.set_yscale("log")
            ax.grid(True)

        if "gate_dist" in available_plots:
            ax = get_ax()
            gate_col = next(c for c in df.columns if c.startswith("pgs_head_agg_gate_value_avg"))
            sns.histplot(df[gate_col].dropna(), kde=True, bins=30, ax=ax, stat="density")
            ax.set_title("Dist of Avg Gate Values")
            ax.set_xlabel("Gate Value")
            ax.set_xlim(-0.05, 1.05)

        if "gate_boxplot" in available_plots:
            ax = get_ax()
            gate_col = next(c for c in df.columns if c.startswith("pgs_head_agg_gate_value_avg"))
            if LAYER_IDX in df.columns:
                sns.boxplot(data=df, x=LAYER_IDX, y=gate_col, ax=ax, showfliers=False)
                ax.set_xlabel("Layer Index")
            else:
                sns.boxplot(data=df, y=gate_col, ax=ax, showfliers=False)
            ax.set_title("Gate Values by Layer")
            ax.set_ylabel("Avg Gate Value")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(axis="y")

        if "entropy_line" in available_plots:
            ax = get_ax()
            entropy_col = next(c for c in df.columns if c.startswith("pgs_head_agg_assignment_entropy"))
            if LAYER_IDX in df.columns:
                sns.lineplot(data=df, x=STEP, y=entropy_col, hue=LAYER_IDX, palette="coolwarm", legend=False, ax=ax)
            else:
                sns.lineplot(data=df, x=STEP, y=entropy_col, ax=ax)
            ax.set_title("Avg Assignment Entropy")
            ax.set_xlabel("Step")
            ax.set_ylabel("Entropy")
            ax.grid(True)

        if "dynk_line" in available_plots:
            ax = get_ax()
            dynk_col = next(c for c in df.columns if DYNK_ACTIVE_COUNT in c and "_agg" in c)
            if LAYER_IDX in df.columns:
                sns.lineplot(data=df, x=STEP, y=dynk_col, hue=LAYER_IDX, palette="coolwarm", legend=False, ax=ax)
            else:
                sns.lineplot(data=df, x=STEP, y=dynk_col, ax=ax)
            ax.set_title("Active Clusters (Dynamic K)")
            ax.set_xlabel("Step")
            ax.set_ylabel("Num Active Clusters")
            max_k = df[dynk_col].max()
            min_k = df[dynk_col].min()
            if not pd.isna(max_k) and not pd.isna(min_k):
                ax.set_ylim(max(0, min_k - 1), max_k + 1)
                ax.yaxis.get_major_locator().set_params(integer=True)

        if "update_norms_line" in available_plots:
            ax = get_ax()
            before_norm_col = (
                next((c for c in df.columns if "update_norm_before_norm" in c and "_agg" in c), None)
                or next((c for c in df.columns if "update_norm_after_clip" in c and "_agg" in c), None)
                or next((c for c in df.columns if "update_norm_after_dropout" in c and "_agg" in c), None)
            )  # Find best available "before" norm
            after_norm_col = next((c for c in df.columns if "update_norm_after_norm" in c and "_agg" in c), None)
            if before_norm_col:
                sns.lineplot(x=df[STEP], y=df[before_norm_col], label="Before Norm", ax=ax, errorbar="sd")
            if after_norm_col:
                sns.lineplot(x=df[STEP], y=df[after_norm_col], label="After Norm", ax=ax, errorbar="sd")
            ax.set_title("Avg Update Vector Norm")
            ax.set_xlabel("Step")
            ax.set_ylabel("Average Norm")
            if before_norm_col or after_norm_col:
                ax.legend()
                ax.set_yscale("log")
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, "Update norm data missing", ha="center")

        if "clip_scale_line" in available_plots:
            ax = get_ax()
            clip_scale_col = next((c for c in df.columns if "clip_scale_avg" in c and "_agg" in c), None)
            if clip_scale_col:
                sns.lineplot(
                    data=df,
                    x=STEP,
                    y=clip_scale_col,
                    hue=LAYER_IDX if LAYER_IDX in df.columns else None,
                    palette="coolwarm",
                    legend=False,
                    ax=ax,
                )
                ax.set_title("Avg Update Clipping Scale")
                ax.set_xlabel("Step")
                ax.set_ylabel("Scale (<=1)")
                ax.set_ylim(0, 1.1)
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, "Clip scale data missing", ha="center")
                ax.set_title("Avg Update Clipping Scale")

        if "condcomp_line" in available_plots:
            ax = get_ax()
            skip_col = next((c for c in df.columns if "conditional_skip_ratio" in c and "_agg" in c), None)
            if skip_col:
                sns.lineplot(
                    data=df,
                    x=STEP,
                    y=skip_col,
                    hue=LAYER_IDX if LAYER_IDX in df.columns else None,
                    palette="coolwarm",
                    legend=False,
                    ax=ax,
                )
                ax.set_title("Cond. Comp Skip Ratio")
                ax.set_xlabel("Step")
                ax.set_ylabel("Skip Ratio")
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, "CondComp data missing", ha="center")
                ax.set_title("Cond. Comp Skip Ratio")

        if "grad_norms_line" in available_plots:
            ax = get_ax()
            grad_cols = sorted([c for c in df.columns if "_grads_" in c])  # Assumes gradients processed into columns
            if grad_cols:
                df_melt = df.melt(id_vars=[STEP], value_vars=grad_cols, var_name="Parameter", value_name="Gradient Norm")
                df_melt["Parameter"] = df_melt["Parameter"].str.replace("pgs_global_gradients_", "", regex=False)  # Clean name
                sns.lineplot(data=df_melt, x=STEP, y="Gradient Norm", hue="Parameter", ax=ax, errorbar="sd")
                ax.set_title("Gradient Norms")
                ax.set_xlabel("Step")
                ax.set_ylabel("Norm")
                ax.legend(fontsize="xx-small", loc="upper right")
                ax.set_yscale("log")
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, "Gradient data missing", ha="center")
                ax.set_title("Gradient Norms")

        if "meta_params_line" in available_plots:
            ax = get_ax()
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
            else:
                ax.text(0.5, 0.5, "Meta Param data missing", ha="center")
                ax.set_title("Meta-Learned Parameters")

        # Hide unused axes
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        _save_or_show(fig, os.path.join(output_dir, "dynamics_analysis.png") if output_dir else None)

    def plot_correlation_heatmap(df: pd.DataFrame, metrics: Optional[List[str]] = None, output_dir: Optional[str] = None):
        """Plots a heatmap of correlations between selected metrics."""
        if plt is None or df is None or df.empty:
            return
        logger.info("Plotting correlation heatmap...")

        if metrics is None:  # Select some default numeric metrics if not provided
            metrics = [
                c for c in df.select_dtypes(include=np.number).columns if not c.startswith("pgs_geom_weight_") and "_norm" in c
            ]  # Example: norms
            metrics += [c for c in df.columns if "entropy" in c or "loss" in c or "skip_ratio" in c or "clip_scale" in c]
            metrics = list(set(metrics))  # Unique
            if not metrics:
                logger.warning("No suitable metrics found for correlation heatmap.")
                return

        corr_df = df[metrics].corr()
        plt.figure(figsize=(max(8, len(metrics) * 0.6), max(6, len(metrics) * 0.5)))
        sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, annot_kws={"size": 8})
        plt.title("Metric Correlation Heatmap")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        _save_or_show(fig, os.path.join(output_dir, "correlation_heatmap.png") if output_dir else None)

    def plot_embedding_visualization(
        embeddings_proj: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "Embedding Visualization",
        output_dir: Optional[str] = None,
    ):
        """Plots 2D/3D projected embeddings using scatter plot."""
        if plt is None or embeddings_proj is None:
            logger.warning("Cannot plot embeddings.")
            return
        if not isinstance(embeddings_proj, np.ndarray):
            embeddings_proj = embeddings_proj.numpy()  # Convert tensor if needed
        if labels is not None and not isinstance(labels, np.ndarray):
            labels = labels.numpy()

        logger.info(f"Plotting embedding visualization: {title}")
        dims = embeddings_proj.shape[1]
        fig = plt.figure(figsize=(8, 8) if dims == 2 else (10, 10))

        if dims == 2:
            ax = fig.add_subplot(111)
            sns.scatterplot(
                x=embeddings_proj[:, 0],
                y=embeddings_proj[:, 1],
                hue=labels,
                palette="viridis",
                s=10,
                alpha=0.7,
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
                cmap="viridis",
                s=10,
                alpha=0.6,
                depthshade=True,
            )
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.set_zlabel("Dim 3")
            if labels is not None and len(np.unique(labels)) < 20:  # Add legend only if reasonable number of classes
                legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
                ax.add_artist(legend1)
        else:
            logger.warning("Can only plot 2D or 3D embeddings.")
            plt.close(fig)
            return

        ax.set_title(title)
        ax.grid(True)
        plt.tight_layout()
        _save_or_show(fig, os.path.join(output_dir, f"{title.replace(' ', '_').replace('/', '_')}.png") if output_dir else None)
