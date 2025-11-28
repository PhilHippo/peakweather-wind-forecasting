"""
Probabilistic Forecast Visualization Utilities

This module provides reusable functions for creating beautiful probabilistic forecast plots
similar to ICON ensemble visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Optional, Tuple, List
import torch


def plot_probabilistic_forecast(
    ensemble_predictions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    input_history: Optional[np.ndarray] = None,
    lead_times: Optional[np.ndarray] = None,
    station_name: str = "Station",
    reference_time: Optional[str] = None,
    title: Optional[str] = None,
    show_control: bool = False,
    show_individual_members: bool = True,
    show_mean: bool = True,
    show_median: bool = False,
    show_uncertainty_bands: bool = True,
    uncertainty_quantiles: Tuple[float, float] = (0.025, 0.975),  # 95% interval
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[str] = None,
    ylabel: str = "Temperature (°C)",
    color_scheme: str = "Blues",
) -> plt.Figure:
    """
    Create a beautiful probabilistic forecast plot.
    
    Args:
        ensemble_predictions: Array of shape (n_lead_times, n_ensemble_members) or (n_ensemble_members, n_lead_times)
        ground_truth: Optional array of shape (n_lead_times,) with actual observations
        input_history: Optional array of shape (n_history_steps,) with historical observations
        lead_times: Optional array of lead time indices. If None, uses 0, 1, 2, ...
        station_name: Name of the station for plot title
        reference_time: Reference time string (e.g., "2024-10-10 18:00")
        title: Custom title. If None, generates default title
        show_control: If True, highlights the first ensemble member as "Control"
        show_individual_members: If True, plots all ensemble members
        show_mean: If True, plots the ensemble mean
        show_median: If True, plots the ensemble median
        show_uncertainty_bands: If True, shows uncertainty bands (default: 95% interval)
        uncertainty_quantiles: Tuple of lower and upper quantiles for uncertainty bands
        figsize: Figure size (width, height)
        save_path: If provided, saves figure to this path
        ylabel: Label for y-axis
        color_scheme: Matplotlib colormap name for ensemble members
        
    Returns:
        matplotlib Figure object
    """
    # Ensure correct shape (n_lead_times, n_ensemble_members)
    if ensemble_predictions.ndim != 2:
        raise ValueError(f"ensemble_predictions must be 2D, got shape {ensemble_predictions.shape}")
    
    # Handle both shapes
    if ensemble_predictions.shape[0] > ensemble_predictions.shape[1]:
        # Already (n_lead_times, n_ensemble_members)
        ensembles = ensemble_predictions
    else:
        # Transpose from (n_ensemble_members, n_lead_times)
        ensembles = ensemble_predictions.T
    
    n_lead_times, n_ensembles = ensembles.shape
    
    # Set up lead times
    if lead_times is None:
        lead_times = np.arange(n_lead_times)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get colors for ensemble members
    colors = cm.get_cmap(color_scheme)(np.linspace(0.4, 0.9, n_ensembles - 1 if show_control else n_ensembles))
    
    # Plot input history if provided
    history_offset = 0
    if input_history is not None:
        n_history = len(input_history)
        history_times = np.arange(-n_history, 0)
        ax.plot(history_times, input_history, color='gray', linewidth=2, 
                label='Historical', alpha=0.7, linestyle='-')
        ax.axvline(x=0, color='red', linestyle=':', linewidth=2, alpha=0.5)
        history_offset = -n_history
    
    # Plot individual ensemble members
    if show_individual_members:
        for i in range(n_ensembles):
            if show_control and i == 0:
                # Plot control member differently
                ax.plot(lead_times, ensembles[:, i], color='black', 
                       linewidth=1.5, label='Control', alpha=0.8, zorder=5)
            else:
                label = 'Ensemble Members' if (i == 1 and show_control) or (i == 0 and not show_control) else None
                color_idx = i - 1 if show_control else i
                ax.plot(lead_times, ensembles[:, i], 
                       color=colors[color_idx] if color_idx < len(colors) else colors[-1],
                       linewidth=1, alpha=0.4, label=label, zorder=2)
    
    # Compute statistics
    mean = np.mean(ensembles, axis=1)
    median = np.median(ensembles, axis=1)
    std = np.std(ensembles, axis=1)
    lower = np.quantile(ensembles, uncertainty_quantiles[0], axis=1)
    upper = np.quantile(ensembles, uncertainty_quantiles[1], axis=1)
    
    # Plot uncertainty bands
    if show_uncertainty_bands:
        ax.fill_between(lead_times, lower, upper, 
                       color='grey', alpha=0.2, 
                       label=f'{int((uncertainty_quantiles[1]-uncertainty_quantiles[0])*100)}% Interval',
                       zorder=1)
    
    # Plot ensemble mean
    if show_mean:
        ax.plot(lead_times, mean, color='red', linewidth=2.5, 
               label='Ensemble Mean', alpha=0.9, zorder=10)
    
    # Plot ensemble median
    if show_median:
        ax.plot(lead_times, median, color='orange', linewidth=2.5, 
               label='Ensemble Median', alpha=0.9, linestyle='--', zorder=10)
    
    # Plot ground truth if provided
    if ground_truth is not None:
        # Handle different shapes
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.cpu().numpy()
        if ground_truth.ndim > 1:
            ground_truth = ground_truth.squeeze()
        
        # Ensure correct length
        gt_len = min(len(ground_truth), len(lead_times))
        ax.plot(lead_times[:gt_len], ground_truth[:gt_len], color='darkgreen', 
               linewidth=2.5, label='Ground Truth', marker='o', markersize=4,
               alpha=0.9, zorder=15)
    
    # Formatting
    if title is None:
        title = f"Probabilistic Forecast - {station_name}"
        if reference_time:
            title += f" ({reference_time})"
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Lead Time (hours)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_multistation_forecast(
    ensemble_predictions: np.ndarray,
    station_names: List[str],
    ground_truth: Optional[np.ndarray] = None,
    lead_times: Optional[np.ndarray] = None,
    reference_time: Optional[str] = None,
    ncols: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    ylabel: str = "Temperature (°C)",
) -> plt.Figure:
    """
    Plot probabilistic forecasts for multiple stations in a grid.
    
    Args:
        ensemble_predictions: Array of shape (n_ensemble_members, n_lead_times, n_stations)
                             or (n_lead_times, n_stations, n_ensemble_members)
        station_names: List of station names
        ground_truth: Optional array of shape (n_lead_times, n_stations)
        lead_times: Optional array of lead time indices
        reference_time: Reference time string
        ncols: Number of columns in subplot grid
        figsize: Figure size. If None, auto-computed based on number of stations
        save_path: If provided, saves figure to this path
        ylabel: Label for y-axis
        
    Returns:
        matplotlib Figure object
    """
    n_stations = len(station_names)
    nrows = int(np.ceil(n_stations / ncols))
    
    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, station in enumerate(station_names):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        # Extract data for this station
        if ensemble_predictions.ndim == 3:
            # Determine shape and extract appropriately
            if ensemble_predictions.shape[0] < ensemble_predictions.shape[1]:
                # (n_ensemble, n_lead, n_station)
                station_ensembles = ensemble_predictions[:, :, idx].T
            else:
                # (n_lead, n_station, n_ensemble)
                station_ensembles = ensemble_predictions[:, idx, :]
        else:
            raise ValueError(f"Expected 3D array, got shape {ensemble_predictions.shape}")
        
        station_gt = ground_truth[:, idx] if ground_truth is not None else None
        
        # Plot on this axis
        plot_single_station_on_axis(
            ax=ax,
            ensemble_predictions=station_ensembles,
            ground_truth=station_gt,
            lead_times=lead_times,
            station_name=station,
            ylabel=ylabel if col == 0 else "",
            show_legend=(idx == 0),  # Only show legend on first plot
        )
    
    # Hide unused subplots
    for idx in range(n_stations, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')
    
    if reference_time:
        fig.suptitle(f"Probabilistic Forecasts ({reference_time})", 
                    fontsize=16, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


def plot_single_station_on_axis(
    ax: plt.Axes,
    ensemble_predictions: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    lead_times: Optional[np.ndarray] = None,
    station_name: str = "Station",
    ylabel: str = "Temperature (°C)",
    show_legend: bool = True,
):
    """Helper function to plot a single station on a given axis."""
    # Ensure correct shape
    if ensemble_predictions.shape[0] < ensemble_predictions.shape[1]:
        ensembles = ensemble_predictions.T
    else:
        ensembles = ensemble_predictions
    
    n_lead_times, n_ensembles = ensembles.shape
    
    if lead_times is None:
        lead_times = np.arange(n_lead_times)
    
    # Compute statistics
    mean = np.mean(ensembles, axis=1)
    lower = np.quantile(ensembles, 0.025, axis=1)
    upper = np.quantile(ensembles, 0.975, axis=1)
    
    # Plot - limit number of members for visibility
    colors = cm.Blues(np.linspace(0.4, 0.9, min(n_ensembles, 20)))
    for i in range(min(n_ensembles, 20)):
        label = 'Ensemble Members' if i == 0 and show_legend else None
        color_idx = min(i, len(colors) - 1)
        ax.plot(lead_times, ensembles[:, i], color=colors[color_idx], 
               linewidth=0.8, alpha=0.3, label=label)
    
    ax.fill_between(lead_times, lower, upper, color='grey', alpha=0.2, 
                    label='95% Interval' if show_legend else None)
    ax.plot(lead_times, mean, color='red', linewidth=2, 
           label='Mean' if show_legend else None)
    
    if ground_truth is not None:
        if isinstance(ground_truth, torch.Tensor):
            ground_truth = ground_truth.cpu().numpy().squeeze()
        # Ensure ground truth is the right length
        gt_len = min(len(ground_truth), len(lead_times))
        ax.plot(lead_times[:gt_len], ground_truth[:gt_len], color='darkgreen', linewidth=2,
               marker='o', markersize=3, label='Truth' if show_legend else None)
    
    ax.set_title(station_name, fontsize=11, fontweight='bold')
    ax.set_xlabel("Lead Time (h)", fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(fontsize=8, loc='best')


def compare_model_forecasts(
    forecasts_dict: dict,
    ground_truth: np.ndarray,
    input_history: Optional[np.ndarray] = None,
    lead_times: Optional[np.ndarray] = None,
    station_name: str = "Station",
    reference_time: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 10),
    save_path: Optional[str] = None,
    ylabel: str = "Temperature (°C)",
) -> plt.Figure:
    """
    Compare probabilistic forecasts from multiple models.
    
    Args:
        forecasts_dict: Dictionary mapping model names to ensemble predictions
                       Each value should be array of shape (n_lead_times, n_ensemble_members)
        ground_truth: Array of shape (n_lead_times,) with actual observations
        input_history: Optional historical observations
        lead_times: Optional lead time indices
        station_name: Name of the station
        reference_time: Reference time string
        figsize: Figure size
        save_path: If provided, saves figure to this path
        ylabel: Label for y-axis
        
    Returns:
        matplotlib Figure object
    """
    n_models = len(forecasts_dict)
    fig, axes = plt.subplots(n_models, 1, figsize=figsize)
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, predictions) in enumerate(forecasts_dict.items()):
        ax = axes[idx]
        
        # Ensure correct shape
        if predictions.shape[0] < predictions.shape[1]:
            ensembles = predictions.T
        else:
            ensembles = predictions
        
        n_lead_times, n_ensembles = ensembles.shape
        
        if lead_times is None:
            lead_times = np.arange(n_lead_times)
        
        # Plot input history
        history_offset = 0
        if input_history is not None and idx == 0:
            n_history = len(input_history)
            history_times = np.arange(-n_history, 0)
            ax.plot(history_times, input_history, color='gray', linewidth=2,
                   label='Historical', alpha=0.7)
            ax.axvline(x=0, color='red', linestyle=':', linewidth=2, alpha=0.5)
        
        # Compute statistics
        mean = np.mean(ensembles, axis=1)
        lower = np.quantile(ensembles, 0.025, axis=1)
        upper = np.quantile(ensembles, 0.975, axis=1)
        
        # Plot ensemble members
        colors = cm.viridis(np.linspace(0.2, 0.8, n_ensembles))
        for i in range(min(n_ensembles, 20)):  # Limit to 20 members for visibility
            label = 'Ensemble' if i == 0 else None
            ax.plot(lead_times, ensembles[:, i], color=colors[i] if i < len(colors) else colors[-1],
                   linewidth=0.8, alpha=0.3, label=label)
        
        # Plot statistics
        ax.fill_between(lead_times, lower, upper, color='grey', alpha=0.2, label='95% Interval')
        ax.plot(lead_times, mean, color='red', linewidth=2.5, label='Mean')
        
        # Plot ground truth
        gt = ground_truth
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy().squeeze()
        # Ensure ground truth is the right length
        gt_len = min(len(gt), len(lead_times))
        ax.plot(lead_times[:gt_len], gt[:gt_len], color='darkgreen', linewidth=2.5,
               marker='o', markersize=4, label='Ground Truth', zorder=10)
        
        # Formatting
        ax.set_title(f"{model_name} - {station_name}", fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        if idx == n_models - 1:
            ax.set_xlabel("Lead Time (hours)", fontsize=10)
    
    if reference_time:
        fig.suptitle(f"Model Comparison ({reference_time})", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


# Example usage function
def example_usage():
    """Example of how to use these plotting functions."""
    # Generate synthetic data
    n_lead_times = 24
    n_ensemble_members = 11
    
    # Create synthetic ensemble predictions
    mean_temp = 15 + 5 * np.sin(np.linspace(0, 2*np.pi, n_lead_times))
    ensemble_predictions = mean_temp[:, np.newaxis] + np.random.randn(n_lead_times, n_ensemble_members) * 2
    
    # Ground truth
    ground_truth = mean_temp + np.random.randn(n_lead_times) * 0.5
    
    # Input history
    input_history = 15 + 5 * np.sin(np.linspace(-2*np.pi, 0, 72))
    
    # Plot
    fig = plot_probabilistic_forecast(
        ensemble_predictions=ensemble_predictions,
        ground_truth=ground_truth,
        input_history=input_history,
        station_name="Example Station",
        reference_time="2024-11-28 12:00",
        show_control=True,
        save_path="example_forecast.png"
    )
    plt.show()
    
    print("Example plot created!")


if __name__ == "__main__":
    example_usage()

