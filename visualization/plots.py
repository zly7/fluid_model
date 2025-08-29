"""
Plotting utilities for fluid dynamics data visualization.

Provides functions for creating charts and plots for data analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

# Set style
plt.style.use('default')
sns.set_palette("husl")

def plot_time_series(data: pd.DataFrame, 
                    columns: List[str], 
                    title: str = "Time Series Plot",
                    figsize: Tuple[int, int] = (12, 6),
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot time series data for specified columns.
    
    Args:
        data: DataFrame with time series data
        columns: List of column names to plot
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in columns:
        if col in data.columns:
            ax.plot(data.index, data[col], label=col, alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_equipment_comparison(data: Dict[str, np.ndarray],
                            equipment_types: List[str],
                            title: str = "Equipment Comparison",
                            figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
    """
    Plot comparison of different equipment types.
    
    Args:
        data: Dictionary mapping equipment type to data arrays
        equipment_types: List of equipment types to plot
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_types = len(equipment_types)
    n_cols = min(3, n_types)
    n_rows = (n_types + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_types == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_types > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, eq_type in enumerate(equipment_types):
        if eq_type in data:
            eq_data = data[eq_type]
            
            # Plot first few samples
            for j in range(min(5, eq_data.shape[0])):
                axes[i].plot(eq_data[j], alpha=0.6, label=f'Sample {j+1}')
            
            axes[i].set_title(f'{eq_type} Equipment')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
    
    # Hide unused subplots
    for i in range(n_types, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

def plot_correlation_matrix(data: pd.DataFrame,
                           title: str = "Correlation Matrix",
                           figsize: Tuple[int, int] = (10, 8),
                           method: str = 'pearson') -> plt.Figure:
    """
    Plot correlation matrix heatmap.
    
    Args:
        data: DataFrame with numerical data
        title: Plot title
        figsize: Figure size
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Matplotlib figure
    """
    # Calculate correlation matrix
    corr_matrix = data.corr(method=method)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True,
                cmap='RdYlBu_r',
                center=0,
                square=True,
                fmt='.2f',
                ax=ax)
    
    ax.set_title(title)
    plt.tight_layout()
    
    return fig

def plot_distribution(data: np.ndarray,
                     title: str = "Data Distribution",
                     bins: int = 50,
                     figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot data distribution histogram with statistics.
    
    Args:
        data: Data array
        title: Plot title
        bins: Number of histogram bins
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Flatten data if multidimensional
    if data.ndim > 1:
        data_flat = data.flatten()
    else:
        data_flat = data
    
    # Histogram
    ax1.hist(data_flat, bins=bins, alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(data_flat), color='red', linestyle='--', label=f'Mean: {np.mean(data_flat):.3f}')
    ax1.axvline(np.median(data_flat), color='green', linestyle='--', label=f'Median: {np.median(data_flat):.3f}')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(data_flat, vert=True)
    ax2.set_ylabel('Value')
    ax2.set_title('Box Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

def plot_prediction_comparison(predictions: np.ndarray,
                             targets: np.ndarray,
                             sample_indices: Optional[List[int]] = None,
                             title: str = "Prediction vs Target",
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot prediction vs target comparison.
    
    Args:
        predictions: Prediction arrays (samples, timesteps, features)
        targets: Target arrays (samples, timesteps, features)
        sample_indices: Specific samples to plot
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if sample_indices is None:
        sample_indices = list(range(min(4, predictions.shape[0])))
    
    n_samples = len(sample_indices)
    n_cols = 2
    n_rows = (n_samples + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_samples == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_samples > 1 else [axes]
    else:
        axes = axes.flatten()
    
    for i, sample_idx in enumerate(sample_indices):
        if sample_idx < predictions.shape[0]:
            pred_sample = predictions[sample_idx]
            target_sample = targets[sample_idx]
            
            # Plot first feature or average all features
            if pred_sample.ndim > 1:
                pred_plot = pred_sample.mean(axis=1)
                target_plot = target_sample.mean(axis=1)
            else:
                pred_plot = pred_sample
                target_plot = target_sample
            
            axes[i].plot(pred_plot, label='Prediction', alpha=0.8)
            axes[i].plot(target_plot, label='Target', alpha=0.8)
            axes[i].set_title(f'Sample {sample_idx}')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

def plot_loss_curves(train_losses: List[float],
                    val_losses: List[float],
                    title: str = "Training Loss Curves",
                    figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, label='Training Loss', marker='o', markersize=4)
    ax.plot(epochs, val_losses, label='Validation Loss', marker='s', markersize=4)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Find best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.7)
    ax.text(best_epoch, best_val_loss, f'Best: Epoch {best_epoch}', 
            verticalalignment='bottom', horizontalalignment='center')
    
    plt.tight_layout()
    
    return fig

def plot_attention_weights(attention_weights: np.ndarray,
                          title: str = "Attention Weights",
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot attention weight heatmaps.
    
    Args:
        attention_weights: Attention weight matrix (seq_len, seq_len) or (heads, seq_len, seq_len)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if attention_weights.ndim == 3:
        # Multiple attention heads
        n_heads = attention_weights.shape[0]
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_heads == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_heads > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i in range(n_heads):
            sns.heatmap(attention_weights[i], 
                       cmap='Blues',
                       cbar=True,
                       ax=axes[i])
            axes[i].set_title(f'Head {i+1}')
        
        # Hide unused subplots
        for i in range(n_heads, len(axes)):
            axes[i].set_visible(False)
            
    else:
        # Single attention matrix
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(attention_weights, 
                   cmap='Blues',
                   cbar=True,
                   ax=ax)
        ax.set_title('Attention Weights')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

def save_plots_to_directory(figures: List[plt.Figure],
                          filenames: List[str],
                          output_dir: str,
                          format: str = 'png',
                          dpi: int = 300):
    """
    Save multiple plots to a directory.
    
    Args:
        figures: List of matplotlib figures
        filenames: List of filenames (without extension)
        output_dir: Output directory path
        format: Image format ('png', 'pdf', 'svg', etc.)
        dpi: Image resolution
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for fig, filename in zip(figures, filenames):
        filepath = output_path / f"{filename}.{format}"
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', format=format)
    
    print(f"Saved {len(figures)} plots to {output_dir}")

def create_summary_report_plots(data: Dict[str, Any],
                               output_dir: str = "plots") -> List[plt.Figure]:
    """
    Create a comprehensive set of plots for data analysis report.
    
    Args:
        data: Dictionary containing various data arrays and metadata
        output_dir: Directory to save plots
        
    Returns:
        List of created figures
    """
    figures = []
    filenames = []
    
    # Equipment data distribution plots
    if 'equipment_data' in data:
        for eq_type, eq_data in data['equipment_data'].items():
            fig = plot_distribution(eq_data, title=f"{eq_type} Equipment Distribution")
            figures.append(fig)
            filenames.append(f"distribution_{eq_type}")
    
    # Time series plots
    if 'time_series' in data:
        fig = plot_time_series(data['time_series'], 
                              data['time_series'].columns[:5], 
                              title="Sample Time Series")
        figures.append(fig)
        filenames.append("time_series_sample")
    
    # Correlation matrix
    if 'correlation_data' in data:
        fig = plot_correlation_matrix(data['correlation_data'], title="Feature Correlations")
        figures.append(fig)
        filenames.append("correlation_matrix")
    
    # Prediction comparisons
    if 'predictions' in data and 'targets' in data:
        fig = plot_prediction_comparison(data['predictions'], data['targets'])
        figures.append(fig)
        filenames.append("prediction_comparison")
    
    # Loss curves
    if 'train_losses' in data and 'val_losses' in data:
        fig = plot_loss_curves(data['train_losses'], data['val_losses'])
        figures.append(fig)
        filenames.append("loss_curves")
    
    # Save all plots
    save_plots_to_directory(figures, filenames, output_dir)
    
    return figures