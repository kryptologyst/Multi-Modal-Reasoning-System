"""Visualization utilities for multi-modal reasoning."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import seaborn as sns


def plot_similarity_matrix(
    similarities: np.ndarray,
    image_labels: Optional[List[str]] = None,
    text_labels: Optional[List[str]] = None,
    title: str = "Image-Text Similarity Matrix",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot similarity matrix heatmap.
    
    Args:
        similarities: Similarity matrix of shape (n_images, n_texts).
        image_labels: Labels for images.
        text_labels: Labels for text descriptions.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        similarities,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        xticklabels=text_labels,
        yticklabels=image_labels,
        ax=ax,
    )
    
    ax.set_title(title)
    ax.set_xlabel('Text Descriptions')
    ax.set_ylabel('Images')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_attention_weights(
    attention_weights: torch.Tensor,
    image_patches: Optional[List[str]] = None,
    text_tokens: Optional[List[str]] = None,
    title: str = "Cross-Modal Attention Weights",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot cross-modal attention weights.
    
    Args:
        attention_weights: Attention weights tensor.
        image_patches: Labels for image patches.
        text_tokens: Labels for text tokens.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    # Convert to numpy and average over heads
    if attention_weights.dim() > 2:
        attention_weights = attention_weights.mean(dim=0)  # Average over heads
    
    attention_np = attention_weights.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        attention_np,
        annot=True,
        fmt='.3f',
        cmap='Blues',
        xticklabels=text_tokens,
        yticklabels=image_patches,
        ax=ax,
    )
    
    ax.set_title(title)
    ax.set_xlabel('Text Tokens')
    ax.set_ylabel('Image Patches')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_embedding_space(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    image_labels: Optional[List[str]] = None,
    text_labels: Optional[List[str]] = None,
    method: str = 'tsne',
    title: str = "Embedding Space Visualization",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot embedding space visualization.
    
    Args:
        image_embeddings: Image embeddings.
        text_embeddings: Text embeddings.
        image_labels: Labels for images.
        text_labels: Labels for text descriptions.
        method: Dimensionality reduction method ('tsne' or 'pca').
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    # Combine embeddings
    all_embeddings = np.vstack([image_embeddings, text_embeddings])
    
    # Apply dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    embeddings_2d = reducer.fit_transform(all_embeddings)
    
    # Split back into image and text embeddings
    n_images = len(image_embeddings)
    image_2d = embeddings_2d[:n_images]
    text_2d = embeddings_2d[n_images:]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot image embeddings
    scatter1 = ax.scatter(
        image_2d[:, 0], image_2d[:, 1],
        c='red', alpha=0.7, s=100,
        label='Images', marker='o'
    )
    
    # Plot text embeddings
    scatter2 = ax.scatter(
        text_2d[:, 0], text_2d[:, 1],
        c='blue', alpha=0.7, s=100,
        label='Text', marker='s'
    )
    
    # Add labels if provided
    if image_labels:
        for i, label in enumerate(image_labels):
            ax.annotate(label, (image_2d[i, 0], image_2d[i, 1]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    if text_labels:
        for i, label in enumerate(text_labels):
            ax.annotate(label, (text_2d[i, 0], text_2d[i, 1]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_title(title)
    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: List[str],
    title: str = "Metrics Comparison",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot comparison of metrics across different models/configurations.
    
    Args:
        metrics_dict: Dictionary mapping model names to metrics.
        metric_names: List of metric names to plot.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = list(metrics_dict.keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        values = [metrics_dict[model].get(metric, 0) for metric in metric_names]
        ax.bar(x + i * width, values, width, label=model)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_image_grid(
    images: List[Image.Image],
    texts: List[str],
    similarities: List[float],
    n_cols: int = 4,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create a grid of images with text and similarity scores.
    
    Args:
        images: List of PIL Images.
        texts: List of text descriptions.
        similarities: List of similarity scores.
        n_cols: Number of columns in the grid.
        figsize: Figure size.
        save_path: Optional path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (image, text, similarity) in enumerate(zip(images, texts, similarities)):
        row = i // n_cols
        col = i % n_cols
        
        ax = axes[row, col]
        ax.imshow(image)
        ax.set_title(f'{text}\nSimilarity: {similarity:.3f}', fontsize=10)
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[Dict[str, List[float]]] = None,
    val_metrics: Optional[Dict[str, List[float]]] = None,
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training curves.
    
    Args:
        train_losses: Training losses.
        val_losses: Validation losses.
        train_metrics: Training metrics.
        val_metrics: Validation metrics.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the plot.
        
    Returns:
        Matplotlib figure.
    """
    n_plots = 1
    if train_metrics or val_metrics:
        n_plots += len(train_metrics or val_metrics)
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    # Plot losses
    ax = axes[0]
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot metrics
    plot_idx = 1
    if train_metrics:
        for metric_name, values in train_metrics.items():
            if plot_idx < len(axes):
                ax = axes[plot_idx]
                ax.plot(epochs, values, 'b-', label=f'Train {metric_name}')
                
                if val_metrics and metric_name in val_metrics:
                    ax.plot(epochs, val_metrics[metric_name], 'r-', 
                           label=f'Val {metric_name}')
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name} Curves')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plot_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
