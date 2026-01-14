# pylint: disable=import-error
import matplotlib.pyplot as plt
import numpy as np
from typing import List

def create_grid(images, col_names=List[str]):
    """Create a grid of images for visualization."""
    n_images = len(images)
    n_cols = len(col_names)
    n_rows = np.ceil(n_images / n_cols).astype(int)
    fig, axes = plt.subplots(n_rows, n_cols)
    axes = axes.ravel()
    for i in range(n_images):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        if i < n_cols:
            axes[i].set_title(col_names[i])
    fig.tight_layout()
    return fig

def create_counterfactual_grid(images, heatmaps, labels):
    assert len(images) == len(heatmaps) == len(labels), \
        "images, heatmaps, and labels should have the same length"
    fig, axes = plt.subplot(n_rows=len(images), n_cols = 2)
    for i in range(0, 2 * len(images), 2):
        axes[i][0].imshow(images)
        axes[i][0].ylabel("RG" if labels[i] else "NRG")
        axes[i][1].imshow(heatmaps[i])
    fig.tight_layout()
    return fig
