import matplotlib.pyplot as plt
import numpy as np
from typing import List

def create_grid(images, col_names=List[str]):
    """Create a grid of images for visualization."""
    n_images = len(images)
    n_cols = len(col_names)
    n_rows = np.ceil(n_images / n_cols).astype(int)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.ravel()
    for i in range(n_images):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        if i < n_cols:
            axes[i].set_title(col_names[i])
    fig.tight_layout()
    return fig