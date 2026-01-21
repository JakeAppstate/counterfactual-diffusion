# pylint: disable=import-error
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import wandb

def _remove_axis(ax):
    # remove axis lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # Remove ticks and tick labels
    ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False
    )

def create_grid(images, col_names=List[str]):
    """Create a grid of images for visualization."""
    n_images = len(images)
    n_cols = len(col_names)
    n_rows = np.ceil(n_images / n_cols).astype(int)
    scale = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * scale, n_rows * scale))
    axes = axes.ravel()
    for i in range(n_images):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        if i < n_cols:
            axes[i].set_title(col_names[i], fontsize = scale * 5, pad = 10)
    fig.tight_layout()
    return fig

def create_counterfactual_grid(images, heatmaps, labels):
    # TODO May be fun to add another column containing the counterfactual image
    assert len(images) == len(heatmaps) == len(labels), \
        "images, heatmaps, and labels should have the same length"
    scale = 4
    fig, axes = plt.subplots(len(images), 2, figsize=(2 * scale, len(images) * scale))
    for i, (img, hmap) in enumerate(zip(images, heatmaps)):
        label_str = "RG" if labels[i] else "NRG"
        axes[i][0].imshow(img)
        axes[i][0].set_ylabel(label_str, fontsize=scale * 4 )
        _remove_axis(axes[i][0])
        axes[i][1].imshow(hmap, cmap="plasma")
        axes[i][1].axis('off')
        if i == 0:
            axes[i][0].set_title("Original Image", fontsize = scale * 5, pad = 10)
            axes[i][1].set_title("Generated Heatmap", fontsize = scale * 5, pad = 10)
    fig.tight_layout()
    return fig

def trace_handler(p, output_dir, file_prefix):
    filename = f"{file_prefix}_step_{p.step_num}.json.gz"
    path = f"{output_dir}/{filename}"
    p.export_chrome_trace(path)
    print(f"Trace saved to {path}")
    wandb.save(path)