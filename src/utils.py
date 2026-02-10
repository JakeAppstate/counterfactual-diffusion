# pylint: disable=import-error
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import wandb

from src.inference import CounterfactualPipeline

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

# TODO add optional row labels
def create_grid(images, col_names: List[str], row_names = None):
    """Create a grid of images for visualization."""
    n_images = len(images)
    n_cols = len(col_names)
    if row_names is not None:
        n_rows = len(row_names)
        assert (n_rows - 1) * n_cols < n_images <= n_rows * n_cols, \
            f"The images should use {n_rows} of the plot"
    else:
        n_rows = np.ceil(n_images / n_cols).astype(int)
    scale = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * scale, n_rows * scale))
    axes = axes.ravel()
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            axes[idx].imshow(images[idx])
            if j == 0 and row_names is not None:
                axes[idx].set_ylabel(row_names[i], fontsize=scale * 4 )
                _remove_axis(axes[idx])
            else:
                axes[idx].axis('off')
            if idx < n_cols:
                axes[idx].set_title(col_names[j], fontsize = scale * 5, pad = 10)
    fig.tight_layout()
    return fig

def create_counterfactual_grid(images, cf_images, heatmaps, labels):
    # TODO May be fun to add another column containing the counterfactual image
    assert len(images) == len(cf_images) == len(heatmaps) == len(labels), \
        "images, heatmaps, and labels should have the same length"
    scale = 4
    fig, axes = plt.subplots(len(images), 3, figsize=(3 * scale, len(images) * scale))
    for i, (img, cf, hmap) in enumerate(zip(images, cf_images, heatmaps)):
        label_str = "RG" if labels[i] else "NRG"
        axes[i][0].imshow(img)
        axes[i][0].set_ylabel(label_str, fontsize=scale * 4 )
        _remove_axis(axes[i][0])
        axes[i][1].imshow(cf)
        axes[i][1].axis("off")
        axes[i][2].imshow(hmap, cmap="plasma")
        axes[i][2].axis('off')
        if i == 0:
            axes[i][0].set_title("Original Image", fontsize = scale * 5, pad = 10)
            axes[i][1].set_title("Counterfactual Image", fontsize = scale * 5, pad = 10)
            axes[i][2].set_title("Heatmap", fontsize = scale * 5, pad = 10)
    fig.tight_layout()
    return fig

def plot_counterfactual_hyperparams(images, labels, pipeline, hyperparams, defaults):
    label_names = ["RG" if label else "NRG" for label in labels]
    figs = {}
    for hp in hyperparams:
        name = hp["name"]
        print("plotting:", name)
        min_val = hp["min"]
        max_val = hp["max"]
        inc = hp["inc"]
        x = min_val
        cf_images = []
        vals = []
        kwargs = defaults.copy()
        while x <= max_val:
            kwargs[name] = x
            output = pipeline(images, output_type="numpy", **kwargs)
            cf_images.append(output.heatmaps)
            vals.append(str(x))
            x += inc
        _, _, w, h = images.shape
        # use stack and reshape to zipper merge the ndarrays
        cf_images = np.stack(cf_images, axis=1).reshape((-1, w, h, 1)) # heatmaps are grayscale
        cf_images = np.clip((cf_images + 1) / 2, 0, 1)
        fig = create_grid(cf_images, vals, row_names = label_names)
        figs[name] = fig
    return figs



def trace_handler(p, output_dir, file_prefix):
    filename = f"{file_prefix}_step_{p.step_num}.json.gz"
    path = f"{output_dir}/{filename}"
    p.export_chrome_trace(path)
    print(f"Trace saved to {path}")
    wandb.save(path)