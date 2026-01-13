# pylint: disable=E0401; pyright: reportMissingImports=false
from torch import nn
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DDIMScheduler

# TODO
# Create UNET
# Create Training Pipeline
# Create Inference Pipeline

class ClassEmbedder(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int):
        super().__init__()
        self.null_class_label = num_classes
        self.label_emb = nn.Embedding(num_classes + 1, emb_dim, padding_idx = self.null_class_label)
        self.class_emb = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, class_labels):
        return self.class_emb(self.label_emb(class_labels))
    
