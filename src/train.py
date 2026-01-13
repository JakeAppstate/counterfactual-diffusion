#pylint: disable=E0401
from typing import Union
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from src.model import ClassEmbedder

def train_model(vae: AutoencoderKL, class_embedder: ClassEmbedder, unet: UNet2DConditionModel,
                scheduler: Union[DDPMScheduler, DDIMScheduler],
                optimizer: torch.optim.Optimizer, train_ds: DataLoader,
                val_ds: DataLoader, transformations: torch.nn.Module,
                augmentations: torch.nn.Module, epochs: int, batch_size: int, mixed_precision: str,
                num_workers: int, p_label_dropout: float):
    # Run on GPU if available
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    vae.to(device)
    class_embedder.to(device)
    unet.to(device)
    transformations.to(device)
    augmentations.to(device)
    vae.requires_grad_(False)
    vae.eval()
    unet.train()
    class_embedder.train()
    # Determine dtype for mixed precision training
    if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        scaler = None
        vae.to(dtype=dtype)
    elif mixed_precision == "fp16":
        dtype = torch.float16
        scaler = torch.amp.GradScaler(device_str)
    else:
        dtype = torch.float32
        scaler = None
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, collate_fn=train_ds.collate_fn,
                       pin_memory=True)
    val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=val_ds.collate_fn)
    # Helper function to unpack dataloader and repack after transformations
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for step, (images, labels) in enumerate(tqdm(train)):
            images = images.to(device, dtype=dtype)
            labels = labels.to(device)
            images = augmentations(transformations(images))
            # Encode images to latents
            with torch.no_grad():
                if dtype == torch.bfloat16:
                    images.to(dtype=torch.bfloat16)
                latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
            latents = latents.to(torch.float32)
            # Get noise for training
            batch_size = latents.size(0)
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            # Label dropout
            if p_label_dropout > 0.0:
                drop_mask = torch.rand(labels.shape, device=device) < p_label_dropout
                labels = labels.masked_fill(drop_mask, class_embedder.null_class_label)
            # Forward pass
            optimizer.zero_grad()
            with torch.amp.autocast(device_str, dtype = dtype, enabled=(mixed_precision in ["fp16", "bf16"])):
                class_embeddings = class_embedder(labels)
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states = class_embeddings).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
            # Backward pass
            if mixed_precision == "fp16" and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        # Validation code
    # Finished training :)
            





    
