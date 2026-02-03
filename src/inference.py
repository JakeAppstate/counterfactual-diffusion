#pylint: disable=import-error
from dataclasses import dataclass
from typing import Union, List, Tuple
import numpy as np
from PIL.Image import Image
import torch
from tqdm import tqdm
from diffusers import DiffusionPipeline, ImagePipelineOutput, DDIMScheduler, DDIMInverseScheduler
from diffusers.utils import BaseOutput

# TODO Move sampling code to its own function and call that to clean up code
def _dynamic_normalization(img, p = 0.99):
    assert img.ndim == 4
    # from paper:
    # th = max(1, percentile(img, p))
    # img = clip(-th, th)
    x = torch.abs(img).flatten(2).float()
    q = torch.quantile(x, p, dim=-1, keepdim=True)
    q = torch.max(q, torch.ones_like(q))
    q = q.unsqueeze(-1).expand(img.shape)
    return torch.clamp(img, -q, q)

class _BasePipeline(DiffusionPipeline):
    def __init__(self, vae, class_embedder, unet, scheduler):
        super().__init__()
        self.register_modules(
            vae=vae,
            class_embedder=class_embedder,
            unet=unet,
            scheduler=scheduler
        )

    def __call__(self, **kwargs):
        raise NotImplementedError("This is an abstract base class. Use subclass instead")
    
    def _sample(self, latents, labels, n_steps, guidance_scale = 0, p_steps = 1.0, use_dn = True):
        print(guidance_scale)
        device = self.unet.device
        stop_idx = int(n_steps * p_steps)
        class_embeddings = self.class_embedder(labels)
        if guidance_scale > 1.0:
            null_labels = torch.full_like(labels, self.class_embedder.null_class_label)
            null_embeddings = self.class_embedder(null_labels)
            class_embeddings = torch.cat([null_embeddings, class_embeddings])
        
        self.scheduler.set_timesteps(n_steps, device=device)
        # TODO: add tqdm
        with torch.no_grad():
            for t in tqdm(self.scheduler.timesteps[:stop_idx]):
                latent_model_input = latents if guidance_scale <= 1.0 else torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.unet(latent_model_input, t, class_embeddings).sample
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                if use_dn:
                    noise_pred = _dynamic_normalization(noise_pred)

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        return latents

class ImageGenerationPipeline(_BasePipeline):
    """A Pipeline for generating images from class label
    
    This pipeline is for generating new images, not for performing the counterfactual.
    Args:
        vae (AutoencoderKL): Variational Auto-Encoder to encode and decode images to and from latent representations.
        class_embedder (ClassEmbedder): Model to convert class labels into embeddings.
        unet (UNet2DConditionModel): U-Net model to denoise the latent representations.
        scheduler (DDPMScheduler or DDIMScheduler): Scheduler to manage the denoising process.
    """

    @torch.no_grad()
    def __call__(self, labels: torch.Tensor, num_inference_steps=50, guidance_scale=3.0, generator=None, output_type="pil"):
        """Generate images from class labels.
        
        Args:
            labels (torch.Tensor[torch.long]): Tensor of class labels to condition the image generation.
            num_inference_steps (int): Number of denoising steps. More steps usually lead to better quality.
            guidance_scale (float): Scale for classifier-free guidance. Higher values lead to stronger conditioning.
            generator (torch.Generator, optional): A torch generator for reproducible results.
            output_type (str): The output format of the generated images. Choose between "pil", "numpy", and "torch".
        """
        assert output_type in ["torch", "pil", "numpy"], "output_type must be 'torch', 'pil', or 'numpy'"
        batch_size = labels.shape[0]
        # self.device should exist but self.device throws an eror
        # not sure why this is the case
        device = self.unet.device
        labels = labels.to(device)
        # Starting noises
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, self.unet.sample_size, self.unet.sample_size),
            generator=generator,
            device=device,
            dtype = self.unet.dtype
        )
        latents = latents * self.scheduler.init_noise_sigma

        # Gernerate latents
        # TODO may want to change use_dn to false for generating images
        # Should run test to compare
        print("Generating Images...")
        latents = self._sample(latents, labels, num_inference_steps, guidance_scale, use_dn = False)
        # Decode latents to images
        latents = 1 / self.vae.config.scaling_factor * latents
        # latents = latents.to(self.vae.dtype) # If using mixed precision
        images = self.vae.decode(latents).sample
        # images = torch.clamp((images + 1) / 2, 0, 1).cpu() # scale to [0, 1]
        images = images.to(torch.float32).cpu()
        # convert to output format
        if output_type == "pil":
            images = torch.clip((images + 1) / 2, 0, 1)
            images = self.numpy_to_pil(images.permute(0, 2, 3, 1).numpy())
        elif output_type == "numpy":
            images = images.permute(0, 2, 3, 1).numpy()

        return ImagePipelineOutput(images=images)

@dataclass
class CounterfactualOutput(BaseOutput):
    images: Union[List[Image], np.ndarray]
    heatmaps: Union[List[Image], np.ndarray]

class CounterfactualPipeline(_BasePipeline):
    def __init__(self, vae, class_embedder, unet, scheduler: DDIMScheduler):
        super().__init__(vae, class_embedder, unet, scheduler)
        reverse_scheduler = DDIMInverseScheduler.from_config(scheduler.config)
        forward_scheduler = DDIMScheduler.from_config(scheduler.config)
        self.register_modules(
            reverse_scheduler=reverse_scheduler,
            forward_scheduler=forward_scheduler
        )
    
    @torch.no_grad()
    def __call__(self, images: torch.Tensor, num_inference_steps=50,
                 guidance_scale=3.0, percent_steps=1.0, use_dn = True, output_type="pil"):
        """Generate counterfactual images given source images and target class labels.
        
        Args:
            images (torch.Tensor): Batch of input images to be transformed.
            num_inference_steps (int): Number of denoising steps. 
                More steps usually lead to better quality.
            guidance_scale (float): Scale for classifier-free guidance. 
                Higher values lead to stronger conditioning.
            generator (torch.Generator, optional): A torch generator for reproducible results. 
            output_type (str): The output format of the generated images. Choose between 
                "pil", "numpy", and "torch".
        """
        assert output_type in ["torch", "pil", "numpy"], "output_type must be 'torch', 'pil', or 'numpy'"
        device = self.unet.device
        images = images.to(device, dtype = self.vae.dtype)
        batch_size = images.size(0)

        latents = self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor
        # latents = latents.to(self.unet.dtype)
        # Backwards Process: x_0 -> x_T
        null_labels = torch.full((batch_size,), self.class_embedder.null_class_label,
                                 device = device, dtype = torch.long)
        healthy_labels = torch.ones((batch_size,), device = device, dtype = torch.long)
        # Backwards process: encoding img into spacial latent space
        print("Perfoming the Backwards Process...")
        self.scheduler = self.reverse_scheduler
        latents = self._sample(latents, null_labels, num_inference_steps, p_steps = percent_steps, use_dn=use_dn)
        print("Performing the Forwards Process...")
        # Forward Process: decoding img back into pixel space
        self.scheduler = self.forward_scheduler
        latents = self._sample(latents, healthy_labels, num_inference_steps, guidance_scale, p_steps = percent_steps, use_dn=use_dn)
        # Decode latents to images
        latents = 1 / self.vae.config.scaling_factor * latents
        new_images = self.vae.decode(latents).sample.to(images.dtype).float().cpu()
        heat_map = torch.mean(torch.abs(new_images - images.cpu()), dim=1, keepdim=True)
        # new_images = torch.clamp((new_images + 1) / 2, 0, 1).cpu() # scale to [0, 1]
        # convert to output format
        heat_map = heat_map.to(torch.float32)
        if output_type == "pil":
            new_images = self.numpy_to_pil(new_images.permute(0, 2, 3, 1).numpy())
            heat_map = self.numpy_to_pil(heat_map.permute(0, 2, 3, 1).numpy())
        elif output_type == "numpy":
            new_images = new_images.permute(0, 2, 3, 1).numpy()
            heat_map = heat_map.permute(0, 2, 3, 1).numpy()
        return CounterfactualOutput(images=new_images, heatmaps=heat_map)

