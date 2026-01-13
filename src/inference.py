#pylint: disable=import-error
import torch
from diffusers import DiffusionPipeline, ImagePipelineOutput, DDIMScheduler, DDIMInverseScheduler

# TODO Move sampling code to its own function and call that to clean up code

class ImageGenerationPipeline(DiffusionPipeline):
    """A Pipeline for generating images from class label
    
    This pipeline is for generating new images, not for performing the counterfactual.
    Args:
        vae (AutoencoderKL): Variational Auto-Encoder to encode and decode images to and from latent representations.
        class_embedder (ClassEmbedder): Model to convert class labels into embeddings.
        unet (UNet2DConditionModel): U-Net model to denoise the latent representations.
        scheduler (DDPMScheduler or DDIMScheduler): Scheduler to manage the denoising process.
    """
    def __init__(self, vae, class_embedder, unet, scheduler):
        """Initialize the ImageGenerationPipeline.
        
        Args:
            vae (AutoencoderKL): Variational Auto-Encoder to encode and decode images to and from latent representations.
            class_embedder (ClassEmbedder): Model to convert class labels into embeddings.
            unet (UNet2DConditionModel): U-Net model to denoise the latent representations.
            scheduler (DDPMScheduler or DDIMScheduler): Scheduler to manage the denoising process."""
        super().__init__()
        self.register_modules(
            vae=vae,
            class_embedder=class_embedder,
            unet=unet,
            scheduler=scheduler
        )

    @torch.no_grad()
    def __call__(self, labels: torch.Tensor[torch.long], num_inference_steps=50, guidance_scale=3.0, generator=None, output_type="pil"):
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
        device = self.device
        labels = labels.to(device)

        # Prepare class embeddings
        class_embeddings = self.class_embedder(labels)
        if guidance_scale > 1.0:
            null_labels = torch.full_like(labels, self.class_embedder.null_class_label,
                                          device=device, dtype=labels.dtype)
            null_class_embeddings = self.class_embedder(null_labels)
            class_embeddings = torch.cat([null_class_embeddings, class_embeddings])

        # Prepare latent noise
        latents = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
            generator=generator,
            device=device
        )

        latents = latents * self.scheduler.init_noise_sigma

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        # Denoising loop
        for t in self.scheduler.timesteps:
            latent_model_input = latents if guidance_scale <= 1.0 else torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise
            noise_pred = self.unet(latent_model_input, t, class_embeddings).sample

            # Guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Implement Dynamic Normalization here if needed
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents to images
        latents = 1 / self.vae.config.scaling_factor * latents
        images = self.vae.decode(latents).sample
        images = torch.clamp((images + 1) / 2, 0, 1).cpu() # scale to [0, 1]
        # convert to output format
        if output_type == "pil":
            images = self.numpy_to_pil(images.permute(0, 2, 3, 1).numpy())
        elif output_type == "numpy":
            images = images.permute(0, 2, 3, 1).numpy()

        return ImagePipelineOutput(images=images)

class CounterfactualPipeline(DiffusionPipeline):
    def __init__(self, vae, class_embedder, unet, scheduler: DDIMScheduler):
        super().__init__()
        reverse_scheduler = DDIMInverseScheduler().from_config(scheduler.config)
        forward_scheduler = scheduler
        self.register_modules(
            vae=vae,
            class_embedder=class_embedder,
            unet=unet,
            reverse_scheduler=reverse_scheduler,
            forward_scheduler=forward_scheduler
        )
    
    @torch.no_grad()
    def __call__(self, images: torch.Tensor, num_inference_steps=50,
                 guidance_scale=3.0, output_type="pil"):
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
        device = self.device
        images = images.to(device)

        latents = self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor
        # Backwards Process: x_0 -> x_T
        null_embeddings = self.class_embedder(
            torch.full((images.size(0),), self.class_embedder.null_class_label,
                       device=device, dtype=torch.long)
        )
        self.reverse_scheduler.set_timesteps(num_inference_steps, device=device)
        for t in self.reverse_scheduler.timesteps:
            latent_model_input = self.reverse_scheduler.scale_model_input(latents, t)
            noise_pred = self.unet(latent_model_input, t, null_embeddings).sample
            latents = self.reverse_scheduler.step(noise_pred, t, latents).prev_sample
        # Forward Process: x_T -> x_0
        healthy_embeddings = self.class_embedder(
            torch.full((images.size(0),), 0, device=device, dtype=torch.long)
        )
        if guidance_scale > 1.0:
            healthy_embeddings = torch.cat([null_embeddings, healthy_embeddings])
        self.forward_scheduler.set_timesteps(num_inference_steps, device=device)
        for t in self.forward_scheduler.timesteps:
            latent_model_input = self.forward_scheduler.scale_model_input(latents, t)
            noise_pred = self.unet(latent_model_input, t, healthy_embeddings).sample
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.forward_scheduler.step(noise_pred, t, latents).prev_sample
        # Decode latents to images
        latents = 1 / self.vae.config.scaling_factor * latents
        new_images = self.vae.decode(latents).sample
        new_images = torch.clamp((new_images + 1) / 2, 0, 1).cpu() # scale to [0, 1]
        heat_map = torch.mean(torch.abs(new_images - images.cpu()), dim=1, keepdim=True)
        # convert to output format
        if output_type == "pil":
            images = self.numpy_to_pil(heat_map.permute(0, 2, 3, 1).numpy())
        elif output_type == "numpy":
            images = heat_map.permute(0, 2, 3, 1).numpy()
        return ImagePipelineOutput(images=heat_map)
