#pylint: disable=E0401
from typing import Union
import os
import numpy as np
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler
import wandb
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from src.model import ClassEmbedder
from src.inference import ImageGenerationPipeline, CounterfactualPipeline
from src.utils import create_grid, create_counterfactual_grid

class Trainer:
    def __init__(self, scheduler, num_epochs, batch_size, n_counterfactual, mixed_precision,
                 num_workers, p_label_dropout, num_inference_steps, guidance_scale,
                 percent_steps, use_dn, num_generate, save_path, seed):
        self.scheduler = scheduler
        # self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.n_counterfactual = n_counterfactual
        self.mixed_precision = mixed_precision
        self.num_workers = num_workers
        self.p_label_dropout = p_label_dropout
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.percent_steps = percent_steps
        self.use_dn = use_dn
        self.num_generate = num_generate
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.seed = seed

        self.vae = None
        self.class_embedder = None
        self.unet = None
        self.optimizer = None
        self.transformations = None
        self.augmentations = None
        
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)

        if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
            self.scaler = None
        elif mixed_precision == "fp16":
            self.dtype = torch.float16
            self.scaler = torch.amp.GradScaler(self.device_str)
        else:
            self.dtype = torch.float32
            self.scaler = None

    def train(self, vae, class_embedder, unet, train_ds, val_ds,
              optimizer, transformations, augmentations,
              train_sampler, val_sampler):
        # TODO: maybe move transformations and augmentations to init?
        # TODO may not want to add fields to class outside of init
        # Get device and move all models on to it
        self.vae = vae.to(self.device)
        self.class_embedder = class_embedder.to(self.device)
        self.unet = unet.to(self.device)
        self.optimizer = optimizer
        self.transformations = transformations.to(self.device)
        self.augmentations = augmentations.to(self.device)
        # Freeze paramaters of models not being trained
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.unet.train()
        self.class_embedder.train()
        self.vae.to(dtype=self.dtype)
        # train and validation data loaders
        # shuffle is set to false as sampler handles that
        train = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False,
                       num_workers=self.num_workers, pin_memory=True, sampler = train_sampler)
        # Moved to val code
        val = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False,
                         pin_memory=True, num_workers=self.num_workers, sampler=val_sampler)
        global_step = 0
        for epoch in range(1, self.num_epochs + 1):
            print(f"Starting epoch: {epoch}")
            for images, labels in tqdm(train):
                loss = self._train_step(images, labels)
                wandb.log({
                    "loss": loss,
                    "epoch": epoch,
                    "global_step": global_step
                })
                global_step += 1
            self._validation_step(val, epoch)
            # if epoch % 5 == 0 or epoch == self.num_epochs - 1:
            #     print("Running validation code")
            #     self._validation_step(val_ds, epoch)

    def _train_step(self, images, labels, training=True):
        images = images.to(self.device, non_blocking = True)
        labels = labels.to(self.device, non_blocking = True)
        if training:
            # images = self.augmentations(self.transformations(images))
            images = self.transformations(self.augmentations(images))
        else:
            images = self.transformations(images)
        images = images.to(self.dtype)
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor
        # TODO Verify correct dtype
        # latents = latents.to(torch.float32)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps,
                                  (images.size(0),), device = self.device)
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        if self.p_label_dropout > 0:
            drop_mask = torch.rand(labels.shape, device = self.device) < self.p_label_dropout
            labels = labels.masked_fill(drop_mask, self.class_embedder.null_class_label)
        self.optimizer.zero_grad()
        with torch.amp.autocast(self.device_str, dtype = self.dtype,
                                enabled = (self.mixed_precision in ["bf16", "fp16"])):
            class_embeddings = self.class_embedder(labels)
            noise_pred = self.unet(noisy_latents, timesteps,
                                   encoder_hidden_states = class_embeddings).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
        if training and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif training:
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def generate_images(self, scheduler):
        generation_pipeline = ImageGenerationPipeline(
                vae=self.vae,
                class_embedder=self.class_embedder,
                unet=self.unet,
                scheduler=scheduler
            )
        classes = [0, 1, self.class_embedder.null_class_label]
        labels = torch.tensor(
                classes * self.num_generate,
                device=self.device
            )
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        new_images = generation_pipeline(
            labels,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            output_type="numpy").images
        new_images = np.clip((new_images + 1) / 2, 0, 1)
        label_name_map = ["NRG", "RG", "Null"]
        fig = create_grid(new_images, [label_name_map[c] for c in classes])
        return fig
    
    def generate_counterfactual(self, images, labels, scheduler):
        counterfactual_pipeline = CounterfactualPipeline(
                vae=self.vae,
                class_embedder=self.class_embedder,
                unet=self.unet,
                scheduler=scheduler
            )
        output = counterfactual_pipeline(
            images = images,
            num_inference_steps = self.num_inference_steps,
            guidance_scale = self.guidance_scale, 
            percent_steps = self.percent_steps,
            use_dn = self.use_dn,
            output_type="numpy")
        def map_for_plotting(np_array):
            # maps data to [0,1) interval
            return np.clip((np_array + 1) / 2, 0, 1)
        cf_images = map_for_plotting(output.images)
        heatmap = map_for_plotting(output.heatmaps)
        images_np = images.permute((0, 2, 3, 1)).cpu().numpy()
        images_np = map_for_plotting(images_np)
        fig = create_counterfactual_grid(images_np, cf_images, heatmap, labels)
        return fig
    
    def _get_counterfactual_images(self, val):
        n_neg, n_pos = 0, 0
        n = self.n_counterfactual
        # Reset the seed so that order is the same between epochs
        val.sampler.generator.manual_seed(self.seed)
        val_iter = iter(val)
        neg_list, pos_list = [], []
        # Assume val dataloader was passed a sampler with a generator
        # This ensures the order is always the same
        # Code shouln't break if this is not the case; images may differ between epochs
        while n_neg < n or n_pos < n:
            batch = next(val_iter)
            for img, label in zip(*batch):
                if label == 1 and n_pos < n:
                    pos_list.append(img)
                    n_pos += 1
                elif label == 0 and n_neg < n:
                    neg_list.append(img)
                    n_neg += 1
                if n_neg >= n and n_pos >= n:
                    break
        img = torch.stack(neg_list + pos_list)
        labels = torch.cat([torch.zeros((n_neg,)), torch.ones((n_pos,))])
        return img, labels
    
    def _validation_step(self, val, epoch):
        # Calculate val_loss
        val_loss = 0
        print("Calculating validation loss")
        with torch.no_grad():
            for images, labels in tqdm(val):
                val_loss += self._train_step(images, labels, training = False)
        wandb.log({"val_loss": val_loss, "epoch": epoch})
        torch.cuda.empty_cache()
        if epoch == 1 or epoch % 5 == 0 or epoch == self.num_epochs:
            val_scheduler = DDIMScheduler.from_config(self.scheduler.config)
            # Data for counterfactual. Balanced subset of val
            print("Getting validation images for counterfactual estimation")
            images, labels = self._get_counterfactual_images(val)
            images = self.transformations(images)
            with torch.amp.autocast(self.device_str, dtype = self.dtype,
                                    enabled = (self.mixed_precision in ["bf16", "fp16"])):
                print("Generating new images")
                gen_fig = self.generate_images(val_scheduler)
                print("Performing Counterfactual")
                cf_fig = self.generate_counterfactual(images, labels, val_scheduler)
            wandb.log({
                "Generated Images": wandb.Image(gen_fig),
                "Counterfactual Images": wandb.Image(cf_fig),
                "epoch": epoch
                })
            plt.close("all")
            print("Saving model")
            self._save_model(epoch)
        print("Finished Validating")

    def _save_model(self, epoch):
        save_path = os.path.join(self.save_path, wandb.run.id, f"{epoch:03d}")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        self.unet.save_pretrained(save_path)
        torch.save(self.class_embedder.state_dict(),
                   os.path.join(save_path, f"class_embedder.pt"))
