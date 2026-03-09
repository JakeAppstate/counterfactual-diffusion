# pylint: disable=E0401; pyright: reportMissingImports=false
from __future__ import annotations
from abc import ABC, abstractmethod
import os
import pickle
import json
from tqdm import tqdm
from cv2 import medianBlur
import numpy as np
import sklearn
import torch
import torchvision
from torch import nn
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler, DDIMInverseScheduler

class ModelInterface(ABC):
    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def save(self, save_path: str):
        pass

    @abstractmethod
    @classmethod
    def load(cls, save_path: str):
        pass

class ClassEmbedder(nn.Module, ModelInterface):
    def __init__(self, num_classes: int, emb_dim: int):
        super().__init__()
        # Used to save state of model to json
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.load_path = None

        self.null_class_label = num_classes
        self.label_emb = nn.Embedding(num_classes + 1, emb_dim,
                                            padding_idx = self.null_class_label)
        self.class_emb = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, *args):
        assert len(args) == 1
        labels, = args
        x = self.class_emb(self.label_emb(labels))
        return x.unsqueeze(1)

    def save(self, save_path):
        # if model is already saved then create a link to saved model to save storage
        if self.load_path is not None:
            os.symlink(self.load_path, save_path)
            return
        with open(os.path.join(save_path, "model_args.json"), "w", encoding="utf-8") as file:
            data = {"num_classes": self.num_classes, "emb_dim": self.emb_dim}
            json.dump(data, file, indent=4)
        torch.save(self.state_dict(),
                   os.path.join(save_path, "model.pt"))

    @classmethod
    def load(cls, save_path: str) -> ClassEmbedder:
        # can raise file not found error
        with open(os.path.join(save_path), "r", encoding="utf-8") as file:
            args = json.load(file)
        
        model = cls(**args)
        model.load_state_dict(torch.load(os.path.join("model.pt")))
        model.load_path = save_path
        return model

        
class VAE(nn.Module, ModelInterface):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae
        self.load_path = None
        
    def forward(self, *args):
        assert len(args) == 1
        img, = args
        posterior = self.vae.encode(img).latent_dist
        latents = posterior.sample()
        reconstruction = self.vae.decode(latents).sample
        return posterior, reconstruction
    
    @torch.no_grad()
    def encode(self, img: torch.Tensor):
        return self.vae.encode(img).latent_dist.mode() * self.vae.config.scaling_factor
    
    @torch.no_grad()
    def decode(self, latents: torch.Tensor):
        latents =  1 / self.vae.config.scaling_factor * latents
        return self.vae.decode(latents).sample
    
    @property
    def device(self):
        return self.vae.device

    def save(self, save_path: str):
        # if model is already saved then create a link to saved model to save storage
        if self.load_path is not None:
            os.symlink(self.load_path, save_path)
            return
        self.vae.save_pretrained(save_path)

    @classmethod
    def load(cls, save_path: str) -> VAE:
        # TODO currently wrapper only accepts AutoencoderKL
        # Would maybe like to be able to use other autoencoders in the future
        model = cls(AutoencoderKL.from_pretrained(save_path))
        model.load_path = save_path
        return model

class LatentDiffusionModel(nn.Module, ModelInterface):
    def __init__(self, vae: VAE, class_embedder: ClassEmbedder,
                 unet: UNet2DConditionModel, scheduler: DDIMScheduler):
        super().__init__()
        vae = vae.eval()
        # Freeze parameters of vae
        for param in vae.parameters():
            param.requires_grad = False
        self.vae = vae
        self.class_embedder = class_embedder
        self.unet = unet
        self.scheduler = scheduler
        self.load_path = None

    def forward(self, *args):
        assert len(args) == 3
        latents, timesteps, labels = args
        class_embeddings = self.class_embedder(labels)
        noise_pred = self.unet(latents, timesteps, encoder_hidden_states = class_embeddings).sample
        return noise_pred

    def _dynamic_normalization(self, img, p = 0.99):
        assert img.ndim == 4
        # from paper:
        # th = max(1, percentile(img, p))
        # img = clip(-th, th)
        x = torch.abs(img).flatten(2).float()
        q = torch.quantile(x, p, dim=-1, keepdim=True)
        q = torch.max(q, torch.ones_like(q))
        q = q.unsqueeze(-1).expand(img.shape)
        return torch.clamp(img, -q, q)

    # TODO Might should use forward method instead of calling sub models?
    @torch.no_grad()
    def _sample(self, latents, labels, scheduler,
                num_inference_steps = 100, guidance_scale = 0,
                use_dn = False, start_idx = 0, stop_idx = -1, log = False):
        # TODO Deal with having all models on the save device/dtype
        # and maybe inplement .to()
        # Update: should be handled autmatically by pytorch; need to test
        device = self.unet.device
        stop_idx = num_inference_steps if stop_idx == -1 else stop_idx
        class_embeddings = self.class_embedder(labels)
        if guidance_scale > 1.0:
            null_labels = torch.full_like(labels, self.class_embedder.null_class_label)
            null_embeddings = self.class_embedder(null_labels)
            class_embeddings = torch.cat([null_embeddings, class_embeddings])
        
        scheduler.set_timesteps(num_inference_steps, device=device)
        with torch.no_grad():
            for t in tqdm(scheduler.timesteps[start_idx:stop_idx], disable = not log):
                latent_model_input = latents if guidance_scale <= 1.0 else torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.unet(latent_model_input, t, class_embeddings).sample
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                if use_dn:
                    noise_pred = self._dynamic_normalization(noise_pred)

                latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        return latents
    
    def _rescale(self, img):
        return torch.clamp((img + 1) / 2, 0, 1)
    
    # TODO Have option to disable logging
    def generate(self, labels, num_inference_steps, guidance_scale, rescale = False, generator = None, log = True):
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
        latents = self._sample(latents, labels, self.scheduler, num_inference_steps, guidance_scale, use_dn = False, log = log)
        # Decode latents to images
        images = self.vae.decode(latents)
        images = images.cpu().float()
        if rescale:
            images = self._rescale(images)
        return images

    # TODO Have option to disable logging
    @torch.no_grad
    def get_counterfactual(self, images, num_inference_steps, guidance_scale = 3.0,
                           percent_steps = 1.0, use_dn = True, rescale = False,
                           log = False):
        # Get forward and reverse scheduler
        reverse_scheduler = DDIMInverseScheduler.from_config(self.scheduler.config)
        forward_scheduler = self.scheduler
        
        # TODO verify assumption that images are already on the same device as VAE
        device = self.unet.device
        batch_size = images.size(0)

        idx = int(np.ceil(percent_steps * num_inference_steps))

        images = images.to(self.vae.device)
        latents = self.vae.encode(images)
        images = images.cpu() # remove images from gpu to save vram
        latents = latents.to(device, dtype = self.unet.dtype)
        # latents = latents.to(self.unet.dtype)
        # Backwards Process: x_0 -> x_T
        null_labels = torch.full((batch_size,), self.class_embedder.null_class_label,
                                 device = device, dtype = torch.long)
        healthy_labels = torch.ones((batch_size,), device = device, dtype = torch.long)
        # Backwards process: encoding img into spacial latent space
        if log:
            print("Perfoming the Backwards Process...")
        latents = self._sample(latents, null_labels, reverse_scheduler, num_inference_steps,
                               use_dn=use_dn, stop_idx=idx, log = log)
        if log:
            print("Performing the Forwards Process...")
        # Forward Process: decoding img back into pixel space
        latents = self._sample(latents, healthy_labels, forward_scheduler, num_inference_steps,
                               guidance_scale=guidance_scale, use_dn=use_dn, start_idx=-idx, log = log)
        # Decode latents to images
        new_images = self.vae.decode(latents).float().cpu()
        heatmap = torch.mean(torch.abs(new_images - images), dim=1, keepdim=True)
        # new_images = torch.clamp((new_images + 1) / 2, 0, 1).cpu() # scale to [0, 1]
        # convert to output format
        heatmap = heatmap.to(torch.float32)
        if rescale:
            images = self._rescale(images)
            new_images = self._rescale(new_images)
            heatmap = self._rescale(heatmap)
        return images, new_images, heatmap

    def save(self, save_path):
        # if model is already saved then create a link to saved model to save storage
        if self.load_path is not None:
            os.symlink(self.load_path, save_path)
            return
        self.unet.save_pretrained(save_path)
        self.class_embedder.save(save_path)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        return self

    @classmethod
    def load(cls, save_path):
        # load class embedder
        class_embedder = ClassEmbedder.load(os.path.join(save_path, "class_embedder"))
        # load unet
        unet = UNet2DConditionModel.from_pretrained(os.path.join(save_path, "unet"))
        # load vae
        vae = VAE.load(os.path.join(save_path, "vae"))
        # load scheduler
        scheduler = DDIMScheduler.from_pretrained(os.path.join(save_path, "scheduler"))
        model = cls(vae, class_embedder, unet, scheduler)
        model.load_path = save_path
        return model

# class CounterfactualMLClassifier(ModelInterface):
#     def __init__(self, model: sklearn.base.BaseEstimator):
#         self.model = model

#     def __call__(self, x):
#         return self.forward(x)

#     def _convert_to_numpy(self, hm):
#         is_torch = isinstance(hm, torch.Tensor)
#         is_numpy = isinstance(hm, np.ndarray)
#         assert is_torch or is_numpy

#         # convert to numpy
#         if is_torch:
#             hm = hm.cpu().numpy()
#         # have 4 dimensions with channel dimension being last
#         # (batch, width, height, channel)
#         if hm.ndim == 3:
#             hm = hm.expand_dims(0)
#         elif hm.ndim == 2:
#             hm = hm.expand_dims((0, -1))

#         assert hm.ndim == 4 and hm.shape[-1] == 1
#         hm = hm.permute((0, 2, 3, 1))
#         return hm
    
#     def _extract_features(self, hm):
#         mean = np.mean(hm, axis = (1, 2, 3))
#         maximum = np.max(hm, axis = (1, 2, 3))
#         var = np.var(hm, axis = (1, 2, 3))

#         # may want to normalize; could be part of pipeline
#         return np.stack([mean, maximum, var], axis = 1)
    
#     def get_features(self, hm):
#         hm = self._convert_to_numpy(hm)

#         # Apply median filter
#         # convert to uint8
#         x_min, x_max = np.min(hm), hm.max(hm)
#         x_range = x_max - x_min
#         scale = 255.0 / x_range
#         hm = (hm * scale).astype(np.unt8)
#         hm = medianBlur(hm, 5)

#         return self._extract_features(hm)
    
#     def forward(self, *args):
#         x, = args
#         assert isinstance(x, torch.Tensor) or isinstance(x, np.ndarray)
#         if isinstance(x, torch.Tensor) or x.ndim > 2:
#             # x is a heatmap; need to extract features
#             x = self.get_features(x)
#         return self.model.predict_proba(x)
    
#     def fit(self, X, y):
#         self.model.fit(X, y)

#     def save(self, save_path):
#         filepath = os.path.join(save_path, "cf_model.pkl")
#         with open(filepath, "wb") as file:
#             pickle.dump(self.model, file)

# TODO Pass model as parameter and use hydra to create model
class CounterfactualTorchClassifier(nn.Module, ModelInterface):
    def __init__(self):
        super().__init__()
        self.color = torchvision.transforms.v2.Grayscale(num_output_channels=3)
        self.transform = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2.transforms()
        self.model = torchvision.models.mobilenet_v2(weights="DEFAULT")
        self.load_path = None

        for param in self.model.parameters():
            param.requires_grad = False
        # Change last layer to have 2 classes
        self.model.classifier[-1] = nn.Linear(in_features=self.model.classifier[-1].in_features,
                                              out_features=1)

    def forward(self, *args):
        x, = args
        x = self.color(x)
        x = self.transform(x)
        return self.model(x).squeeze()

    def save(self, save_path):
        torch.save(self.state_dict(),
                   os.path.join(save_path, "cf_classifier.pt"))

    @classmethod
    def load(cls, save_path):
        model = cls()
        model.load_path = save_path
        model.load_state_dict(torch.load(os.path.join(save_path, "cf_classifier.pt")))
        return model
