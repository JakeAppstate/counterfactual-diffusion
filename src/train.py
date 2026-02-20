#pylint: disable=E0401
from enum import Enum
from typing import List, Dict, Tuple, Union, Iterator, TypeVar
from collections.abc import Callable
from abc import ABC, abstractmethod
import os
from cv2 import medianBlur
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, \
                            f1_score, precision_score, recall_score
from diffusers import SchedulerMixin, DDIMScheduler
import wandb
from torch.utils.data import Dataset, TensorDataset, DataLoader, Sampler
import torch
from tqdm import tqdm

import src.model
from src.inference import ImageGenerationPipeline, CounterfactualPipeline
from src.utils import create_grid, create_counterfactual_grid

# Custom Types
Metrics = Dict[str, Callable[[float, float], float]]
TorchModel = src.model.TorchModelInterface
# loss, pred, label
TrainStepReturn = Tuple[float, torch.Tensor, torch.Tensor]
OptimizerFun = Callable[[Union[List[Tuple[str, Iterator]], Iterator]], torch.optim.Optimizer]

class ParameterGroupNames(Enum):
    UNET = "unet"
    CLASS_EMBEDDER = "class_embedder"
    VAE = "vae"

class TrainerInterface(ABC):
    @abstractmethod
    def train(self, train_ds, val_ds, train_sampler, val_sampler):
        pass

    @abstractmethod
    def evaluate(self, ds, sampler, epoch = None):
        pass

class TorchTrainer(TrainerInterface):
    def __init__(self, model: TorchModel, num_epochs: int, optimizer_fun: OptimizerFun,
                 batch_size: int, num_workers: int, mixed_precision: str,
                 precompute_train: bool, precompute_val: bool, metrics: Metrics,
                 calculate_metrics: bool, seed: int, num_save_epochs: int, save_path: str):
        self.model = model
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_precision = mixed_precision
        self.precompute_train = precompute_train
        self.precompute_val = precompute_val
        self.metrics = metrics
        self.calculate_metrics = calculate_metrics
        self.seed = seed
        self.num_save_epochs = num_save_epochs
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)

        # Get optimizer
        self.optimizer = optimizer_fun(self.model.parameters())

        if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
            self.scaler = None
        elif mixed_precision == "fp16":
            self.dtype = torch.float16
            self.scaler = torch.amp.GradScaler(self.device_str)
        else:
            self.dtype = torch.float32
            self.scaler = None

    # pylint:disable-next=unused-argument
    def _precompute(self, dataloader: DataLoader) -> Dataset:
        return None

    def _load_dataloader(self, ds: Dataset, sampler: Sampler, precompute: bool = False) -> DataLoader:
        data = DataLoader(ds, batch_size = self.batch_size, shuffle = False,
                           num_workers = self.num_workers, pin_memory = True,
                           sampler = sampler)
        if precompute:
            # _precompute might not be implemented and return None
            ds = self._precompute(data) or ds
            data = DataLoader(ds, batch_size = self.batch_size, shuffle = False,
                           num_workers = self.num_workers, pin_memory = True,
                           sampler = sampler)
        return data
    
    def train(self, train_ds: Dataset, val_ds: Dataset, train_sampler: Sampler, val_sampler: Sampler):
        self.model.train()
        self.model.to(self.device)
        train = self._load_dataloader(train_ds, train_sampler, self.precompute_train)
        val = self._load_dataloader(val_ds, val_sampler, self.precompute_val)
        global_step = 0
        for epoch in range(1, self.num_epochs + 1):
            pred_list = []
            target_list = []
            print("Starting epoch:", epoch)
            for images, labels in tqdm(train):
                images = images.to(self.device, non_blocking = True).to(self.dtype)
                labels = labels.to(self.device, non_blocking = True)
                loss, pred, target = self._train_step(images, labels)
                wandb.log({
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": loss
                })
                # For diffusion models prediction and targets are full sized images
                # Storing them would use too much memory (N * 512 * 512 * 3 * 4 bytes)
                if self.calculate_metrics:
                    pred_list.append(pred)
                    target_list.append(target)
                global_step += 1
            if self.calculate_metrics:
                pred = torch.cat(pred_list).numpy()
                target = torch.cat(target_list).numpy()
                metrics = self._calculate_metrics(pred, target)
                metrics["epoch"] = epoch
                wandb.log(metrics)
            print("Performing validation")
            self._evaluate(val, epoch)
            if epoch % self.num_save_epochs == 0:
                save_path = os.path.join(self.save_path, wandb.run.id, str(epoch))
                os.makedirs(save_path, exist_ok=True)
                self.model.save(save_path)

    @abstractmethod
    def _train_step(self, images: torch.Tensor, labels: torch.Tensor, training: bool = True) -> TrainStepReturn:
        pass
    
    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader, epoch: int):
        train_mode = self.model.training
        self.model.eval()
        pred_list = []
        target_list = []
        loss_acc = 0.0
        i = 0
        for images, labels in tqdm(dataloader):
            images = images.to(self.device, non_blocking = True).to(self.dtype)
            labels = labels.to(self.device, non_blocking = True)
            loss, pred, target = self._train_step(images, labels, training = False)
            if self.calculate_metrics:
                pred_list.append(pred)
                target_list.append(target)
            loss_acc += loss
            i += 1
        loss = loss_acc / i
        if self.calculate_metrics:
            preds = torch.cat(pred_list).numpy()
            targets = torch.cat(target_list).numpy()
            metrics = self._calculate_metrics(preds, targets)
            metrics = {f"val_{k}": v for k, v in metrics.items()}
            metrics["val_loss"] = loss
            if epoch is not None:
                metrics["epoch"] = epoch
            wandb.log(metrics)
        #pylint:disable-next=expression-not-assigned
        self.model.train() if train_mode else self.model.eval()
        return metrics
    
    @torch.no_grad()
    def evaluate(self, ds: Dataset, sampler: Sampler, epoch: int = None):
        data = self._load_dataloader(ds, sampler)
        return self._evaluate(data, epoch)
        
    def _calculate_metrics(self, pred: np.array, target: np.array):
        metrics = {}
        for key, fun in self.metrics.items():
            metrics[key] = fun(target, pred)
        return metrics

# TODO Modify code to match new dataloader code in parent class
class DiffusionTrainer(TorchTrainer):
    def __init__(self, scheduler: SchedulerMixin, p_label_dropout: float,
                 num_inference_steps: int, num_generate: int, num_counterfactual: int, 
                 guidance_scale: float, use_dn: bool, percent_steps: float, **kwargs):
        # Dont initalizer optimizer just yet
        # Want to pass parameter groups and remove VAE
        optimizer_fun = kwargs.pop("optimizer_fun")
        # pass dummy function that returns None
        kwargs["optimizer_fun"] = lambda *args: None
        # Metrics are not supported as storing predictions would use too much memory
        # Also, there are not any metrics that use only one diffusion step
        assert not kwargs["metrics"], "No metrics should be used for training diffusion model as \
            saving predictions and targets would use too much memory for they are full sized images.\
            Can implenent custom metrics on generated or counterfactual images in evaluate if needed."
        super().__init__(**kwargs)
        self.p_label_dropout = p_label_dropout
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.num_generate = num_generate
        self.num_counterfactual = num_counterfactual
        self.guidance_scale = guidance_scale
        self.use_dn = use_dn
        self.percent_steps = percent_steps

        params = [
            (ParameterGroupNames.CLASS_EMBEDDER.value, self.model.class_embedder.parameters()),
            (ParameterGroupNames.UNET.value, self.model.unet.parameters())
        ]
        self.optimizer = optimizer_fun(params=params)

    @torch.no_grad()
    def _precompute(self, dataloader: DataLoader) -> Dataset:
        # store latents and labels in two large tensors and create dataset
        latents = []
        labels = []
        # Only have VAE on GPU for now
        self.model.to("cpu")
        vae = self.model.vae
        vae.to(self.device)
        torch.cuda.empty_cache()
        print("Precomputing latents...")
        for img, label in tqdm(dataloader):
            latent = vae.encode(img.to(self.device))
            latents.append(latent.cpu())
            labels.append(label)
        latents = torch.cat(latents, dim=0)
        labels = torch.cat(labels, dim=0)
        new_dataset = TensorDataset(latents, labels)
        # Move VAE to CPU as it isn't being used durring training
        # Move all other sub-models to GPU
        self.model.to(self.device)
        self.model.vae.to("cpu")
        torch.cuda.empty_cache()
        return new_dataset

    def _train_step(self, images: torch.Tensor, labels: torch.Tensor, training: bool = True) -> float:
        # Device locations should be okay for training
        # TODO verify
        precompute_latents = self.precompute_train if training else self.precompute_val
        latents = images if precompute_latents else self.model.vae.encode(images)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps,
                                  (images.size(0),), device = self.device)
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        if self.p_label_dropout > 0:
            drop_mask = torch.rand(labels.shape, device = self.device) < self.p_label_dropout
            labels = labels.masked_fill(drop_mask, self.model.class_embedder.null_class_label)
        self.optimizer.zero_grad()
        with torch.amp.autocast(self.device_str, dtype = self.dtype,
                                enabled = (self.mixed_precision in ["bf16", "fp16"])):
            class_embeddings = self.model.class_embedder(labels)
            noise_pred = self.model.unet(noisy_latents, timesteps,
                                   encoder_hidden_states = class_embeddings).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
        if training and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif training:
            loss.backward()
            self.optimizer.step()
        # Don't want to save target and prediction as they are full sized images
        return loss.item(), None, None
    
    def _generate_images(self):
        generator = torch.Generator(self.device).manual_seed(self.seed)
        classes = [0, 1, self.model.class_embedder.null_class_label]
        labels = torch.tensor(classes * self.num_generate, device = self.device)
        new_images = self.model.generate(labels, self.num_inference_steps, self.guidance_scale, rescale = True, generator = generator)
        np_images = new_images.cpu().permute(0, 2, 3, 1).numpy()
        label_names = ["NRG", "RG", "Null"]
        fig = create_grid(np_images, col_names = [label_names[c] for c in classes])
        return fig
    
    def _get_counterfactuals(self, val: DataLoader):
        val = iter(val)
        n_neg, n_pos = 0, 0
        neg_list, pos_list = [], []
        N = self.num_counterfactual
        done = False
        for batch in val:
            for img, label in zip(batch):
                if label == 0 and n_neg < N:
                    neg_list.append(img)
                    n_neg += 1
                elif label == 1 and n_pos < N:
                    pos_list.append(img)
                    n_pos += 1
                if n_neg >= N and n_pos >= N:
                    done = True
                    break
            if done:
                break
        images = torch.stack(neg_list + pos_list).to(self.device)
        labels = torch.tensor([0] * N + [1] * N)
        scheduler = DDIMScheduler.from_config(self.scheduler.config)
        counterfactuals = self.model.get_counterfactual(images, scheduler,
                                                        num_inference_steps = self.num_inference_steps,
                                                        guidance_scale = self.guidance_scale,
                                                        percent_steps = self.percent_steps,
                                                        use_dn = self.use_dn,
                                                        rescale = True)
        orig, new, heatmap = (x.cpu().permute(0, 2, 3, 1).numpy() for x in counterfactuals)
        fig = create_counterfactual_grid(orig, new, heatmap, labels)
        return fig
    
    def _evaluate(self, dataloader: DataLoader, epoch: int):
        super()._evaluate(dataloader, epoch)
        if epoch % self.num_save_epochs == 0 or epoch == self.num_epochs:
            gen_fig = self._generate_images()
            cf_fig = self._get_counterfactuals(dataloader)
            wandb.log({
                "epoch": epoch,
                "Generated Images": wandb.Image(gen_fig),
                "Counterfactual Images": wandb.Image(cf_fig)
            })
            self.model.save(os.path.join(self.save_path, wandb.run.id, f"{epoch:3d}"))
        

class CounterfactualTorchTrainer(TorchTrainer):
    def __init__(self, diffusion_model: src.model.LatentDiffusionModel, scheduler: DDIMScheduler, inference_hyperparams: dict, **kwargs):
        super().__init__(**kwargs)
        diffusion_model.eval()
        self.diffusion_model = diffusion_model
        self.scheduler = scheduler
        self.inference_hyperparams = inference_hyperparams

    def _precompute(self, dataloader: DataLoader):
        self.diffusion_model.to(self.device)
        self.model.to("cpu")
        cf_heatmaps = []
        labels = []
        logged = False
        print("Precomputing heatmaps...")
        for img, label in tqdm(dataloader):
            img = img.to(self.device)
            _, _, hm = self.diffusion_model.get_counterfactual(img, self.scheduler, rescale = True, **self.inference_hyperparams)
            if not logged:
                hm_numpy = hm.permute((0, 2, 3, 1)).numpy()
                wandb.log({
                    "Classifier Inputs": create_grid(hm_numpy),
                })
                logged = True
            cf_heatmaps.append(hm)
            labels.append(label)
        heatmaps = torch.cat(cf_heatmaps)
        labels = torch.cat(labels)
        new_dataset = TensorDataset(heatmaps, labels)
        self.diffusion_model.to("cpu")
        self.model.to(self.device)
        torch.cuda.empty_cache()
        return new_dataset
    
    # TODO Can probably factor this code out and put it in TorchTrainer _train_step
    def _train_step(self, images, labels, training = True):
        self.optimizer.zero_grad()
        with torch.amp.autocast(self.device_str, dtype = self.dtype,
                                enabled = (self.mixed_precision in ["bf16", "fp16"])):
            preds = self.model(images)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, labels.float())
        if training and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif training:
            loss.backward()
            self.optimizer.step()
        return loss, preds.detach().cpu().float(), labels.cpu()

class CounterfacturalSklearnTrainer(TrainerInterface):
    def __init__(self, diffusion_model: src.model.LatentDiffusionModel, classifier,
                 scheduler: DDIMScheduler, batch_size: int, num_workers: int, inference_hyperparams: dict):
        self.diffusion_model = diffusion_model
        self.classifier = classifier
        self.scheduler = scheduler

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.inference_hyperparams = inference_hyperparams
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
    
    @torch.no_grad()
    def _get_heatmaps(self, img_batch):
        # TODO Bundle scheduler with LatentDiffusionModel and save scheduler with model
        # TODO move inference hyperparams to own section in config
        # TODO finish method
        scheduler = self.scheduler
        return self.diffusion_model.get_counterfactual(img_batch, scheduler, rescale=True, **self.inference_hyperparams)

    def _calculate_metrics(self, y_probs: np.ndarray, y: np.ndarray, threshold = 0.5):
        # accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, \
        #                     precision_score, recall_score
        y_pred = (y_probs >= threshold).astype(int)
        metrics = {}
        metrics["Accuracy"] = accuracy_score(y, y_pred)
        metrics["Balanced Accuracy"] = balanced_accuracy_score(y, y_pred)
        metrics["ROC-AUC"] = roc_auc_score(y, y_probs)
        metrics["F1 score"] = f1_score(y, y_pred)
        metrics["Specificity"] = precision_score(y, y_pred)
        metrics["Sensitivity"] = recall_score(y, y_pred)
        # TODO add sensitivity at 95% specificity
        # TODO: should be a parameter to init or obtained from dataset
        CLASS_NAMES = ["NRG", "RG"]
        metrics["Confusion Matrix"] = wandb.plot.confusion_matrix(preds = y_pred, y_true = y,
                                                                  class_names=CLASS_NAMES)
        metrics["ROC Curve"] = wandb.plot.roc_curve(y, y_probs, CLASS_NAMES)
        metrics["PR Curve"] = wandb.plot.pr_curve(y, y_probs, CLASS_NAMES)

        wandb.log(metrics)
    
    def _get_data(self, ds, sampler):
        self.diffusion_model.to(self.device)
        dataloader = DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                       num_workers=self.num_workers, pin_memory=True, sampler = sampler)
        X = []
        y = []
        for img, label in dataloader:
            hm = self._get_heatmaps(img)
            # Convert to numpy on batch to reduce memory useage
            X = self.classifier.get_features(hm)
            y = label.cpu().numpy()
            X.append(X)
            y.append(y)

        # pylint: disable-next=invalid-name
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y)
        return X, y
    
    def train(self, train_ds, val_ds, train_sampler, val_sampler):
        X, y = self._get_data(train_ds, train_sampler)

        self.classifier.fit(X, y)

        # Training metrics
        y_probs = self.classifier.model.predict_proba(X)
        self._calculate_metrics(y_probs, y)
        
        # Validation metrics
        val_metrics = self.evaluate(val_ds, val_sampler)
        val_metrics = {f"val_{key}": val for key, val in val_metrics.items()}
        wandb.log(val_metrics)

    def evaluate(self, ds, sampler, epoch=None):
        X, y = self._get_data(ds, sampler)
        y_probs = self.classifier(X)
        self._calculate_metrics(y_probs, y)
        return (y_probs >= 0.5).astype(int)
    
