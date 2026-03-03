#pylint: disable=E0401
from enum import Enum
from typing import List, Dict, Tuple, Union, Iterator, TypeVar
from collections.abc import Callable
from abc import ABC, abstractmethod
import os
from cv2 import medianBlur
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, \
                            f1_score, precision_score, recall_score
from diffusers import DDPMScheduler, DDIMScheduler, EMAModel
import wandb
from torch.utils.data import Dataset, TensorDataset, DataLoader, Sampler, Subset
import torch
from tqdm import tqdm

import src.model
from src.utils import create_grid, create_counterfactual_grid

# Custom Types
Metrics = Dict[str, Callable[[float, float], float]]
# pylint: disable-next=missing-class-docstring
class TorchModel(src.model.ModelInterface, torch.nn.Module):
    pass
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
    def evaluate(self, ds, sampler):
        pass

class TorchTrainer(TrainerInterface):
    def __init__(self, model: TorchModel, num_epochs: int, optimizer_fun: OptimizerFun,
                 loss: torch.nn.Module, batch_size: int, num_workers: int, mixed_precision: str,
                 precompute: bool, metrics: Metrics,
                 calculate_metrics: bool, seed: int, num_save_epochs: int, save_path: str):
        self.model = model
        self.num_epochs = num_epochs
        self.loss = loss
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_precision = mixed_precision
        self.precompute = precompute
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

    def _load_dataloader(self, ds: Dataset, sampler: Sampler) -> DataLoader:
        data = DataLoader(ds, batch_size = self.batch_size, shuffle = False,
                           num_workers = self.num_workers, pin_memory = True,
                           sampler = sampler)
        if self.precompute:
            # _precompute might not be implemented and return None
            ds = self._precompute(data) or ds
            data = DataLoader(ds, batch_size = self.batch_size, shuffle = False,
                           num_workers = self.num_workers, pin_memory = True,
                           sampler = sampler)
        return data
    
    def train(self, train_ds: Dataset, val_ds: Dataset, train_sampler: Sampler, val_sampler: Sampler):
        self.model.train()
        self.model.to(self.device)
        train = self._load_dataloader(train_ds, train_sampler)
        val = self._load_dataloader(val_ds, val_sampler)
        global_step = 0
        for epoch in range(1, self.num_epochs + 1):
            pred_list = []
            target_list = []
            print("Starting epoch:", epoch)
            for images, labels in tqdm(train):
                images = images.to(self.device, non_blocking = True)
                labels = labels.to(self.device, non_blocking = True)
                loss, pred, target = self._train_step((images,), labels)
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

    def _train_step(self, x: Tuple[torch.Tensor, ...], y: torch.Tensor, training: bool = True) -> TrainStepReturn:
        self.optimizer.zero_grad()
        with torch.amp.autocast(self.device_str, dtype = self.dtype,
                                enabled = (self.mixed_precision in ["bf16", "fp16"])):
            pred = self.model(*x)
            loss = self.loss(pred, y)
        if training and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif training:
            loss.backward()
            self.optimizer.step()
        # Don't return predictions and labels if they are not being used
        if not self.calculate_metrics:
            return loss.item(), None, None
        return loss.item(), pred.detach().cpu().float(), y.cpu()
    
    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader, epoch: int):
        train_mode = self.model.training
        self.model.eval()
        pred_list = []
        target_list = []
        loss_acc = 0.0
        i = 0
        for images, labels in tqdm(dataloader):
            images = images.to(self.device, non_blocking = True)
            labels = labels.to(self.device, non_blocking = True)
            loss, pred, target = self._train_step((images,), labels, training = False)
            if self.calculate_metrics:
                pred_list.append(pred)
                target_list.append(target)
            loss_acc += loss
            i += 1
        loss = loss_acc / i
        metrics = {}
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
    def evaluate(self, ds: Dataset, sampler: Sampler):
        data = self._load_dataloader(ds, sampler)
        return self._evaluate(data, None)
        
    def _calculate_metrics(self, pred: np.array, target: np.array):
        metrics = {}
        for key, fun in self.metrics.items():
            metrics[key] = fun(target, pred)
        return metrics

# TODO Modify code to match new dataloader code in parent class
class DiffusionTrainer(TorchTrainer):
    def __init__(self, p_label_dropout: float, num_inference_steps: int,
                 ema_decay: float, num_generate: int, num_counterfactual: int,
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
        self.num_inference_steps = num_inference_steps
        self.num_generate = num_generate
        self.num_counterfactual = num_counterfactual
        # TODO may want to bundle into inference dict and unpack when calling model
        self.guidance_scale = guidance_scale
        self.use_dn = use_dn
        self.percent_steps = percent_steps
        self.scheduler = DDPMScheduler.from_config(self.model.scheduler.config)
        self.cf_data = None

        self.ema_model = EMAModel(
            self.model.unet.parameters(),
            decay = ema_decay,
            model_cls = type(self.model.unet),
            model_config = self.model.unet.config
        )
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
    
    def train(self, train_ds, val_ds, train_sampler, val_sampler):
        self.cf_data = self._get_counterfactual_dataset(val_ds)
        super().train(train_ds, val_ds, train_sampler, val_sampler)

    def _train_step(self, x, y, training = True):
        # Device locations should be okay for training
        # TODO verify
        images, = x
        latents = images if self.precompute else self.model.vae.encode(images)
        timesteps = torch.randint(0, self.scheduler.config["num_train_timesteps"],
                                  (images.size(0),), device = self.device)
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        if self.p_label_dropout > 0:
            drop_mask = torch.rand(y.shape, device = self.device) < self.p_label_dropout
            y = y.masked_fill(drop_mask, self.model.class_embedder.null_class_label)
        loss, _, _ = super()._train_step((noisy_latents, timesteps, y), noise, training)
        # Don't want to save target and prediction as they are full sized images
        return loss, None, None
    
    def _generate_images(self):
        generator = torch.Generator(self.device).manual_seed(self.seed)
        classes = [0, 1, self.model.class_embedder.null_class_label]
        labels = torch.tensor(classes * self.num_generate, device = self.device)
        new_images = self.model.generate(labels, self.num_inference_steps, self.guidance_scale, rescale = True, generator = generator, log = True)
        np_images = new_images.cpu().permute(0, 2, 3, 1).numpy()
        label_names = ["NRG", "RG", "Null"]
        fig = create_grid(np_images, col_names = [label_names[c] for c in classes])
        return fig
    
    def _get_counterfactual_dataset(self, ds):
        n_neg, n_pos = 0, 0
        neg_list, pos_list = [], []
        N = self.num_counterfactual
        i = 0
        while n_neg < N or n_pos < N:
            img, label = ds[i]
            if label == 1 and n_pos < N:
                pos_list.append(img)
                n_pos += 1
            elif label == 0 and n_neg < N:
                neg_list.append(img)
                n_neg += 1
            i += 1
        images = torch.stack(neg_list + pos_list)
        labels = torch.tensor([0] * N + [1] * N)
        return images, labels

    def _get_counterfactuals(self, data: Tuple[torch.Tensor, torch.Tensor]):
        images, labels = data
        counterfactuals = self.model.get_counterfactual(images,
                                                        num_inference_steps = self.num_inference_steps,
                                                        guidance_scale = self.guidance_scale,
                                                        percent_steps = self.percent_steps,
                                                        use_dn = self.use_dn, rescale = True,
                                                        log = True)
        orig, new, heatmap = (x.cpu().permute(0, 2, 3, 1).numpy() for x in counterfactuals)
        fig = create_counterfactual_grid(orig, new, heatmap, labels)
        return fig
    
    def _evaluate(self, dataloader: DataLoader, epoch: int):
        super()._evaluate(dataloader, epoch)
        if epoch % self.num_save_epochs == 0 or epoch == self.num_epochs or epoch == 1:
            if self.precompute:
                self.model.vae.to(self.device)
            gen_fig = self._generate_images()
            cf_fig = self._get_counterfactuals(self.cf_data)
            wandb.log({
                "epoch": epoch,
                "Generated Images": wandb.Image(gen_fig),
                "Counterfactual Images": wandb.Image(cf_fig)
            })
            plt.close()
            self.model.save(os.path.join(self.save_path, wandb.run.id, f"{epoch:3d}"))
            if self.precompute:
                self.model.vae.to("cpu")
                torch.cuda.empty_cache()
        

class CounterfactualTorchTrainer(TorchTrainer):
    def __init__(self, diffusion_model: src.model.LatentDiffusionModel, inference_hyperparams: dict, **kwargs):
        super().__init__(**kwargs)
        diffusion_model.eval()
        self.diffusion_model = diffusion_model
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
            _, _, hm = self.diffusion_model.get_counterfactual(img, rescale = True, **self.inference_hyperparams)
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
    
    def _train_step(self, images, labels, training = True):
        labels = labels.float()
        return super()._train_step(images, labels, training)

    
class CrossValidationTrainer(TrainerInterface):
    def __init__(self,  trainers: List[TrainerInterface],
                 n_folds: int, seed: int):
        self.trainers = trainers
        self.n_folds = n_folds
        self.seed = seed

    def _get_folds(self, ds):
        folds = []
        y = ds.get_targets()
        X = np.zeros_like(y)
        strat_kfold = sklearn.model_selection.StratifiedKFold(n_splits=self.n_folds,
                                                              random_state = self.seed)
        for train_idx, val_idx in strat_kfold.split(X, y):
            train = Subset(ds, train_idx)
            val = Subset(ds, val_idx)
            folds.append((train, val))
        return folds
    
    def train(self, train_ds, val_ds, train_sampler, val_sampler):
        folds = self._get_folds(train_ds)
        for trainer in self.trainers:
            for i, (train, val) in enumerate(folds):
                trainer.save_path = os.path.join(trainer.save_path, f"fold{i}")
                trainer.train(train, val, train_sampler, val_sampler)

    def evaluate(self, ds, sampler, epoch = None):
        for trainer in self.trainers:
            trainer.evaluate(ds, sampler, epoch)
    
