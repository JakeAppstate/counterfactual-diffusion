#pylint: disable=E0401
from enum import Enum
from typing import List, Dict, Tuple, Union, Iterator, Optional
from collections.abc import Callable, Sequence
from functools import partial
from abc import ABC, abstractmethod
import os
from cv2 import medianBlur
import numpy as np
import matplotlib.pyplot as plt
import lpips
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, \
                            f1_score, precision_score, recall_score
from diffusers import DDPMScheduler, DDIMScheduler, EMAModel
import wandb
from torch.utils.data import Dataset, TensorDataset, DataLoader, Sampler, RandomSampler, WeightedRandomSampler, SequentialSampler, Subset
import torch
from tqdm import tqdm

import src.model
from src.utils import create_grid, create_counterfactual_grid
from src.data import BaseDataset, MapDataset, ZippedDataset

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
                 precompute: bool, metrics: Metrics, steps_per_log: int,
                 wandb_init_fun: Callable[[], Optional[wandb.Run]], calculate_metrics: bool,
                 seed: int, num_save_epochs: int, save_path: str):
        self.model = model
        self.num_epochs = num_epochs
        self.loss = loss
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_precision = mixed_precision
        self.precompute = precompute
        self.metrics = metrics
        self.steps_per_log = steps_per_log
        self.wandb_init_fun = wandb_init_fun
        self.calculate_metrics = calculate_metrics
        self.seed = seed
        self.num_save_epochs = num_save_epochs
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)

        # Get optimizer in train function
        self.optimizer_fun = optimizer_fun
        self.optimizer = None
        # self.optimizer = optimizer_fun(self.model.parameters())

        if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
            self.scaler = None
        elif mixed_precision == "fp16":
            self.dtype = torch.float16
            self.scaler = torch.amp.GradScaler(self.device_str)
        else:
            self.dtype = torch.float32
            self.scaler = None

        self.global_step = 0

    # pylint:disable-next=unused-argument
    def _precompute(self, dataloader: DataLoader) -> Dataset:
        return None
    
    def _get_dataloader(self, ds: Dataset, sampler: Sampler):
        return DataLoader(ds, batch_size = self.batch_size, shuffle = False,
                           num_workers = self.num_workers, pin_memory = True,
                           sampler = sampler)

    def _load_dataloader(self, ds: Dataset, sampler: Sampler) -> DataLoader:
        if self.precompute:
            identity_sampler = SequentialSampler(ds)
            data = self._get_dataloader(ds, identity_sampler)
            # _precompute might not be implemented and return None
            ds = self._precompute(data) or ds
        data = self._get_dataloader(ds, sampler)
        return data

    def _get_optimizer(self):
        return self.optimizer_fun(self.model.parameters())

    def _train_setup(self, train_ds: Dataset, val_ds: Dataset, train_sampler: Sampler,
                     val_sampler: Sampler) -> Tuple[DataLoader, DataLoader]:
        self.model.train()
        self.model.to(self.device)
        self.loss.train()
        self.optimizer = self._get_optimizer()
        train = self._load_dataloader(train_ds, train_sampler)
        val = self._load_dataloader(val_ds, val_sampler)
        self.global_step = 0
        return train, val

    def train_one_epoch(self, train: DataLoader, val: DataLoader, epoch: int = 1):
        pred_list = []
        target_list = []
        print("Starting epoch:", epoch)
        running_loss = 0.0
        total_loss = 0.0
        running_samples = 0
        total_samples = 0
        for step, (x, y) in enumerate(tqdm(train)):
            # torch tensors are not considered sequences
            x = (x,) if not isinstance(x, Sequence) else x
            x = tuple(t.to(self.device, non_blocking = True) for t in x)
            y = y.to(self.device, non_blocking = True)
            n = len(y)
            loss, pred, target = self._train_step(x, y)
            running_loss += loss * n
            total_loss += loss * n
            running_samples += n
            total_samples += n
            self.global_step += 1
            # Reduces the number of network requests
            if self.steps_per_log != 0 and step % self.steps_per_log == 0:
                avg_step_loss = running_loss / running_samples
                wandb.log({
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "loss": avg_step_loss
                })
                running_loss = 0.0
                running_samples = 0
            # For diffusion models prediction and targets are full sized images
            # Storing them would use too much memory (N * 512 * 512 * 3 * 4 bytes)
            if self.calculate_metrics:
                pred_list.append(pred)
                target_list.append(target)
        metrics = {}
        avg_loss = total_loss / total_samples
        metrics["avg_loss"] = avg_loss
        metrics["loss"] = running_loss / running_samples
        if self.calculate_metrics:
            pred = torch.cat(pred_list).numpy()
            target = torch.cat(target_list).numpy()
            metrics = self._calculate_metrics(pred, target)
            metrics["epoch"] = epoch
            # wandb.log(metrics)
        print("Performing validation")
        metrics = metrics | self._evaluate(val, epoch)
        return metrics

    def _save_model(self, epoch):
        save_path = os.path.join(self.save_path, wandb.run.id, str(epoch))
        os.makedirs(save_path, exist_ok=True)
        self.model.save(save_path)
    
    def train(self, train_ds: Dataset, val_ds: Dataset,
              train_sampler: Sampler, val_sampler: Sampler):
        train, val = self._train_setup(train_ds, val_ds, train_sampler, val_sampler)
        self.wandb_init_fun()
        for epoch in range(1, self.num_epochs + 1):
            metrics = self.train_one_epoch(train, val, epoch)
            wandb.log(metrics)
            if epoch % self.num_save_epochs == 0:
                self._save_model(epoch)
                

    def _train_step(self, x: Tuple[torch.Tensor, ...], y: torch.Tensor, training: bool = True) -> TrainStepReturn:
        self.optimizer.zero_grad(set_to_none = True)
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
        self.loss.eval()
        pred_list = []
        target_list = []
        loss_acc = 0.0
        num_samples = 0
        for x, y in tqdm(dataloader):
            x = (x,) if not isinstance(x, Sequence) else x
            x = tuple(t.to(self.device, non_blocking = True) for t in x)
            y = y.to(self.device)
            n = len(y)
            loss, pred, target = self._train_step(x, y, training = False)
            if self.calculate_metrics:
                pred_list.append(pred)
                target_list.append(target)
            loss_acc += loss * n
            num_samples += n
        loss = loss_acc / num_samples
        metrics = {}
        if self.calculate_metrics:
            preds = torch.cat(pred_list).numpy()
            targets = torch.cat(target_list).numpy()
            metrics = self._calculate_metrics(preds, targets)
            metrics = {f"val_{k}": v for k, v in metrics.items()}
        metrics["val_loss"] = loss
        if epoch is not None:
            metrics["epoch"] = epoch
        # wandb.log(metrics)
        #pylint:disable-next=expression-not-assigned
        self.model.train() if train_mode else self.model.eval()
        #pylint:disable-next=expression-not-assigned
        self.loss.train() if train_mode else self.loss.eval()
        return metrics
    
    @torch.no_grad()
    def evaluate(self, ds: Dataset, sampler: Sampler):
        data = self._load_dataloader(ds, sampler)
        metrics = self._evaluate(data, None)
        wandb.log(metrics)
        return metrics
        
    def _calculate_metrics(self, pred: np.array, target: np.array):
        metrics = {}
        for key, fun in self.metrics.items():
            metrics[key] = fun(target, pred)
        return metrics

# TODO Modify code to match new dataloader code in parent class
class DiffusionTrainer(TorchTrainer):
    def __init__(self, p_label_dropout: float, num_inference_steps: int,
                 ema_decay: float, num_generate: int, num_counterfactual: int,
                 guidance_scale: float, use_dn: bool, percent_steps: float,
                 encode_base: bool, **kwargs):
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
        self.ema_decay = ema_decay
        # TODO may want to bundle into inference dict and unpack when calling model
        self.guidance_scale = guidance_scale
        self.use_dn = use_dn
        self.percent_steps = percent_steps
        self.encode_base = encode_base
        self.scheduler = DDPMScheduler.from_config(self.model.scheduler.config)
        self.cf_data = None

        self.ema_model = None

    def _get_optmizer(self):
        params = [
            (ParameterGroupNames.CLASS_EMBEDDER.value, self.model.class_embedder.parameters()),
            (ParameterGroupNames.UNET.value, self.model.unet.parameters())
        ]
        return self.optimizer_fun(params = params)

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
    
    def _train_setup(self, train_ds, val_ds, train_sampler, val_sampler):
        self.cf_data = self._get_counterfactual_dataset(val_ds)
        self.ema_model = EMAModel(
            self.model.unet.parameters(),
            decay = self.ema_decay,
            model_cls = type(self.model.unet),
            model_config = self.model.unet.config
        )
        self.ema_model.to(self.device)
        return super()._train_setup(train_ds, val_ds, train_sampler, val_sampler)

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
        self.ema_model.step(self.model.unet.parameters())
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
                                                        encode_base = self.encode_base, log = True)
        orig, new, heatmap = (x.cpu().permute(0, 2, 3, 1).numpy() for x in counterfactuals)
        fig = create_counterfactual_grid(orig, new, heatmap, labels)
        return fig
    
    def _save_model(self, epoch):
        if self.ema_model is not None:
            self.ema_model.store(self.model.unet.parameters())
            self.ema_model.copy_to(self.model.unet.parameters())
            super()._save_model(epoch)
            self.ema_model.restore(self.model.unet.parameters())
        else:
            super()._save_model(epoch)

    
    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader, epoch: int):
        if self.ema_model is not None:
            # move ema weights to unet for evaluation
            self.ema_model.store(self.model.unet.parameters())
            self.ema_model.copy_to(self.model.unet.parameters())
        metrics = super()._evaluate(dataloader, epoch)
        if epoch % self.num_save_epochs == 0 or epoch == self.num_epochs or epoch == 1:
            if self.precompute:
                self.model.vae.to(self.device)
            gen_fig = self._generate_images()
            cf_fig = self._get_counterfactuals(self.cf_data)
            metrics["Generated Images"] = wandb.Image(gen_fig)
            metrics["Counterfactual Images"] = wandb.Image(cf_fig)
            plt.close()
            if self.precompute:
                self.model.vae.to("cpu")
                torch.cuda.empty_cache()
        if self.ema_model is not None:
            # move ema weights to unet for evaluation
            self.ema_model.restore(self.model.unet.parameters())
        return metrics

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
    
    def _train_step(self, x, y, training = True):
        y = y.float()
        return super()._train_step(x, y, training)

    
class CrossValidationTrainer(TrainerInterface):
    def __init__(self,  create_trainer_fun: Callable[[], TorchTrainer],
                 n_folds: int, seed: int, num_epochs: int,
                 wandb_init_fun: Callable[[], Optional[wandb.Run]]):
        # Dont have sub trainer call wandb.init
        create_trainer_fun = partial(create_trainer_fun, wandb_init_fun = lambda: None)
        self.create_trainer_fun = create_trainer_fun
        self.n_folds = n_folds
        self.seed = seed
        self.num_epochs = num_epochs
        self.wandb_init_fun = wandb_init_fun
        # loss is logged every 50 epochs
        # avg_loss would make more sense to average across folds
        self.non_metric_keys = ["loss", "epoch", "global_step"]

    def _get_sampler(self, sampler: Sampler, ds: Subset):
        if isinstance(sampler, SequentialSampler):
            sampler = SequentialSampler(ds)
        elif isinstance(sampler, RandomSampler):
            sampler = RandomSampler(ds, sampler.replacement, generator = sampler.generator)
        elif isinstance(sampler, WeightedRandomSampler):
            idx = torch.tensor(ds.indicies)
            weights = sampler.weights[idx]
            sampler = WeightedRandomSampler(weights, replacement = sampler.replacement,
                                            generator = sampler.generator)
        else:
            raise TypeError(f"Sampler type {type(sampler)} is not supported")
        return sampler

    def _get_folds(self, ds: Dataset):
        trainer = self.create_trainer_fun()
        ds = self._precompute(ds, trainer)
        # Calculate fold indicies and get subsets
        folds = []
        if isinstance(ds, BaseDataset):
            y = ds.get_targets()
        elif isinstance(ds, TensorDataset):
            y = ds.tensors[1].numpy()
        else:
            raise RuntimeError("Can't currently get targets on dataset other than BaseDataset or TensorDataset")
        X = np.zeros_like(y)
        strat_kfold = sklearn.model_selection.StratifiedKFold(n_splits=self.n_folds, shuffle = True,
                                                              random_state = self.seed)
        for train_idx, val_idx in strat_kfold.split(X, y):
            train = Subset(ds, train_idx)
            val = Subset(ds, val_idx)
            folds.append((train, val))
        del trainer
        return folds

    def _precompute(self, ds: Dataset, trainer: TorchTrainer):
        sampler = SequentialSampler(ds)
        precompute_val = trainer.precompute
        trainer.precompute = False
        # pylint: disable-next=protected-access
        dataloader = trainer._get_dataloader(ds, sampler)
        # pylint: disable-next=protected-access
        new_ds = trainer._precompute(dataloader) if precompute_val else ds
        trainer.precompute = precompute_val
        return new_ds

    def _modify_metrics(self, metrics: dict, new_metrics: dict) -> dict:
        new_metrics = {k: v for k, v in new_metrics.items()
                       if isinstance(v, float) or isinstance(v, int)}
        if metrics is None:
            return new_metrics
        for k in metrics:
            if k in self.non_metric_keys:
                continue
            metrics[k] += new_metrics[k]
        return metrics
    
    def _average_metrics(self, metrics: dict) -> dict:
        for k in metrics:
            if k in self.non_metric_keys:
                continue
            metrics[k] /= self.n_folds
        return metrics

    def train(self, train_ds: Dataset, val_ds: Dataset,
              train_sampler: Dataset, val_sampler: Dataset):
        # ds = self._precompute(train_ds, train_sampler)
        self.wandb_init_fun()
        folds = self._get_folds(train_ds)
        trainers = [self.create_trainer_fun() for i in range(len(folds))]
        for epoch in range(1, self.num_epochs + 1):
            metrics = None
            for i, trainer in enumerate(trainers):
                trainer.precompute = False
                train_ds, val_ds = folds[i]
                fold_train_sampler = self._get_sampler(train_sampler, train_ds)
                fold_val_sampler = self._get_sampler(val_sampler, val_ds)
                train, val = trainer._train_setup(train_ds, val_ds, fold_train_sampler, fold_val_sampler)
                trainer.save_path = os.path.join(trainer.save_path, f"fold{i}")
                new_metrics = trainer.train_one_epoch(train, val, epoch)
                metrics = self._modify_metrics(metrics, new_metrics)
            metrics = self._average_metrics(metrics)
            wandb.log(metrics)

    def evaluate(self, ds, sampler):
        trainer = self.create_trainer_fun()
        return trainer.evaluate(ds, sampler)

class HyperParameterTuningTrainer(TrainerInterface):
    def __init__(self, seed: int, n_folds: int, num_epochs: int,
                 trainer_fun: Callable[..., TorchTrainer],
                 project_name: str,
                 sweep_config: dict, wandb_init_fun):
        self.seed = seed
        self.n_folds = n_folds
        self.trainer_fun = trainer_fun
        self.num_epochs = num_epochs
        self.sweep_config = sweep_config
        self.project_name = project_name
        self.wandb_init_fun = wandb_init_fun

    def train_sweep(self, train_ds, train_sampler, val_sampler):
        with self.wandb_init_fun():
            hyperparams = {k: v for k, v in dict(wandb.config).items()
                           if k in self.sweep_config["parameters"]}
            # TODO modify CFTrainer to take hyperparams directly with dict unpacking
            trainer_fun = partial(self.trainer_fun, inference_hyperparams = hyperparams)
            trainer = CrossValidationTrainer(trainer_fun, self.n_folds,
                                             self.seed, self.num_epochs, lambda: None)
            trainer.train(train_ds, None, train_sampler, val_sampler)

    def train(self, train_ds, val_ds, train_sampler, val_sampler):
        sweep_id = wandb.sweep(self.sweep_config, project=self.project_name)
        print("Starting sweep for ID:", sweep_id)
        sweep_fun = partial(self.train_sweep, train_ds = train_ds,
                            train_sampler = train_sampler, val_sampler = val_sampler)
        wandb.agent(sweep_id, function = sweep_fun, count = 20)

    def evaluate(self, ds, sampler, epoch = None):
        raise NotImplementedError("Evaluating does not make sense for hyperparameter tuning class")

class NewVAELoss(torch.nn.Module):
    def __init__(self, lpips_weight: float):
        super().__init__()
        self.lpips_weight = lpips_weight
        # TODO might want to use trainer device instead of hardcoding cuda
        lpips_loss = lpips.LPIPS(net="alex").to("cuda").eval()
        for param in lpips_loss.parameters():
            param.requires_grad = False
        self.lpips_loss = lpips_loss

    def forward(self, pred, target):
        mae_loss = torch.nn.functional.l1_loss(pred, target)
        lpips_loss = self.lpips_loss(pred, target).mean()
        val_key = "val_" if not self.training else ""
        wandb.log({
            f"{val_key}l1_loss": mae_loss,
            f"{val_key}lpips_loss": lpips_loss
        }, commit = False)
        return mae_loss + self.lpips_weight * lpips_loss

    def train(self, mode: bool = True):
        super().train(mode)
        self.lpips_loss.eval()
        return self

# TODO Figure out how to average lpips and l1 over num_log_steps
class NewVAETrainer(TorchTrainer):
    def __init__(self, num_heatmaps, **kwargs):
        super().__init__(**kwargs)
        # TODO may need to modify trainable parameters
        self.num_heatmaps = num_heatmaps
        self.hm_data = None
        # Freeze encoder weights
        for param in self.model.vae.encoder.parameters():
            param.requires_grad = False

    def _load_dataloader(self, ds, sampler):
        # TODO Check if there is a bug that latents and images dont line up
        # Not sure how to check but solution would be 
        identity_sampler = SequentialSampler(ds)
        data = self._get_dataloader(ds, identity_sampler)
        # Don't want img to be converted into TensorDataset as they are too large to fit into memory
        # Keep latents in memory and load images on the fly
        # Label isn't used for training the VAE
        def join_fn(img, label, mean, logvar):
            return (mean, logvar), img
            
        if self.precompute:
            new_ds = self._precompute(data)
            zipped_ds = ZippedDataset(ds, new_ds)
            ds = MapDataset(zipped_ds, join_fn)
        else:
            # Want img to be label
            # Do this after creating the original dataloader to save memory when precomputing
            ds = MapDataset(ds, lambda x, y : (x, x))
        data = self._get_dataloader(ds, sampler)
        return data
    
    def _precompute(self, dataloader) -> Dataset:
        self.model.vae.encoder.to(self.device)
        self.model.vae.decoder.to("cpu")
        torch.cuda.empty_cache()
        mean_list, logvar_list = [], []
        print("Precomputing latents")
        for img, _ in tqdm(dataloader):
            img = img.to(self.device)
            mean, logvar = self.model.get_latent_dist(img)
            mean_list.append(mean.cpu())
            logvar_list.append(logvar.cpu())
        mean_tensor = torch.cat(mean_list)
        logvar_tensor = torch.cat(logvar_list)
        new_dataset = TensorDataset(mean_tensor, logvar_tensor)
        self.model.vae.encoder.to("cpu")
        self.model.vae.decoder.to(self.device)
        torch.cuda.empty_cache()
        return new_dataset
    
    def _get_images_for_heatmap(self, dataset: Dataset):
        img_list, label_list = [], []
        for i in range(self.num_heatmaps):
            img, label = dataset[i]
            img_list.append(img)
            label_list.append(label)
        images = torch.stack(img_list)
        labels = torch.tensor(label_list)
        return images, labels

    def _train_setup(self, train_ds, val_ds, train_sampler, val_sampler):
        self.hm_data = self._get_images_for_heatmap(val_ds)
        return super()._train_setup(train_ds, val_ds, train_sampler, val_sampler)
    
    def _train_step(self, x, y, training = True):
        if not self.precompute:
            # Currently, get_latent_dist uses .encode which does not use gradients
            # x is a tuple of length 1
            x = self.model.get_latent_dist(x[0])
        return super()._train_step(x, y, training)

    def _calculate_heatmaps(self, images, labels):
        images = images.to(self.device)
        new_images = self.model.decode(self.model.encode(images, sample = False)).cpu()
        new_images = torch.clamp(new_images, -1, 1)
        images = images.cpu()
        heatmaps = torch.mean(torch.abs(new_images - images), dim = 1, keepdim = True)
        images = ((images + 1) / 2).permute((0, 2, 3, 1)).numpy()
        new_images = ((new_images + 1) / 2).permute((0, 2, 3, 1)).numpy()
        heatmaps = ((heatmaps + 1) / 2).permute((0, 2, 3, 1)).numpy()
        col_names = ["Original Image", "Reconstructed Image", "Difference Map"]
        fig = create_counterfactual_grid(images, new_images, heatmaps, labels, col_names)
        return fig
    
    def _evaluate(self, dataloader, epoch):
        metrics = super()._evaluate(dataloader, epoch)
        # move encoder to GPU if it isn't on it
        encoder_device = next(self.model.vae.encoder.parameters()).device
        self.model.vae.encoder.to(self.device)
        fig = self._calculate_heatmaps(*self.hm_data)
        wandb.log({"difference_heatmaps": fig, "epoch": epoch})
        plt.close(fig)
        # Move encoder back to its origional device
        self.model.vae.encoder.to(encoder_device)
        return metrics

class VAETrainer(TorchTrainer):
    def __init__(self, num_heatmaps, mask_threshold: float, **kwargs):
        super().__init__(**kwargs)
        self.num_heatmaps = num_heatmaps
        self.mask_threshold = mask_threshold
        self.hm_data = None
        self.model.vae.enable_gradient_checkpointing()

    def _get_images_for_heatmap(self, dataset: Dataset):
        img_list, label_list = [], []
        for i in range(self.num_heatmaps):
            img, label = dataset[i]
            img_list.append(img)
            label_list.append(label)
        images = torch.stack(img_list)
        labels = torch.tensor(label_list)
        return images, labels
    
    def _get_mask(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim = -3, keepdim = True)[0] > self.mask_threshold
    
    def _load_dataloader(self, ds: Dataset, sampler: Sampler) -> DataLoader:
        def add_mask(img, _):
            # data isn't batched yet so concat on dim 0
            return img, torch.cat((img, self._get_mask(img)))
        # NOTE can't access labels when using dataloaders
        ds = MapDataset(ds, add_mask)
        return super()._load_dataloader(ds, sampler)
    
    def _train_setup(self, train_ds, val_ds, train_sampler, val_sampler):
        self.hm_data = self._get_images_for_heatmap(val_ds)
        super()._train_setup(train_ds, val_ds, train_sampler, val_sampler)

    def _calculate_heatmaps(self, images, labels):
        images = images.to(self.device)
        new_images = self.model.decode(self.model.encode(images)).cpu()
        new_images = torch.clamp(new_images, -1, 1)
        images = images.cpu()
        mask = self._get_mask(images)
        heatmaps = torch.mean(torch.abs(new_images - images) * mask, dim = 1, keepdim = True)
        images = ((images + 1) / 2).permute((0, 2, 3, 1)).numpy()
        new_images = ((new_images + 1) / 2).permute((0, 2, 3, 1)).numpy()
        heatmaps = ((heatmaps + 1) / 2).permute((0, 2, 3, 1)).numpy()
        col_names = ["Original Image", "Reconstructed Image", "Difference Map"]
        fig = create_counterfactual_grid(images, new_images, heatmaps, labels, col_names)
        return fig

    def _evaluate(self, dataloader, epoch):
        metrics = super()._evaluate(dataloader, epoch)
        fig = self._calculate_heatmaps(*self.hm_data)
        wandb.log({"difference_heatmaps": fig, "epoch": epoch})
        plt.close(fig)
        return metrics
