#pylint: disable=E0401
from enum import Enum, auto
import hydra
from hydra.utils import instantiate
from diffusers import AutoencoderKL
import torch
from torchvision.transforms import v2
import wandb
from omegaconf import DictConfig, OmegaConf

class ModeEnum(Enum):
    TRAIN = auto()
    TRAIN_UNET = auto()
    TRAIN_CF = auto()
    EVALUATE = auto()
    EVALUATE_UNET = auto()
    EVALUATE_CF = auto()
    PLOT_CF_HYPERPARAMS = auto()

# List enum values here
# OmegaConf.register_new_resolver("mode", lambda name: ModeEnum[name])

def start_run(cfg):
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(project = cfg.project_name, config = wandb_config)

def train_unet(cfg, train, val, train_sampler, val_sampler):
    class_embedder = instantiate(cfg.model.class_embedder)
    unet = instantiate(cfg.model.unet)
    vae = AutoencoderKL.from_pretrained(cfg.model.vae.hf_id)
    optimizer=instantiate(cfg.training.optimizer)(params=[
        {"params": class_embedder.parameters()},
        {"params": unet.parameters()}])
    transformations = instantiate(cfg.data.transformations)
    augmentations = instantiate(cfg.data.augmentations)
    trainer = instantiate(cfg.training.trainer)
    start_run(cfg)
    trainer.train(vae, class_embedder, unet, train, val, optimizer,
                transformations, augmentations, train_sampler, val_sampler)
    
def train_cf(cfg, val):
    pass

def select_mode(mode, cfg, train, val):
    # TODO convert this to dict
    match mode:
        case ModeEnum.TRAIN:
            pass
        case ModeEnum.TRAIN_UNET:
            train
        case ModeEnum.TRAIN_CF:
            pass
        case ModeEnum.EVALUATE:
            pass
        case ModeEnum.EVALUATE_UNET:
            pass
        case ModeEnum.EVALUATE_CF:
            pass
        case ModeEnum.PLOT_CF_HYPERPARAMS:
            pass
        case ModeEnum.INFERENCE:
            pass
        case _:
            raise RuntimeError(f"'{mode}' must be in ModeEnum")

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    data_module = instantiate(cfg.data.data_module)
    datasets = data_module.load_datasets(cfg.data.val_ratio, cfg.data.test_ratio,
                                         cfg.data.include_real)
    train, val, test, real = datasets
    train_sampler = data_module.get_sampler(train, oversample = cfg.data.oversample)
    val_sampler = data_module.get_sampler(val, oversample = False,
                                          replacement = False, use_generator = True)
    print("Datasets loaded:")
    print(f"Train set size: {len(train)}")
    print(f"Validation set size: {len(val)}")
    print(f"Test set size: {len(test)}")
    print(f"Real set size: {len(real)}")

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
