#pylint: disable=E0401
import hydra
from hydra.utils import instantiate
from diffusers import AutoencoderKL
import torch
from torchvision.transforms import v2
import wandb
from omegaconf import DictConfig, OmegaConf

# List enum values here
OmegaConf.register_new_resolver("interpolation", lambda name: v2.InterpolationMode[name])

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
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(project = cfg.project_name, config = wandb_config)
    class_embedder = instantiate(cfg.model.class_embedder)
    # wandb.watch(class_embedder, log="all", log_freq=100)
    unet = instantiate(cfg.model.unet)
    # wandb.watch(unet, log="all", log_freq=100)
    vae = AutoencoderKL.from_pretrained(cfg.model.vae.hf_id)
    optimizer=instantiate(cfg.training.optimizer)(params=[
        {"params": class_embedder.parameters()},
        {"params": unet.parameters()}])
    # scheduler = instantiate(cfg.model.scheduler)
    transformations = instantiate(cfg.data.transformations)
    augmentations = instantiate(cfg.data.augmentations)
    trainer = instantiate(cfg.training.trainer)
    trainer.train(vae, class_embedder, unet, train, val, optimizer,
                  transformations, augmentations, train_sampler, val_sampler)

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter