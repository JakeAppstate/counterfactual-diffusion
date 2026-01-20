#pylint: disable=E0401
from hydra.utils import instantiate
import hydra
from diffusers import AutoencoderKL
import torchvision
import pandas as pd
import wandb
from omegaconf import DictConfig,OmegaConf

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    data_module = instantiate(cfg.dataset.data_module)
    train, val, test, real = data_module.load_datasets(cfg.dataset.val_ratio, cfg.dataset.test_ratio, cfg.dataset.include_real)
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
    wandb.watch(class_embedder, log="all", log_freq=100)
    unet = instantiate(cfg.model.unet)
    wandb.watch(unet, log="all", log_freq=100)
    vae = AutoencoderKL.from_pretrained(cfg.model.vae.hf_id)
    optimizer=instantiate(cfg.training.optimizer)(params=[
        {"params": class_embedder.parameters()},
        {"params": unet.parameters()}])
    # scheduler = instantiate(cfg.model.scheduler)
    transformations = torchvision.transforms.v2.Compose(instantiate(cfg.dataset.transforms))
    augmentations = torchvision.transforms.v2.Compose(instantiate(cfg.dataset.augmentations))
    img, _ = train[0]
    img = img.permute(1, 2, 0).numpy()
    wandb.log({"Train Dataset Image": wandb.Image(img)})
    trainer = instantiate(cfg.training.trainer)
    trainer.train(vae, class_embedder, unet, train, val,
              optimizer, transformations, augmentations)
    # train_dataloader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=train.collate_fn)
    # x, y = next(iter(train_dataloader))
    # print(f"Sample batch x shape: {x.shape}, y shape: {y.shape}")
    # for i in range(x.size(0)):
    #     save_image(x[i], f"sample_image_{i}_label_{y[i].item()}.png")

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
