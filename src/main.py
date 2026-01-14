#pylint: disable=E0401
from hydra.utils import instantiate
import hydra
from diffusers import AutoencoderKL
import torchvision
import wandb
from omegaconf import DictConfig

from src.train import train_model

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    wandb.init(project = cfg.project_name, config = cfg)
    data_module = instantiate(cfg.dataset.data_module)
    train, val, test, real = data_module.load_datasets(cfg.dataset.val_ratio, cfg.dataset.test_ratio, cfg.dataset.include_real)
    print("Datasets loaded:")
    print(f"Train set size: {len(train)}")
    print(f"Validation set size: {len(val)}")
    print(f"Test set size: {len(test)}")
    print(f"Real set size: {len(real)}")
    class_embedder = instantiate(cfg.model.class_embedder)
    wandb.watch(class_embedder, log="all", log_freq=100)
    unet = instantiate(cfg.model.unet)
    wandb.watch(unet, log="all", log_freq=100)
    vae = AutoencoderKL.from_pretrained(cfg.model.vae.hf_id)
    scheduler = instantiate(cfg.model.scheduler)
    transformations = torchvision.transforms.Compose([
        instantiate(cfg.dataset.yolo),
        *instantiate(cfg.dataset.transforms),
    ])
    augmentations = torchvision.transforms.Compose(*instantiate(cfg.dataset.augmentations))
    train_model(
        vae=vae,
        class_embedder=class_embedder,
        unet=unet,
        scheduler=scheduler,
        optimizer=instantiate(cfg.training.optimizer,
                              params=[{"ClassEmbedder": class_embedder.parameters()},
                                      {"UNet": unet.parameters()}]
                             ),
        train_ds=train,
        val_ds=val,
        transformations=transformations,
        augmentations=augmentations,
        **cfg.training.train_loop
    )
    # train_dataloader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=train.collate_fn)
    # x, y = next(iter(train_dataloader))
    # print(f"Sample batch x shape: {x.shape}, y shape: {y.shape}")
    # for i in range(x.size(0)):
    #     save_image(x[i], f"sample_image_{i}_label_{y[i].item()}.png")

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
