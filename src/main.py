import hydra
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from omegaconf import DictConfig

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    data_module = hydra.utils.instantiate(cfg.dataset.data_module)
    train, val, test, real = data_module.load_datasets(cfg.dataset.val_ratio, cfg.dataset.test_ratio, cfg.dataset.include_real)
    print("Datasets loaded:")
    print(f"Train set size: {len(train)}")
    print(f"Validation set size: {len(val)}")
    print(f"Test set size: {len(test)}")
    print(f"Real set size: {len(real)}")
    train_dataloader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=train.collate_fn)
    x, y = next(iter(train_dataloader))
    print(f"Sample batch x shape: {x.shape}, y shape: {y.shape}")
    for i in range(x.size(0)):
        save_image(x[i], f"sample_image_{i}_label_{y[i].item()}.png")

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
