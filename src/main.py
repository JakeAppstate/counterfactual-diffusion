import hydra

from omegaconf import DictConfig, OmegaConf
from src.data import DataModule

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    data_module = hydra.utils.instantiate(cfg.dataset.data_module)
    train, val, test, real = data_module.load_datasets(cfg.val_ratio, cfg.test_ratio)
    print("Datasets loaded:")
    print(f"Train set size: {len(train)}")
    print(f"Validation set size: {len(val)}")
    print(f"Test set size: {len(test)}")
    print(f"Real set size: {len(real)}")

if __name__ == "__main__":
    main() # pylint: disable=E1120
