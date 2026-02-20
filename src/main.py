#pylint: disable=E0401
import hydra
from hydra.utils import instantiate
from sklearn import metrics
from diffusers import AutoencoderKL
import torch
from torchvision.transforms import v2
import wandb
from omegaconf import DictConfig, OmegaConf\

from src.train import ParameterGroupNames

def get_metric_yaml(fun: str, threshold: float = None, switch_args: bool = False,
                    is_wandb: bool = False, kwargs: dict = None):
    fun = f"sklearn.metrics.{fun}" if hasattr(metrics, fun) else fun
    kwargs = kwargs if kwargs is not None else {}
    yaml = {
        "_target_": "src.utils.metric_wrapper",
        "_partial_": True,
        "threshold": threshold,
        "switch_args": switch_args,
        "is_wandb": is_wandb,
        "metric_fun": {
            "_target_": fun,
            "_partial_": True,
            **kwargs
        }
    }
    return OmegaConf.create(yaml)

# List enum values here
OmegaConf.register_new_resolver("interpolation", lambda name: v2.InterpolationMode[name])
OmegaConf.register_new_resolver("param_group_name", lambda name: ParameterGroupNames[name])
OmegaConf.register_new_resolver("metric", get_metric_yaml)

@hydra.main(config_path="../conf2", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    # TODO add tags to init e.g. diffusion vs vae vs cf_classifier
    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(project = cfg.project_name, config = wandb_config)
    data_module = instantiate(cfg.data.data_module)
    datasets = data_module.load_datasets()
    train, val, test, real = datasets
    train_sampler = data_module.get_sampler(train, oversample = cfg.data.oversample)
    val_sampler = data_module.get_sampler(val, oversample = False,
                                          replacement = False, use_generator = True)
    print("Datasets loaded:")
    print(f"Train set size: {len(train)}")
    print(f"Validation set size: {len(val)}")
    print(f"Test set size: {len(test)}")
    print(f"Real set size: {len(real)}")
    trainer = instantiate(cfg.model.trainer)
    trainer.train(train, val, train_sampler, val_sampler)

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter
