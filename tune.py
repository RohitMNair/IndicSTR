from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from torchvision import transforms
from utils.transforms import RescaleTransform, PadTransform
from lightning.pytorch.callbacks import StochasticWeightAveraging
from omegaconf import DictConfig
from hydra.utils import instantiate
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from omegaconf import DictConfig
from hydra.utils import instantiate

import hydra
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')

print(f"Device: {device}")


def train(config):
    """
    Function to train character embeddings
    """
    composed = transforms.Compose([
        transforms.RandomRotation(
            degrees= config["cfg"].transforms.rotation, 
            expand = True, 
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        RescaleTransform(config["cfg"].transforms.img_size),
        PadTransform(config["cfg"].transforms.img_size),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ])
    datamodule = instantiate(config["cfg"].datamodule, transforms = composed)

    model = instantiate(
        config["cfg"].Img2Vec,
        lr = config["lr"],
        weight_decay = config["weight_decay"],
        rep_dim = config["rep_dim"]
        )
    swa = StochasticWeightAveraging(swa_lrs=config["swa_lrs"])

    trainer = instantiate(
                    config["cfg"].training, 
                    strategy=RayDDPStrategy(),
                    callbacks=[RayTrainReportCallback(), swa],
                    plugins=[RayLightningEnvironment()],
                    enable_progress_bar=False,
                )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule)

@hydra.main(version_base=None, config_path="configs", config_name="tune")
def main(cfg: DictConfig):
    search_space = {
        "lr": tune.loguniform(cfg.search_space.lr.min, cfg.search_space.lr.max),
        "weight_decay": tune.loguniform(cfg.search_space.weight_decay.min, cfg.search_space.weight_decay.max),
        "rep_dim": tune.choice(cfg.search_space.rep_dim),
        "swa_lrs": tune.loguniform(cfg.search_space.swa_lrs.min, cfg.search_space.swa_lrs.max),
        "cfg":cfg,
    }

    scheduler = instantiate(cfg.scheduler)

    scaling_config = instantiate(cfg.scaling_config)

    run_config = instantiate(cfg.run_config)

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_loop_per_worker = train,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="val_acc",
            mode="max",
            num_samples= 100,
            scheduler=scheduler,
        ),
    )
    results = tuner.fit()

    print("The best parameters are:", results.get_best_result(metric="val_acc",mode="max"))

if __name__ == '__main__':
    main()