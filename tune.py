from ray.train.lightning import (
    RayTrainReportCallback,
    prepare_trainer,
)
from torchvision import transforms
from utils.transforms import RescaleTransform, PadTransform
from lightning.pytorch.callbacks import StochasticWeightAveraging
from omegaconf import DictConfig
from hydra.utils import instantiate
from ray import tune
from ray.train.torch import TorchTrainer
from omegaconf import DictConfig
from hydra.utils import instantiate

import hydra
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')

print(f"Device: {device}")


def train(hparams, cfg):
    """
    Function to train character embeddings
    """
    composed = transforms.Compose([
        transforms.RandomRotation(
            degrees= cfg.transforms.rotation, 
            expand = True, 
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        RescaleTransform(cfg.transforms.img_size),
        PadTransform(cfg.transforms.img_size),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ])
    datamodule = instantiate(cfg.datamodule, transforms = composed)

    model = instantiate(
        cfg.Img2Vec,
        lr = hparams["lr"],
        weight_decay = hparams["weight_decay"],
        rep_dim = hparams["rep_dim"]
        )
    swa = StochasticWeightAveraging(swa_lrs=hparams["swa_lrs"])
    tensorboard = instantiate(cfg.tensorboard_logger)
    trainer = instantiate(
                    cfg.training, 
                    strategy="auto", #DDP is not supported
                    callbacks=[swa],
                    logger = [tensorboard],
                    enable_progress_bar=False,
                )
    trainer.fit(model, datamodule)

@hydra.main(version_base=None, config_path="configs", config_name="tune")
def main(cfg: DictConfig):
    search_space = {
        "lr": tune.loguniform(cfg.search_space.lr.min, cfg.search_space.lr.max),
        "weight_decay": tune.loguniform(cfg.search_space.weight_decay.min, cfg.search_space.weight_decay.max),
        "rep_dim": tune.choice(cfg.search_space.rep_dim),
        "swa_lrs": tune.loguniform(cfg.search_space.swa_lrs.min, cfg.search_space.swa_lrs.max),
    }

    scheduler = instantiate(cfg.scheduler)

    reporter = tune.CLIReporter(
        parameter_columns=[cfg.tune_config.metric],
        metric_columns=[
            'val_acc', 
            'val_loss',
            'val_half_character2_acc',
            'val_half_character1_acc',
            'val_combined_half_character_acc',
            'val_diacritic_acc',
            'val_character_acc'
            ]
    )

    analysis = tune.run(
        tune.with_parameters(train, cfg=cfg),
        metric= cfg.tune_config.metric,
        mode= cfg.tune_config.mode,
        config=search_space,
        resources_per_trial= {
            'cpu': cfg.tune_config.cpu_per_trial,
            'gpu': cfg.tune_config.gpu_per_trial
        }, 
        num_samples=cfg.tune_config.num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
    )

    print('Best hyperparameters found were: ', analysis.best_config)

if __name__ == '__main__':
    main()