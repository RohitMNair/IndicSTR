from torchvision import transforms
from utils.transforms import RescaleTransform, PadTransform
from lightning.pytorch.callbacks import StochasticWeightAveraging
from omegaconf import DictConfig
from hydra.utils import instantiate
from ray import tune
from ray.train.lightning import RayTrainReportCallback
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
    tune_report_callback = RayTrainReportCallback()
    swa = StochasticWeightAveraging(swa_lrs=hparams["swa_lrs"])
    trainer = instantiate(
                    cfg.training,
                    strategy="auto", #DDP is not supported
                    callbacks=[swa, tune_report_callback],
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

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train, cfg=cfg), 
            {"cpu": cfg.tune_config.cpu_per_trial, "gpu": cfg.tune_config.gpu_per_trial}),
        tune_config= tune.TuneConfig(
            search_alg= instantiate(cfg.tune_config.search_alg),  
            metric= cfg.tune_config.metric,
            mode= cfg.tune_config.mode,
            scheduler=scheduler,
            num_samples=cfg.tune_config.num_samples,
        ),
        run_config= instantiate(cfg.run_config),
        param_space=search_space,
    )
    results = tuner.fit()
    best_results = results.get_best_result(cfg.tune_config.metric,cfg.tune_config.mode)
    print('Best hyperparameters found were: ', best_results.config)

if __name__ == '__main__':
    main()