import torch
import torchvision
import numpy as np
import hydra
import os
import lightning.pytorch as pl
from pathlib import Path
from torchvision import transforms, utils
from utils.transforms import RescaleTransform, PadTransform
from data.module import DevanagariDataModule
from model.ResNet import ResNet
from model.ViT import ViT
from model.Img2Vec import Img2Vec
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from omegaconf import DictConfig
from hydra.utils import instantiate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')

print(f"Device: {device}")


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    """
    Function to train character embeddings
    """
    assert isinstance(cfg.model.img_size, int)
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

    model = instantiate(cfg.Img2Vec)

    csv_logger = instantiate(cfg.csv_logger)
    tensorboard_logger = instantiate(cfg.tensorboard_logger)

    checkpoint_callback = instantiate(cfg.model_checkpoint)

    early_stopping_callback = instantiate(cfg.early_stopping)

    swa = StochasticWeightAveraging(swa_lrs=1e-2)

    trainer = instantiate(
                    cfg.training, 
                    callbacks = [checkpoint_callback, early_stopping_callback, swa],
                    logger = [csv_logger, tensorboard_logger]
                )
    trainer.fit(model, datamodule, ckpt_path = cfg.ckpt_path)

if __name__ == "__main__":
    main()