import torch
import torchvision
import numpy as np
import hydra
import lightning.pytorch as pl
from pathlib import Path
from torchsummary import summary
from torchvision import transforms, utils
from data.transforms import RescaleTransform, PadTransform
from data.module import DevanagariDataModule
from model.ResNet import ResNet
from model.ViT import ViT
from model.GroupEmbed import GrpEmbed
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from omegaconf import DictConfig


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
        RescaleTransform(cfg.model.img_size),
        PadTransform(cfg.model.img_size),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        ])

    datamodule = DevanagariDataModule(
        train_img_dir = cfg.data.train_img_dir,
        train_gt = cfg.data.train_gt,
        val_img_dir = cfg.data.val_img_dir,
        val_gt = cfg.data.val_gt,
        charset= cfg.data.charset,
        diacritics= cfg.data.diacritics,
        halfer= cfg.data.halfer,
        seperator=cfg.data.seperator,
        batch_size = cfg.data.batch_size,
        num_workers = cfg.data.num_workers,
        transforms= composed,
        drop_last = False
        )

    if cfg.model.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam
    elif cfg.model.optimizer.lower() == 'adadelta':
        optimizer = torch.optim.Adadelta
    elif cfg.model.optimizer.lower() == 'adagra':
        optimizer = torch.optim.Adagrad
    elif cfg.model.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD
    elif cfg.model.optimizer.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop
    else:
        raise ModuleNotFoundError("specified Optimizer not found!")
    
    if cfg.model.name.lower() == 'resnet':
        model = ResNet(
                    num_classes= 54, 
                    version= cfg.model.version, 
                    optimizer= optimizer, 
                    lr= cfg.model.lr
                )
    elif cfg.model.name.lower() == 'vit':
        model = ViT(
                    num_classes = 54,
                    version= cfg.model.version,
                    optimizer = optimizer,
                    lr = cfg.model.lr
                )
    elif cfg.model.name.lower() == 'grpembed':
        model = GrpEmbed(
                    character_classes = len(cfg.data.charset), 
                    diacritic_classes = len(cfg.data.diacritics), 
                    half_character_classes = len(cfg.data.half_charset),
                    optimizer = optimizer, 
                    lr= 1e-3, 
                    threshold = 0.5
                )
    else:
        raise ModuleNotFoundError("Specified Model does not exist")

    assert Path(cfg.log.log_dir).exists
    logger = CSVLogger(save_dir = cfg.log.log_dir, name = f"{cfg.model.name}_{str(cfg.model.version)}")

    checkpoint_dir = (
        Path(logger.log_dir)
        / "checkpoints"
    )
    
    checkpoint_callback = ModelCheckpoint(
                            dirpath=checkpoint_dir, 
                            save_top_k=2, 
                            monitor=cfg.training.chkpt_monitor,
                            filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}",
                            mode = cfg.training.chkpt_mode,
                            save_last = True
                        )

    early_stop_callback = EarlyStopping(
                            monitor=cfg.training.stop_monitor, 
                            min_delta=0.01, 
                            patience=4, 
                            verbose=True,
                            check_finite = True,
                            mode=cfg.training.stop_mode
                        )

    swa = StochasticWeightAveraging(swa_lrs=1e-2)

    trainer = pl.Trainer(
        accelerator=cfg.training.device, 
        max_epochs=cfg.training.max_epochs, # set number of epochs
        check_val_every_n_epoch=1,
        gradient_clip_val = cfg.training.gradient_clip_val,
        precision="64-true",
        logger = logger,
        callbacks = [checkpoint_callback, early_stop_callback, swa]
    )
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()


