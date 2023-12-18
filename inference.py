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
    Function to check predictions of character embedding model
    """
    columns = ["half_char2", "half_char1", "char", "diac"]
    predictions = pd.DataFrame(columns=columns)
    rows = []
    for b_no in range(batch.shape[0]): # iterate over batches
        indices = torch.nonzero(diac_preds_binary[b_no], as_tuple=False)
        diacs = ""
        for i in indices:
            diacs += self.diacritic_classes[i] + "+ "

        rows.append(
            {
                "half_char2" : self.half_character_classes[half_char2_preds[b_no]],
                "half_char1" : self.half_character_classes[half_char1_preds[b_no]],
                "char" : self.character_classes[char_preds[b_no]],
                "diac": diacs 
            }
        )

    for row in rows:
        predictions = predictions.append(row, ignore_index=True)