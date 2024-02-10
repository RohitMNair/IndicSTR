from utils.metrics import (DiacriticAccuracy, FullCharacterAccuracy, CharGrpAccuracy, NED,
                   HalfCharacterAccuracy, CombinedHalfCharAccuracy, WRR, WRR2, ComprihensiveWRR)
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import OneCycleLR
from typing import Tuple, Optional
from torch import Tensor
from data.tokenizer import Tokenizer

import lightning.pytorch.loggers as pl_loggers
import lightning.pytorch as pl
import torch
import torch.nn as nn

class PLMDecoder(pl.LightningModule):
    def __init__(self):
        