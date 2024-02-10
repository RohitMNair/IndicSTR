from transformers import ViTConfig, ViTModel, FocalNetConfig, FocalNetModel
import torch.nn as nn
import torch
import lightning.pytorch as pl
from typing import Union

class ConvEmbedding(pl.LightningModule):
    """
    Convolutional Embeddings as mentioned in "Early Convolutions Help Transformers See Better by Xiao et al. NeurIPS 2021"
    Equivalent to ViT-b
    """
    
    def __init__(self, num_channels:int = 3) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.net =  nn.Sequential(
            nn.Conv2d(in_channels= self.num_channels, out_channels= 64, kernel_size= 3, stride= 2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(in_channels= 64, out_channels= 128, kernel_size= 3, stride= 2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(in_channels= 128, out_channels= 128, kernel_size= 3, stride = 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(in_channels= 128, out_channels= 256, kernel_size= 3, stride= 2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(in_channels= 256, out_channels= 256, kernel_size= 3, stride= 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(in_channels= 256, out_channels= 512, kernel_size= 3, stride= 2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(in_channels= 512, out_channels= 768, kernel_size= 1, stride= 1) 
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x.flatten(start_dim= 2, end_dim= -1).transpose(dim0=1, dim1=2)

                