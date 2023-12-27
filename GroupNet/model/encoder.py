from transformers import ViTConfig, ViTModel
import torch.nn as nn
import torch
import lightning.pytorch as pl

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
        

class ViTEncoder(pl.LightningModule):
    def __init__(self, hidden_size: int = 768, num_hidden_layers: int = 12, num_attention_heads: int = 12,
                 intermediate_size: int = 3072, hidden_act: str = "gelu", hidden_dropout_prob: float = 0.0,
                 attention_probs_dropout_prob: float = 0.0, initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12, image_size: int = 224, patch_size: int = 16, 
                 num_channels: int = 3, qkv_bias: bool = True, encoder_stride: int = 16) -> None:
        super().__init__()
        self.config =  ViTConfig(
                            hidden_size = hidden_size, 
                            num_hidden_layers = num_hidden_layers, 
                            num_attention_heads = num_attention_heads,
                            intermediate_size = intermediate_size,
                            hidden_act = hidden_act,
                            hidden_dropout_prob = hidden_dropout_prob,
                            attention_probs_dropout_prob = attention_probs_dropout_prob,
                            initializer_range = initializer_range,
                            layer_norm_eps = layer_norm_eps,
                            image_size = image_size,
                            patch_size = patch_size,
                            num_channels = num_channels,
                            qkv_bias = qkv_bias,
                            encoder_stride = encoder_stride,
                            )
        
        self.vit = ViTModel(self.config)

    def forward(self, x):
        return self.vit(x)
    
