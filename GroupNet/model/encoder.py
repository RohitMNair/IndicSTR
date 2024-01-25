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
        

class ViTEncoder(pl.LightningModule):
    def __init__(self, hidden_size: int = 768, num_hidden_layers: int = 12, num_attention_heads: int = 12,
                 intermediate_size: int = 3072, hidden_act: str = "gelu", hidden_dropout_prob: float = 0.0,
                 attention_probs_dropout_prob: float = 0.0, initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12, image_size: int = 224, patch_size: int = 16, 
                 num_channels: int = 3, qkv_bias: bool = True, encoder_stride:int = 16) -> None:
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
        
        self.vit = ViTModel(self.config, add_pooling_layer= False)

    def forward(self, x):
        return self.vit(x, output_attentions= False, output_hidden_states= False).last_hidden_state

class FocalNetEncoder(pl.LightningModule):
    def __init__(self, hidden_dropout_prob: float = 0.0, initializer_range: float = 0.02,
                 image_size: int = 224, patch_size: int = 16, num_channels: int = 3,
                 embed_dim:int = 96, hidden_sizes:list = [192, 384, 768, 768], 
                 depths:list = [2, 2, 6, 2], focal_levels:list = [2, 2, 2, 2],
                 focal_windows:list = [3, 3, 3, 3], mlp_ratio:float = 4.0,
                 drop_path_rate:float = 0.1, layer_norm_eps:float = 1e-5):
                
        self.config = FocalNetConfig(
                            image_size = image_size,
                            patch_size = patch_size,
                            num_channels = num_channels,
                            embed_dim = embed_dim,
                            use_conv_embed = True,
                            hidden_sizes = hidden_sizes,
                            depths = depths,
                            focal_levels = focal_levels,
                            focal_windows = focal_windows,
                            hidden_act = "gelu",
                            mlp_ratio = mlp_ratio,
                            hidden_dropout_prob = hidden_dropout_prob,
                            drop_path_rate = drop_path_rate,
                            use_layerscale = False,
                            use_post_layernorm = False,
                            use_post_layernorm_in_modulation = False,
                            normalize_modulator = False,
                            initializer_range =  initializer_range,
                            layer_norm_eps = layer_norm_eps,
                            encoder_stride = 32,
                        )
        self.model = FocalNetModel(self.config, add_pooling_layer= False)

    def forward(self, x):
        return self.model(x, return_dict = True).last_hidden_state
                