from .encoder import ViTEncoder
from model.commons import HindiBaseSystem

from torch.optim.lr_scheduler import OneCycleLR
from typing import Tuple
from torch import Tensor

import lightning.pytorch.loggers as pl_loggers
import lightning.pytorch as pl
import torch
import torch.nn as nn

class ViTSTR(HindiBaseSystem):
    """
    Group implementation of ViTSTR
    """
    def __init__(self, half_character_classes:list, full_character_classes:list,
                 diacritic_classes:list, halfer:str, hidden_size: int = 768,
                 num_hidden_layers: int = 12, num_attention_heads: int = 12,
                 mlp_ratio: float= 4.0, hidden_dropout_prob: float = 0.0,
                 attention_probs_dropout_prob: float = 0.0, initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12, image_size: int = 224, patch_size: int = 16, 
                 num_channels: int = 3, qkv_bias: bool = True, threshold:float= 0.5,
                 learning_rate: float= 1e-4, weight_decay: float= 1.0e-4, warmup_pct:float= 0.3):
        
        max_grps = (image_size // patch_size)**2 + 1
        super().__init__(half_character_classes = half_character_classes, full_character_classes= full_character_classes,
                         diacritic_classes= diacritic_classes, halfer= halfer, max_grps= max_grps, hidden_size= hidden_size,
                         threshold= threshold, learning_rate= learning_rate, weight_decay= weight_decay,
                         warmup_pct= warmup_pct)
        self.save_hyperparameters()
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        
        self.intermediate_size = int(self.mlp_ratio * self.hidden_size)
        
        self.encoder = ViTEncoder(
            hidden_size= self.hidden_size,
            num_hidden_layers= self.num_hidden_layers,
            num_attention_heads= self.num_attention_heads,
            intermediate_size= self.intermediate_size,
            hidden_act= "gelu",
            hidden_dropout_prob= self.hidden_dropout_prob,
            attention_probs_dropout_prob= self.attention_probs_dropout_prob,
            initializer_range= self.initializer_range,
            layer_norm_eps= self.layer_norm_eps,
            image_size= self.image_size,
            patch_size= self.patch_size,
            num_channels= self.num_channels,
            qkv_bias= self.qkv_bias,
        )

    def forward(self, x:torch.Tensor)-> Tuple[Tensor, Tensor, Tensor, Tensor]:
        enc_x = self.encoder(x)
        h_c_2_logits, h_c_1_logits, f_c_logits, d_logits = self.classifier(enc_x)
        return (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits)
    