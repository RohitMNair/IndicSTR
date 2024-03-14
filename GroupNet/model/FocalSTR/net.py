from .encoder import FocalNetEncoder
from model.commons import HindiBaseSystem, DevanagariBaseSystem
from typing import Tuple
from torch import Tensor

import torch
import torch.nn as nn

class FocalSTR(HindiBaseSystem):
    """
    Group implementation of ViTSTR but instead of ViT, we use FocalNet
    """
    def __init__(self, half_character_classes:list, full_character_classes:list,
                 diacritic_classes:list, halfer:str, embed_dim: int = 96, depths:list= [2, 2, 6, 2],
                 focal_levels:list= [2, 2, 2, 2], focal_windows:list= [3, 3, 3, 3],
                 mlp_ratio: float= 4.0, hidden_dropout_prob: float = 0.0,
                 drop_path_rate:float = 0.1, initializer_range: float = 0.02, 
                 layer_norm_eps: float = 1e-12, image_size: int = 224, patch_size: int = 16, 
                 num_channels: int = 3, threshold:float= 0.5,
                 learning_rate: float= 1e-4, weight_decay: float= 1.0e-4, warmup_pct:float= 0.3):
        self.save_hyperparameters()
        self.embed_dim = embed_dim
        self.hidden_sizes = [self.embed_dim * (2 ** i) for i in range(len(depths))] 
        super().__init__(half_character_classes= half_character_classes, full_character_classes= full_character_classes,
                         diacritic_classes= diacritic_classes, halfer= halfer, max_grps= 16, # FocalNet returns just 16 Vectors per image
                         hidden_size= self.hidden_sizes[-1], threshold= threshold, learning_rate= learning_rate,
                         weight_decay= weight_decay, warmup_pct= warmup_pct)
        self.depths = depths
        self.focal_levels = focal_levels
        self.focal_windows = focal_windows
        self.mlp_ratio = mlp_ratio
        self.hidden_dropout_prob = hidden_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        # non parameteric attributes
        self.encoder = FocalNetEncoder(
            hidden_dropout_prob= self.hidden_dropout_prob, 
            initializer_range = self.initializer_range,
            image_size= self.image_size, 
            patch_size= self.patch_size, 
            num_channels = self.num_channels,
            embed_dim= self.embed_dim,
            hidden_sizes = self.hidden_sizes, 
            depths = self.depths,
            focal_levels= self.focal_levels,
            focal_windows= self.focal_windows,
            mlp_ratio= self.mlp_ratio,
            drop_path_rate= self.drop_path_rate,
            layer_norm_eps= self.layer_norm_eps,
        )

    def forward(self, x:torch.Tensor)-> Tuple[Tensor, Tensor, Tensor, Tensor]:
        enc_x = self.encoder(x)
        h_c_2_logits, h_c_1_logits, f_c_logits, d_logits = self.classifier(enc_x)
        return (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits)

class DevanagariFocalSTR(DevanagariBaseSystem):
    """
    Group implementation of ViTSTR but instead of ViT, we use FocalNet
    """
    def __init__(self, svar:list, vyanjan:list, matras:list, ank:list, chinh:list,
                 nukthas:list, halanth:str, embed_dim: int = 96, depths:list= [2, 2, 6, 2],
                 focal_levels:list= [2, 2, 2, 2], focal_windows:list= [3, 3, 3, 3],
                 mlp_ratio: float= 4.0, hidden_dropout_prob: float = 0.0,
                 drop_path_rate:float = 0.1, initializer_range: float = 0.02, 
                 layer_norm_eps: float = 1e-12, image_size: int = 224, patch_size: int = 16, 
                 num_channels: int = 3, threshold:float= 0.5,
                 learning_rate: float= 1e-4, weight_decay: float= 1.0e-4, warmup_pct:float= 0.3):
        self.save_hyperparameters()
        self.embed_dim = embed_dim
        self.hidden_sizes = [self.embed_dim * (2 ** i) for i in range(len(depths))] 
        super().__init__(svar = svar, vyanjan= vyanjan, matras= matras, ank= ank, chinh= chinh, 
                         nukthas= nukthas, halanth= halanth, max_grps= 16, # FocalNet returns just 16 Vectors per image
                         hidden_size= self.hidden_sizes[-1], threshold= threshold, learning_rate= learning_rate,
                         weight_decay= weight_decay, warmup_pct= warmup_pct)
        self.depths = depths
        self.focal_levels = focal_levels
        self.focal_windows = focal_windows
        self.mlp_ratio = mlp_ratio
        self.hidden_dropout_prob = hidden_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        # non parameteric attributes
        self.encoder = FocalNetEncoder(
            hidden_dropout_prob= self.hidden_dropout_prob, 
            initializer_range = self.initializer_range,
            image_size= self.image_size, 
            patch_size= self.patch_size, 
            num_channels = self.num_channels,
            embed_dim= self.embed_dim,
            hidden_sizes = self.hidden_sizes, 
            depths = self.depths,
            focal_levels= self.focal_levels,
            focal_windows= self.focal_windows,
            mlp_ratio= self.mlp_ratio,
            drop_path_rate= self.drop_path_rate,
            layer_norm_eps= self.layer_norm_eps,
        )

    def forward(self, x:torch.Tensor)-> Tuple[Tensor, Tensor, Tensor, Tensor]:
        enc_x = self.encoder(x)
        h_c_2_logits, h_c_1_logits, f_c_logits, d_logits = self.classifier(enc_x)
        return (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits)