from model.FocalSTR.encoder import FocalNetEncoder
from model.ViTSTR.encoder import ViTEncoder
from .decoder import GroupDecoder
from model.commons import GrpClassifier
from utils.metrics import (DiacriticAccuracy, FullCharacterAccuracy, CharGrpAccuracy, NED,
                   HalfCharacterAccuracy, CombinedHalfCharAccuracy, WRR, WRR2, ComprihensiveWRR)
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import OneCycleLR
from typing import Tuple
from torch import Tensor
from data.tokenizer import Tokenizer
from model.commons import BaseSystem

import lightning.pytorch.loggers as pl_loggers
import lightning.pytorch as pl
import torch
import torch.nn as nn

class ViTGroupNet(BaseSystem):
    def __init__(self, emb_path:str, half_character_classes:list, full_character_classes:list,
                 diacritic_classes:list, halfer:str, hidden_size: int = 768,
                 num_hidden_layers: int = 12, num_attention_heads: int = 12,
                 mlp_ratio: float= 4.0, hidden_dropout_prob: float = 0.0,
                 attention_probs_dropout_prob: float = 0.0, initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12, image_size: int = 224, patch_size: int = 16, 
                 num_channels: int = 3, qkv_bias: bool = True, max_grps: int = 25, threshold:float= 0.5,
                 learning_rate: float= 1e-4, weight_decay: float= 1.0e-4, warmup_pct:float= 0.3
                 ):
        super().__init__(half_character_classes= half_character_classes, full_character_classes= full_character_classes,
                         diacritic_classes= diacritic_classes, halfer= halfer, max_grps= max_grps,
                         hidden_size= hidden_size, threshold= threshold, learning_rate= learning_rate,
                         weight_decay= weight_decay, warmup_pct= warmup_pct)
        self.save_hyperparameters()
        self.emb_path = emb_path
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

        # non parameteric attributes
        (self.h_c_2_emb, self.h_c_1_emb, 
         self.f_c_emb, self.d_emb) = self._extract_char_embeddings(
                                                                half_character_classes,
                                                                full_character_classes,
                                                                diacritic_classes
                                                            )   
        
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

        self.decoder = GroupDecoder(
            half_character_2_embeddings= self.h_c_2_emb,
            half_character_1_embeddings= self.h_c_1_emb,
            full_character_embeddings= self.f_c_emb,
            diacritics_embeddigs= self.d_emb,
            hidden_size= self.hidden_size,
            mlp_ratio= self.mlp_ratio,
            layer_norm_eps= self.layer_norm_eps,
            max_grps= self.max_grps,
            num_heads= self.num_attention_heads,
            hidden_dropout_prob= self.hidden_dropout_prob,
            attention_probs_dropout_prob= self.attention_probs_dropout_prob,
            qkv_bias= self.qkv_bias
        )

    def _extract_char_embeddings(self, half_character_classes:list, full_character_classes:list,
                 diacritic_classes:list)-> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """
        Extracts the character embeddings from embedding pth file. The pth file must
        contain the following:
        1) embeddings with names h_c_2_emb, h_c_1_emb, f_c_emb, & d_emb
        2) character classes h_c_classes, f_c_classes & d_classes
        Args:
        - 
        Returns:
        - tuple(Tensor, Tensor, Tensor, Tensor, int): half-char 2, half-char 1, full-char
                                                and diacritic embeddings with the dimension
                                                of the embeddings from checkpoint
        """
        loaded_dict = torch.load(self.emb_path, map_location= torch.device(self.device))

        assert tuple(loaded_dict["h_c_classes"]) == tuple(half_character_classes),\
              f"Embedding Half-character classes and model half-character classes do not match {loaded_dict['h_c_classes']} != {half_character_classes}"
        assert tuple(loaded_dict["f_c_classes"]) == tuple(full_character_classes),\
              f"Embedding Full-character classes and model Full-character classes do not match {loaded_dict['f_c_classes']} != {full_character_classes}"
        assert tuple(loaded_dict["d_classes"]) == tuple(diacritic_classes), \
              f"Embedding diacritic classes and model diacritic classes do not match {loaded_dict['d_classes']} != {diacritic_classes}"
        assert loaded_dict["h_c_2_emb"].shape[1] == loaded_dict["h_c_1_emb"].shape[1] \
              == loaded_dict["f_c_emb"].shape[1] == loaded_dict["d_emb"].shape[1], \
                "embedding dimensions do not match"
    
        print(f"The Embedding Dimension is {loaded_dict['h_c_2_emb'].shape[1]}")

        return (
            loaded_dict["h_c_2_emb"],
            loaded_dict["h_c_1_emb"], 
            loaded_dict["f_c_emb"],
            loaded_dict["d_emb"],
        )
    
    def forward(self, x:torch.Tensor)-> Tuple[Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]] :
        enc_x = self.encoder(x)
        dec_x, pos_vis_attn_weights, chr_grp_attn_weights = self.decoder(enc_x)
        h_c_2_logits, h_c_1_logits, f_c_logits, d_logits = self.classifier(dec_x)
        return (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits)
    
class FocalGroupNet(BaseSystem):
    """
    GrpNet with FocalNet as Visual backbone
    """
    def __init__(self, emb_path:str, half_character_classes:list, full_character_classes:list,
                 diacritic_classes:list, halfer:str, embed_dim: int = 96, depths:list= [2, 2, 6, 2], 
                 focal_levels:list= [3, 3, 3, 3], focal_windows:list= [3, 3, 3, 3], 
                 drop_path_rate:float= 0.1, mlp_ratio: float= 4.0, 
                 hidden_dropout_prob: float = 0.0, attention_probs_dropout_prob: float = 0.0, 
                 initializer_range: float = 0.02, layer_norm_eps: float = 1e-12, image_size: int = 224, 
                 patch_size: int = 16, num_channels: int = 3, qkv_bias: bool = True,
                 num_attention_heads:int= 12, max_grps: int = 25, threshold:float= 0.5,
                 learning_rate: float= 1e-4, weight_decay: float= 1.0e-4, warmup_pct:float= 0.3
                 ):
        super().__init__(half_character_classes= half_character_classes, full_character_classes= full_character_classes,
                         diacritic_classes= diacritic_classes, halfer= halfer, max_grps= max_grps,
                         hidden_size= embed_dim * (2 ** (len(depths) - 1)), threshold= threshold,
                         learning_rate= learning_rate, weight_decay= weight_decay, warmup_pct= warmup_pct)
        self.save_hyperparameters()
        self.emb_path = emb_path
        self.embed_dim = embed_dim
        self.depths = depths
        self.hidden_sizes = [self.embed_dim * (2 ** i) for i in range(len(depths))]
        self.focal_levels = focal_levels
        self.focal_windows = focal_windows
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias

        # non parameteric attributes
        (self.h_c_2_emb, self.h_c_1_emb, 
         self.f_c_emb, self.d_emb) = self._extract_char_embeddings(
                                                                half_character_classes,
                                                                full_character_classes,
                                                                diacritic_classes
                                                            )   
        
        self.intermediate_size = int(self.mlp_ratio * self.hidden_sizes[-1])
    
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

        self.decoder = GroupDecoder(
            half_character_2_embeddings= self.h_c_2_emb,
            half_character_1_embeddings= self.h_c_1_emb,
            full_character_embeddings= self.f_c_emb,
            diacritics_embeddigs= self.d_emb,
            hidden_size= self.hidden_sizes[-1],
            mlp_ratio= self.mlp_ratio,
            layer_norm_eps= self.layer_norm_eps,
            max_grps= self.max_grps,
            num_heads= self.num_attention_heads,
            hidden_dropout_prob= self.hidden_dropout_prob,
            attention_probs_dropout_prob= self.attention_probs_dropout_prob,
            qkv_bias= self.qkv_bias
        )

    def _extract_char_embeddings(self, half_character_classes:list, full_character_classes:list,
                 diacritic_classes:list)-> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """
        Extracts the character embeddings from embedding pth file. The pth file must
        contain the following:
        1) embeddings with names h_c_2_emb, h_c_1_emb, f_c_emb, & d_emb
        2) character classes h_c_classes, f_c_classes & d_classes
        Args:
        - 
        Returns:
        - tuple(Tensor, Tensor, Tensor, Tensor, int): half-char 2, half-char 1, full-char
                                                and diacritic embeddings with the dimension
                                                of the embeddings from checkpoint
        """
        loaded_dict = torch.load(self.emb_path, map_location= torch.device(self.device))

        assert tuple(loaded_dict["h_c_classes"]) == tuple(half_character_classes),\
              f"Embedding Half-character classes and model half-character classes do not match {loaded_dict['h_c_classes']} != {half_character_classes}"
        assert tuple(loaded_dict["f_c_classes"]) == tuple(full_character_classes),\
              f"Embedding Full-character classes and model Full-character classes do not match {loaded_dict['f_c_classes']} != {full_character_classes}"
        assert tuple(loaded_dict["d_classes"]) == tuple(diacritic_classes), \
              f"Embedding diacritic classes and model diacritic classes do not match {loaded_dict['d_classes']} != {diacritic_classes}"
        assert loaded_dict["h_c_2_emb"].shape[1] == loaded_dict["h_c_1_emb"].shape[1] \
              == loaded_dict["f_c_emb"].shape[1] == loaded_dict["d_emb"].shape[1], \
                "embedding dimensions do not match"
    
        print(f"The Embedding Dimension is {loaded_dict['h_c_2_emb'].shape[1]}")

        return (
            loaded_dict["h_c_2_emb"],
            loaded_dict["h_c_1_emb"], 
            loaded_dict["f_c_emb"],
            loaded_dict["d_emb"],
        )
    
    def forward(self, x:torch.Tensor)-> Tuple[Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]] :
        enc_x = self.encoder(x) # encoder o/ps are layer norm'd
        dec_x, pos_vis_attn_weights, chr_grp_attn_weights = self.decoder(enc_x)
        h_c_2_logits, h_c_1_logits, f_c_logits, d_logits = self.classifier(dec_x)
        return (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits)