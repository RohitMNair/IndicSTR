from transformers import ViTConfig, ViTModel
import torch.nn as nn
import torch
import lightning.pytorch as pl

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