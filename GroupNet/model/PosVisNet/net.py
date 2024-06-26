from model.FocalSTR.encoder import FocalNetEncoder
from model.ViTSTR.encoder import ViTEncoder
from .decoder import PosVisDecoder
from model.base import DevanagariBaseSystem, HindiBaseSystem
from torch import Tensor
from typing import Tuple
import lightning.pytorch as pl
import torch
import torch.nn as nn

class HindiViTPosVisNet(HindiBaseSystem):
    """
    Encoder Decoder Transformer network that performs Position-Visual attention on 
    visual representations of the encoder (ViT) and decodes characters
    """
    def __init__(self, hidden_size: int = 768, num_hidden_layers: int = 12, num_decoder_layers: int= 1, num_attention_heads: int = 12,
                 mlp_ratio: float= 4.0, hidden_dropout_prob: float = 0.0,
                 attention_probs_dropout_prob: float = 0.0, initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12, image_size: int = 224, patch_size: int = 16, 
                 num_channels: int = 3, qkv_bias: bool = True, max_grps: int = 25, threshold:float= 0.5,
                 learning_rate: float= 1e-4, weight_decay: float= 0.0, warmup_pct:float= 0.3
                 ):
        """
        Constructor for PosVisNet
        Args:
        - hidden_size (int, default= 768): Hidden size of the model
        - num_hidden_layers (int, default= 12): # of hidden layers for the ViT Encoder
        - num_decoder_layers (int, default= 1): # of decoder (PosVisDecoder) layers
        - num_attention_heads (int, default= 12): number of attention heads for the MHA
        - mlp_ratio (float, default= 4.0): ratio of hidden dim. to embedding dim.
        - hidden_dropout_prob (float, default= 0.0): Dropout probability for the Linear layers
        - attention_probs_dropout_prob (float, default= 0.0): Dropout probability for the MHA probability scores
        - initializer_range (float, default= 2.0e-2): The standard deviation of the truncated_normal_initializer
                                                    for initializing all weight matrices.
        - layer_norm_eps (float, default= 1.0e-12): The epsilon used by the layer normalization layers.
        - image_size (int, default= 224): The resolution of the image (Square image)
        - patch_size (int, default= 16): The resolution of the path size used by ViT
        - num_channels (int, default= 3): # of channels in an image
        - qkv_bias (bool, default= True): Whether to add a bias term in the query, key and value projection in MHA
        - max_grps (int, default= 25): Max. # of groups to decode
        - threshold (float, default= 0.5): Probability threshold for classification
        - learning_rate (float, default= 1.0e-4): Learning rate for AdamW optimizer
        - weight_decay (float, default= 0.0): Weight decay coefficient
        - warmup_pct (float, default= 0.3): The percentage of the cycle (in number of steps) 
                                            spent increasing the learning rate for OneCyleLR
        """
        super().__init__(max_grps= max_grps, hidden_size= hidden_size, threshold= threshold, learning_rate= learning_rate,
                         weight_decay= weight_decay, warmup_pct= warmup_pct)
        self.save_hyperparameters()
        self.num_hidden_layers = num_hidden_layers
        self.num_decoder_layers = num_decoder_layers
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
        self.decoder = nn.Sequential(*[PosVisDecoder(
                                            hidden_size= self.hidden_size,
                                            mlp_ratio= self.mlp_ratio,
                                            layer_norm_eps= self.layer_norm_eps,
                                            max_grps= self.max_grps,
                                            num_heads= self.num_attention_heads,
                                            hidden_dropout_prob= self.hidden_dropout_prob,
                                            attention_probs_dropout_prob= self.attention_probs_dropout_prob,
                                            qkv_bias= self.qkv_bias)
                                        for i in range(self.num_decoder_layers)])
    
    def forward(self, x:torch.Tensor)-> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass for ViTPosVisNet
        Args:
        - x (Tensor): Batch of images shape: (BS x C x H x W)

        Returns:
        - Tuple(Tuple(Tensor, Tensor, Tensor, Tensor), Tensor): 1st tuple contains character logits
                                            in order Half-char 2, Half-char 1, Full-char & diacritics
                                            2nd element is the position visual attention scores
        """
        enc_x = self.encoder(x)
        dec_x, pos_vis_attn_weights = self.decoder(enc_x)
        h_c_2_logits, h_c_1_logits, f_c_logits, d_logits = self.classifier(dec_x)
        return (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits)

class HindiFocalPosVisNet(HindiBaseSystem):
    """
    Encoder Decoder Transformer network that performs Position-Visual attention on 
    visual representations of the encoder (ViT) and decodes characters
    """
    def __init__(self, embed_dim: int = 96, depths:list= [2, 2, 6, 2],
                 focal_levels:list= [2, 2, 2, 2], focal_windows:list= [3, 3, 3, 3], 
                 drop_path_rate:float= 0.1, mlp_ratio: float= 4.0, 
                 hidden_dropout_prob: float = 0.0, attention_probs_dropout_prob: float = 0.0, 
                 initializer_range: float = 0.02, layer_norm_eps: float = 1e-12, image_size: int = 224, 
                 patch_size: int = 16, num_channels: int = 3,  num_decoder_layers: int= 1, num_attention_heads:int= 12, 
                 qkv_bias: bool = True, max_grps: int = 25, threshold:float= 0.5,
                 learning_rate: float= 1e-4, weight_decay: float= 0.0, warmup_pct:float= 0.3
                 ):
        """
        Constructor for PosVisNet
        Args:
        - embed_dim (int, default= 96): Dimensionality of patch embedding.
        - depths (list(int), default= [2, 2, 6, 2]): Depth (number of layers) of each stage in the encoder.
        - focal_levels (list(int), default= [2, 2, 2, 2]): Number of focal levels in each layer of the respective stages in the encoder.
        - focal_windows (list(int), default= [3, 3, 3, 3]): Focal window size in each layer of the respective stages in the encoder.
        - num_decoder_layers (int, default= 1): # of decoder (PosVisDecoder) layers
        - num_attention_heads (int, default= 12): number of attention heads for the MHA
        - mlp_ratio (float, default= 4.0): ratio of hidden dim. to embedding dim.
        - hidden_dropout_prob (float, default= 0.0): Dropout probability for the Linear layers
        - attention_probs_dropout_prob (float, default= 0.0): Dropout probability for the MHA probability scores
        - initializer_range (float, default= 2.0e-2): The standard deviation of the truncated_normal_initializer
                                                    for initializing all weight matrices.
        - layer_norm_eps (float, default= 1.0e-12): The epsilon used by the layer normalization layers.
        - image_size (int, default= 224): The resolution of the image (Square image)
        - patch_size (int, default= 16): The resolution of the path size used by ViT
        - num_channels (int, default= 3): # of channels in an image
        - qkv_bias (bool, default= True): Whether to add a bias term in the query, key and value projection in MHA
        - max_grps (int, default= 25): Max. # of groups to decode
        - threshold (float, default= 0.5): Probability threshold for classification
        - learning_rate (float, default= 1.0e-4): Learning rate for AdamW optimizer
        - weight_decay (float, default= 0.0): Weight decay coefficient
        - warmup_pct (float, default= 0.3): The percentage of the cycle (in number of steps) 
                                            spent increasing the learning rate for OneCyleLR
        """
        self.save_hyperparameters()
        self.embed_dim = embed_dim
        self.hidden_sizes = [self.embed_dim * (2 ** i) for i in range(len(depths))]
        super().__init__(max_grps= max_grps, hidden_size= self.hidden_sizes[-1], threshold= threshold, learning_rate= learning_rate,
                         weight_decay= weight_decay, warmup_pct= warmup_pct)
        self.depths = depths
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
        self.num_decoder_layers = num_decoder_layers
        
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
        self.decoder = nn.Sequential(*[PosVisDecoder(
                                            hidden_size= self.hidden_sizes[-1],
                                            mlp_ratio= self.mlp_ratio,
                                            layer_norm_eps= self.layer_norm_eps,
                                            max_grps= self.max_grps,
                                            num_heads= self.num_attention_heads,
                                            hidden_dropout_prob= self.hidden_dropout_prob,
                                            attention_probs_dropout_prob= self.attention_probs_dropout_prob,
                                            qkv_bias= self.qkv_bias)
                                        for i in range(self.num_decoder_layers)])
    
    def forward(self, x:torch.Tensor)-> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass for FocalPosVisNet
        Args:
        - x (Tensor): Batch of images shape: (BS x C x H x W)

        Returns:
        - Tuple(Tuple(Tensor, Tensor, Tensor, Tensor), Tensor): 1st tuple contains character logits
                                            in order Half-char 2, Half-char 1, Full-char & diacritics
                                            2nd element is the position visual attention scores
        """
        x = self.encoder(x)
        x = self.decoder(x)
        h_c_2_logits, h_c_1_logits, f_c_logits, d_logits = self.classifier(x)
        return (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits)