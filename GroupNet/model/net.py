from encoder import ViTEncoder
from decoder import GroupDecoder

import lightning.pytorch as pl
import torch
import torch.nn as nn

class GrpClassifier(pl.LightningModule):
    def __init__(self, hidden_size:int, half_character_2_classes:list, half_character_1_classes:list,
                 full_character_classes:list, diacritic_classes:list,):
        super().__init__()
        self.hidden_size= hidden_size
        self.h_c_2_classes = half_character_2_classes
        self.h_c_1_classes = half_character_1_classes
        self.f_c_classes = full_character_classes
        self.d_classes = diacritic_classes
        self.half_character2_head = nn.Linear(
                                in_features = self.hidden_size,
                                out_features = len(self.h_c_2_classes) + 1, # extra node for no half-char
                                bias = True
                            )
        self.half_character1_head = nn.Linear(
                                in_features = self.hidden_size,
                                out_features = len(self.h_c_1_classes) + 1,
                                bias = True
                            )
        self.character_head = nn.Linear(
                                in_features = self.hidden_size,
                                out_features = len(self.f_c_classes) + 1,
                                bias = True
                            )
        self.diacritic_head = nn.Linear( # multi-label classification hence no need for extra head
                                in_features = self.hidden_size,
                                out_features = len(self.d_classes),
                                bias = True
                            )
        
        def forward(self, x:torch.Tensor):
            half_char2_logits = self.half_character2_head(x)
            half_char1_logits = self.half_character1_head(x)
            char_logits = self.character_head(x)
            diac_logits = self.diacritic_head(x)
            return half_char2_logits, half_char1_logits, char_logits, diac_logits 

class GroupNet(pl.LightningModule):
    def __init__(self,  half_character_2_embeddings: torch.Tensor, half_character_1_embeddings: torch.Tensor,
                 full_character_embeddings: torch.Tensor, diacritics_embeddigs: torch.Tensor,
                 half_character_2_classes:list, half_character_1_classes:list,
                 full_character_classes:list, diacritic_classes:list,
                 hidden_size: int = 768, num_hidden_layers: int = 12, num_attention_heads: int = 12,
                 mlp_ratio: float= 4.0, hidden_act: str = "gelu", hidden_dropout_prob: float = 0.0,
                 attention_probs_dropout_prob: float = 0.0, initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12, image_size: int = 224, patch_size: int = 16, 
                 num_channels: int = 3, qkv_bias: bool = True, max_grps: int = 25,
                 ):
        super().__init__()
        self.h_c_2_emb = half_character_2_embeddings
        self.h_c_1_emb = half_character_1_embeddings
        self.f_c_emb = full_character_embeddings
        self.d_emb = diacritics_embeddigs
        self.h_c_2_classes = half_character_2_classes
        self.h_c_1_classes = half_character_1_classes
        self.f_c_classes = full_character_classes
        self.d_classes = diacritic_classes
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.max_grps = max_grps
        self.intermediate_size = int(self.mlp_ratio * self.hidden_size)

        self.encoder = ViTEncoder(
            hidden_size= self.hidden_size,
            num_hidden_layers= self.num_hidden_layers,
            num_attention_heads= self.num_attention_heads,
            intermediate_size= self.intermediate_size,
            hidden_act= self.hidden_act,
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
        )

        self.classifier = GrpClassifier(
            hidden_size= self.hidden_size,
            half_character_2_classes= self.h_c_2_classes,
            half_character_1_classes= self.h_c_1_classes,
            full_character_classes= self.f_c_classes,
            diacritic_classes= self.d_classes,
        )
    
    def forward(self, x:torch.Tensor):
        batch_size = x.shape[0]
        enc_x = self.encoder(x)
        dec_x, pos_vis_attn_weights, chr_grp_attn_weights = self.decoder(enc_x)
        half_char2_logits, half_char1_logits, char_logits, diac_logits = self.classifier(dec_x.view(-1, self.hidden_size))
        return (
            half_char2_logits.view(batch_size, self.max_grps, self.hidden_size),
            half_char1_logits.view(batch_size, self.max_grps, self.hidden_size),
            char_logits.view(batch_size, self.max_grps, self.hidden_size),
            diac_logits.view(batch_size, self.max_grps, self.hidden_size),
        ), (pos_vis_attn_weights, chr_grp_attn_weights)

    def training_step(self, batch, batch_no):
        # batch: img (batch_size, C, H, W), h_c_2 (bs, max_grps) 
        # h_c_1 (bs, max_grps), f_c (bs, max_grps)
        # d_c (bs, max_grps, one hot enc.)
        x, h_c_2, h_c_1, f_c, d_c = batch
        


        
        





