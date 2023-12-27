import torch.nn as nn
import torch
import lightning.pytorch as pl

class CharGroupMHA(pl.LightningModule):
    def __init__(self):
        super().__init__()


        
class GroupDecoder(pl.LightningModule):
    def __init__(self, full_character_embeddings: torch.Tensor, half_character_embeddings: torch.Tensor, 
                 diacritics_embeddigs: torch.Tensor, hidden_size: int= 1024, mlp_ratio: float= 4.0, 
                 layer_norm_eps: float= 1.0e-12, num_encoder_embeddings: int= 64, max_grps: int= 25,
                 num_heads: int= 6, dropout: float= 0.0)-> None:
        super().__init__()
        self.f_c_emb = full_character_embeddings
        self.h_c_emb = half_character_embeddings
        self.d_emb = diacritics_embeddigs
        self.layer_norm_eps = layer_norm_eps
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.num_enc_emb = num_encoder_embeddings
        self.max_grps = max_grps
        self.num_heads = num_heads
        self.dropout = dropout

        self.positional_encodings = nn.Parameter(torch.randn(1, self.max_grps, self.hidden_size))
        self.pos_vis_mha = nn.MultiheadAttention(
                                            embed_dim= hidden_size, 
                                            num_heads= self.num_heads, 
                                            dropout= self.dropout, 
                                            add_bias_kv= True,
                                            batch_first= True,
                                        )
        self.chr_grp_mha = 


    def forward(self, x):
        
    



        

