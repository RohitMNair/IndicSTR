import torch.nn as nn
import torch
import lightning.pytorch as pl

from typing import Union

class CharGroupMHA(pl.LightningModule):
    def __init__(self, hidden_size: int, full_character_embeddings:torch.Tensor,
                 half_character_1_embeddings:torch.Tensor, half_character_2_embeddings:torch.Tensor,
                 diacritics_embeddigs:torch.Tensor, dropout:float= 0.0)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.f_c_emb = full_character_embeddings
        self.h_c_1_emb = half_character_1_embeddings
        self.h_c_2_emb = half_character_2_embeddings
        self.d_emb = diacritics_embeddigs
        self.projection_size = self.hidden_size // 4

        # half character 2
        self.query_proj_h_c_2 = nn.Linear(self.hidden_size, self.projection_size, bias = True)
        self.key_proj_h_c_2 = nn.Linear(self.hidden_size, self.projection_size, bias = True)
        self.value_proj_h_c_2 = nn.Linear(self.hidden_size, self.projection_size, bias = True)
        
        # half character 1
        self.query_proj_h_c_1 = nn.Linear(self.hidden_size, self.projection_size, bias = True)
        self.key_proj_h_c_1 = nn.Linear(self.hidden_size, self.projection_size, bias= True)
        self.value_proj_h_c_1 = nn.Linear(self.hidden_size, self.projection_size, bias= True)

        # full character
        self.query_proj_f_c = nn.Linear(self.hidden_size, self.projection_size, bias = True)
        self.key_proj_f_c = nn.Linear(self.hidden_size, self.projection_size , bias= True)
        self.value_proj_f_c = nn.Linear(self.hidden_size, self.projection_size , bias= True)
        
        # diacritic
        self.query_proj_d = nn.Linear(self.hidden_size, self.projection_size, bias = True)
        self.key_proj_d = nn.Linear(self.hidden_size, self.projection_size, bias = True)
        self.value_proj_d = nn.Linear(self.hidden_size, self.projection_size, bias = True)

        self.out_proj = nn.Linear(self.projection_size, self.hidden_size, bias= True)


    def scaled_dot_product_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor)-> tuple:
        scores = torch.matmul(query, key.transpose(-2, -1))/ torch.sqrt(torch.tensor(self.hidden_size, dtype= torch.float32))
        
        attention_probs = nn.functional.softmax(scores, dim= -1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)

        return context, attention_probs

    def forward(self, query: torch.Tensor)-> tuple:
        attention_h_c_2, attention_probs_h_c_2 = self.scaled_dot_product_attention(
                                                                    query= self.query_proj_h_c_2(query),
                                                                    key= self.key_proj_h_c_2(self.h_c_2_emb),
                                                                    value= self.value_proj_h_c_2(self.h_c_2_emb),
                                                                    )
        attention_h_c_1, attention_probs_h_c_1 = self.scaled_dot_product_attention(
                                                                    query= self.query_proj_h_c_1(query), 
                                                                    key= self.key_proj_h_c_1(self.h_c_1_emb),
                                                                    value= self.value_proj_h_c_1(self.h_c_1_emb),                                                                  
                                                                    )
        attention_f_c, attention_probs_f_c = self.scaled_dot_product_attention(
                                                                    query= self.query_proj_f_c(query),
                                                                    key= self.key_proj_f_c(self.f_c_emb),
                                                                    value= self.value_proj_f_c(self.f_c_emb),
                                                                )
        attention_d, attention_probs_d = self.scaled_dot_product_attention(
                                                                    query= self.query_proj_d(query),
                                                                    key= self.key_proj_d(self.d_emb),
                                                                    value= self.value_proj_d(self.d_emb),
                                                                    )
        attention = torch.concat((attention_h_c_2, attention_h_c_1, attention_f_c, attention_d), dim= -1)
        attention = self.out_proj(attention)
        return attention, attention_probs_h_c_2, attention_probs_h_c_1, attention_probs_f_c, attention_probs_d
        
class GroupDecoder(pl.LightningModule):
    def __init__(self, half_character_2_embeddings:torch.Tensor, half_character_1_embeddings:torch.Tensor,
                 full_character_embeddings:torch.Tensor, diacritics_embeddigs:torch.Tensor, 
                 hidden_size:int= 1024, mlp_ratio:float= 4.0, layer_norm_eps:float= 1.0e-12, max_grps:int= 25,
                 num_heads:int= 6, hidden_dropout_prob:float= 0.0, attention_probs_dropout_prob:float= 0.0)-> None:
        super().__init__()
        self.h_c_2_emb = half_character_2_embeddings
        self.h_c_1_emb = half_character_1_embeddings
        self.f_c_emb = full_character_embeddings
        self.d_emb = diacritics_embeddigs
        self.layer_norm_eps = layer_norm_eps
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.max_grps = max_grps
        self.num_heads = num_heads
        self.intermediate_size = int(self.mlp_ratio * self.hidden_size)
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob

        self.positional_encodings = nn.Parameter(torch.randn(1, self.max_grps, self.hidden_size))
        self.hidden_dropout1 = nn.Dropout(self.hidden_dropout_prob)
        self.pos_norm = nn.LayerNorm(normalized_shape= self.hidden_size, eps=self.layer_norm_eps)
        self.pos_vis_mha = nn.MultiheadAttention(
                                            embed_dim= hidden_size, 
                                            num_heads= self.num_heads, 
                                            dropout= self.attention_probs_dropout_prob, 
                                            add_bias_kv= True,
                                            batch_first= True,
                                        )
        self.hidden_dropout2 = nn.Dropout(self.hidden_dropout_prob)
        self.q_norm = nn.LayerNorm(normalized_shape= self.hidden_size, eps=self.layer_norm_eps)
        self.chr_grp_mha = CharGroupMHA(
                                        hidden_size = self.hidden_size, 
                                        full_character_embeddings= self.f_c_emb,
                                        half_character_1_embeddings= self.h_c_1_emb,
                                        half_character_2_embeddings= self.h_c_2_emb,
                                        diacritics_embeddigs= self.d_emb,
                                        dropout = self.attention_probs_dropout_prob,
                                    )
        self.hidden_dropout3 = nn.Dropout(self.hidden_dropout_prob)
        self.mlp_norm = nn.LayerNorm(normalized_shape= self.hidden_size, eps= self.layer_norm_eps)
        self.mlp_1 = nn.Linear(self.hidden_size, self.intermediate_size, bias = True)
        self.act = nn.GELU()
        self.hidden_dropout4 = nn.Dropout(self.hidden_dropout_prob)
        self.mlp_2 = nn.Linear(self.intermediate_size, self.hidden_size, bias = True)
        self.hidden_dropout5 = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, x:torch.Tensor)->tuple:
        x = self.hidden_dropout1(x + self.positional_encodings)
        x_1, pos_vis_attn_weights = self.pos_vis_mha(
                                                query= self.pos_norm(self.positional_encodings),
                                                key= x,
                                                value= x,
                                                need_weights = True,
                                            )
        x = self.positional_encodings + self.hidden_dropout2(x_1)

        x_1, chr_grp_attn_weights = self.chr_grp_mha(self.q_norm(x))
        x = x + self.hidden_dropout3(x_1)

        x_1 = self.mlp_2(self.hidden_dropout4(self.act(self.mlp_1(self.mlp_norm(x)))))
        x = x + self.hidden_dropout5(x_1)
        
        return x, pos_vis_attn_weights, chr_grp_attn_weights

