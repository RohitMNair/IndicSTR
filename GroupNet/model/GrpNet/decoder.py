import torch.nn as nn
import torch
import lightning.pytorch as pl

from torch import Tensor

class CharGroupMHA(pl.LightningModule):
    """
    Implements Character Group attention Block, The queries are the visual embeddings taken from
    the encoder with keys and values being the embeddings taken from the Img2Vec model. The attention
    mechanism basically identifies the most likely character based on the similarity between the 
    visual embeddings and the character embeddings.
    """
    def __init__(self, hidden_size: int, full_character_embeddings:Tensor,
                 half_character_2_embeddings:Tensor, half_character_1_embeddings:Tensor, 
                 diacritics_embeddigs:Tensor, dropout:float= 0.0, qkv_bias:bool = True)->None:
        """
        Constructor for Character Group Attention block
        Args:
        - hidden_size (int): the hidden size for the projection layers before the attention mechanism
        - full_character_embeddings (Tensor): full character embeddings shape: (# of full-chars x embed. dim.)
        - half_character_2_embeddings (Tensor): half character 2 embeddings shape: (# of half-chars x embed. dim.)
        - half_character_1_embeddings (Tensor): half character 1 embeddings shape: (# of half-chars x embed. dim.)
        - diacritics_embeddigs (Tensor): diacritic embeddings shape: (# of diacritics x embed. dim.)
        - dropout (float): Dropout probability for attention mechanism (dropout applied to the attention scores)
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.qkv_bias = qkv_bias
        self.f_c_emb = full_character_embeddings
        self.h_c_1_emb = half_character_1_embeddings
        self.h_c_2_emb = half_character_2_embeddings
        self.d_emb = diacritics_embeddigs
        assert self.f_c_emb.shape[1] == self.h_c_2_emb.shape[1] == self.h_c_1_emb.shape[1] == self.d_emb.shape[1],\
            "The embedding dimensions do not match"
        self.char_embed_dim = self.f_c_emb.shape[1]
        self.projection_size = self.hidden_size // 4

        # half character 2
        self.query_proj_h_c_2 = nn.Linear(self.hidden_size, self.projection_size, bias = self.qkv_bias)
        self.key_proj_h_c_2 = nn.Linear(self.char_embed_dim, self.projection_size, bias = self.qkv_bias)
        self.value_proj_h_c_2 = nn.Linear(self.char_embed_dim, self.projection_size, bias = self.qkv_bias)
        
        # half character 1
        self.query_proj_h_c_1 = nn.Linear(self.hidden_size, self.projection_size, bias = self.qkv_bias)
        self.key_proj_h_c_1 = nn.Linear(self.char_embed_dim, self.projection_size, bias= self.qkv_bias)
        self.value_proj_h_c_1 = nn.Linear(self.char_embed_dim, self.projection_size, bias= self.qkv_bias)

        # full character
        self.query_proj_f_c = nn.Linear(self.hidden_size, self.projection_size, bias = self.qkv_bias)
        self.key_proj_f_c = nn.Linear(self.char_embed_dim, self.projection_size , bias= self.qkv_bias)
        self.value_proj_f_c = nn.Linear(self.char_embed_dim, self.projection_size , bias= self.qkv_bias)
        
        # diacritic
        self.query_proj_d = nn.Linear(self.hidden_size, self.projection_size, bias = self.qkv_bias)
        self.key_proj_d = nn.Linear(self.char_embed_dim, self.projection_size, bias = self.qkv_bias)
        self.value_proj_d = nn.Linear(self.char_embed_dim, self.projection_size, bias = self.qkv_bias)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias= self.qkv_bias)


    def scaled_dot_product_attention(self, query: Tensor, key: Tensor, value: Tensor)-> tuple:
        """
        Performs scaled Dot product QKV attention
        Args:
        - query (Tensor)
        - key (Tensor)
        - value (Tensor)

        Returns:
        - Tensor: attention output shape (# of Queries x hidden_size)
        - Tensor: attention scores shape (# of Queries x # of Keys)
        """
        scores = torch.matmul(query, key.transpose(-2, -1))/ torch.sqrt(torch.tensor(self.projection_size, dtype= torch.float32))
        
        attention_probs = nn.functional.softmax(scores, dim= -1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)

        return context, attention_probs

    def forward(self, query: Tensor)-> tuple:
        """
        Performs Character Group attention Mechanism
        Args:
        - query (Tensor) shape (max_grps x hidden_size)

        Returns:
        - Tensor: output of the attention mechanism
        - tuple (Tensor, Tensor, Tensor, Tensor): attention scores for half-char-2, half-char-1, full-char and diacritic heads 
        """
        # move the embeddings to the correct device
        self.h_c_2_emb = self.h_c_2_emb.to(self.device)
        self.h_c_1_emb = self.h_c_1_emb.to(self.device)
        self.f_c_emb = self.f_c_emb.to(self.device)
        self.d_emb = self.d_emb.to(self.device)

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
        context = torch.concat((attention_h_c_2, attention_h_c_1, attention_f_c, attention_d), dim= -1)
        context = self.out_proj(context)
        return context, (attention_probs_h_c_2, attention_probs_h_c_1, attention_probs_f_c, attention_probs_d)
    
class GroupDecoder(pl.LightningModule):
    """
    Group Decoder which decodes characters based on similarity with the group embeddings taken from the Img2Vec model
    """
    def __init__(self, half_character_2_embeddings:Tensor, half_character_1_embeddings:Tensor,
                 full_character_embeddings:Tensor, diacritics_embeddigs:Tensor, hidden_size:int= 1024, 
                 mlp_ratio:float= 4.0, layer_norm_eps:float= 1.0e-12, max_grps:int= 25, qkv_bias:bool = True,
                 num_heads:int= 6, hidden_dropout_prob:float= 0.0, attention_probs_dropout_prob:float= 0.0)-> None:
        """
        Constructor for GroupDecoder
        Args:
        - half_character_2_embeddings (Tensor): vector embeddings for the half-character 2 shape: (# of half-chars x embedding dim)
        - half_character_1_embeddings (Tensor): embeddings for half-characters 1 shape: (# of half-chars x embedding dim)
        - full_character_embeddings (Tensor): embeddings for full-characters shape: (# of full-chars x embedding dim)
        - diacritics_embeddigs (Tensor): embeddings for diacritics-characters shape: (# of diacritic-chars x embedding dim)
        - hidden_size (int, default= 1024): sizes of linear layers in the Multi-head attention blocks
        - mlp_ratio (float, default= 4.0): size of the hidden_layer to intermediate layer in MLP block 
        - layer_norm_eps (float, default= 1.0e-12): layer norm epsilon
        - max_grps (int, default= 25): Maximum groups the decoder will need to predict
        - num_heads (int, default= 6): # of heads in the position-visual MHA block
        - hidden_dropout_prob (float, default= 0.0): dropout probability for the hidden linear layers
        - attention_probs_dropout_prob (float, defualt= 0.0): dropout probability for MHA Blocks
        """
        super().__init__()
        self.h_c_2_emb = half_character_2_embeddings
        self.h_c_1_emb = half_character_1_embeddings
        self.f_c_emb = full_character_embeddings
        self.d_emb = diacritics_embeddigs
        self.layer_norm_eps = layer_norm_eps
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.max_grps = max_grps
        self.qkv_bias = qkv_bias
        self.num_heads = num_heads
        self.intermediate_size = int(self.mlp_ratio * self.hidden_size)
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob

        self.positional_encodings = nn.Parameter(torch.randn(self.max_grps, self.hidden_size))
        self.hidden_dropout1 = nn.Dropout(self.hidden_dropout_prob)
        self.pos_norm = nn.LayerNorm(normalized_shape= self.hidden_size, eps=self.layer_norm_eps)
        self.pos_vis_mha = nn.MultiheadAttention(
                                            embed_dim= hidden_size, 
                                            num_heads= self.num_heads, 
                                            dropout= self.attention_probs_dropout_prob, 
                                            add_bias_kv= self.qkv_bias,
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
                                        qkv_bias = self.qkv_bias
                                    )

        self.hidden_dropout3 = nn.Dropout(self.hidden_dropout_prob)
        self.mlp_norm = nn.LayerNorm(normalized_shape= self.hidden_size, eps= self.layer_norm_eps)
        self.mlp_1 = nn.Linear(self.hidden_size, self.intermediate_size, bias = True)
        self.act = nn.GELU()
        self.hidden_dropout4 = nn.Dropout(self.hidden_dropout_prob)
        self.mlp_2 = nn.Linear(self.intermediate_size, self.hidden_size, bias = True)
        self.hidden_dropout5 = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, x:Tensor)->tuple:
        """
        Performs Position-Visual attention for identifying the positions of the character groups
        followed by Character Group attention which identifies the groups present.
        Args:
        - x (Tensor): Vector patch representations from a Vision encoder; shape: (N x encoder embed. dim.)

        Returns:
        - tuple: output patch representations shape:
        """
        x = self.hidden_dropout1(x)
        # ViT outputs are layer norm'd
        x_1, pos_vis_attn_weights = self.pos_vis_mha( # expand positional enc. along the batch dimension
                                                query= self.pos_norm(self.positional_encodings.expand(x.shape[0],-1,-1)),
                                                key= x,
                                                value= x,
                                                need_weights = True,
                                            )
        x = self.positional_encodings + self.hidden_dropout2(x_1)

        x_1, chr_grp_attn_weights = self.chr_grp_mha(self.q_norm(x))
        x = x + self.hidden_dropout3(x_1)

        x_1 = self.mlp_2(self.hidden_dropout4(self.act(self.mlp_1(self.mlp_norm(x)))))
        x = x + self.hidden_dropout5(x_1)
        
        return x_1, pos_vis_attn_weights, chr_grp_attn_weights
