from torch import Tensor
from typing import Optional, Sequence, Union
from model.head import get_activation
import lightning.pytorch as pl
import torch.nn as nn
import torch
import math

class DecoderLayer(pl.LightningModule):
    """
    A Transformer decoder layer supporting two-stream attention (XLNet)
    This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch.
    """
    def __init__(self, d_model:int, nhead_self_attn:Sequence[Union[int,Sequence[int]]], nhead_cross_attn:int,
                 dim_feedforward:int =2048, dropout:float =0.1, activation:str ='GELU', layer_norm_eps:float =1e-5, 
                 num_h_c:int = 2, num_d_c:int = 2):
        """
        constructor for DecoderLayer
        Args:
        - d_model (int): dimension of the decoder feature vectors (must be divisible by 3)
        - nhead_self_attn (Sequence[Union[int,Sequence[int]]]): number of multi-head attention heads for each character category
                            -- index-0: Sequence[int] for nheads of half-character
                            -- index-1: int for nheads for full-character
                            -- index-2: Sequnce[int] for nheads of diacritic-character
        - nhead_cross_attn (int): number of attention heads for cross attention
        - dim_feedforwad (int, default= 2048): dimension of the intermediate Dense or Linear layer in the MLP
        - dropout (float, default= 0.1): dropout probability
        - activation (str, default= 'GELU'): string name of activation as specified in torch.nn
        - layer_norm_eps (float, default= 1e-5): value of epsilon in layer norm
        - num_h_c (int, default= 2): number of halc-character classes
        - num_d_c (int, default= 2): number of diacritic classes

        # Note: the specified heads must evenly divide d_model/3 and for each Sequence element is the
        nhead, the Sequence elements must evenly divide d_model/(3 * len(seq))
        eg: [[2,2], 4, [2,2]] here 2 heads for each half_character, 4 heads for full character 
            and 2 heads for each diacritic character. if the d_model is 768 ( 768 % 3 == 0)
            and for half-char-1 it will be 2 heads with 64 dim each ((768/(3 * 2)) % 2 == 0) similarly for half char 2. 
            Full character will have 4 heads with 64 dim. each. Diacritic will be same as half-char
        """
        super().__init__()
        self.d_model = d_model
        self.nhead_self_attn = nhead_self_attn
        self.nhead_cross_attn = nhead_cross_attn
        self.dim_feedforward = dim_feedforward
        self.activation = activation
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_h_c = num_h_c
        self.num_d_c = num_d_c
        # 0th index will correspond to h_c_n and ith index will correspond to h_c_(n-i)
        self.self_attn_h_c = [nn.MultiheadAttention(d_model, nhead_self_attn[0][i], 
                                                    dropout= dropout, batch_first= True) for i in range(num_h_c)]
        self.self_attn_f_c = nn.MultiheadAttention(d_model, nhead_self_attn[1], dropout=dropout, batch_first=True)
        self.self_attn_d_c = [nn.MultiheadAttention(d_model, nhead_self_attn[2][i],
                                                    dropout=dropout, batch_first=True) for i in range(num_d_c)]

        # for merging representations
        self.norm_merge = nn.LayerNorm(d_model*(num_h_c + 1 + num_d_c), eps= layer_norm_eps)
        self.merge1 = nn.Linear(d_model*(num_h_c + 1 + num_d_c), dim_feedforward)
        self.activation_merge = get_activation(activation)()
        self.dropout_merge1 = nn.Dropout(dropout)
        self.merge2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_merge2 = nn.Dropout(dropout)

        # cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead_cross_attn, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_ctx_h_c = [nn.LayerNorm(d_model, eps=layer_norm_eps) for _ in range(num_h_c)]
        self.norm_ctx_f_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_ctx_d_c = [nn.LayerNorm(d_model, eps=layer_norm_eps) for _ in range(num_d_c)]

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation(activation)()

    def forward_stream(self, query:Tensor, query_norm:Tensor, h_c_kv:Sequence[Tensor], f_c_kv:Tensor, d_c_kv:Sequence[Tensor],
                       memory:Tensor, query_mask:Optional[Tensor], key_padding_mask:Optional[Tensor]):
        """
        Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.
        """
        # self-attn
        sa_h_c, sa_weights_h_c = [], []
        for self_attn, h_c in zip(self.self_attn_h_c, h_c_kv):
            sa_, w_= self_attn(query= query_norm, key= h_c, value= h_c, attn_mask= query_mask, 
                               key_padding_mask= key_padding_mask)
            sa_h_c.append(sa_)
            sa_weights_h_c.append(w_)
        sa_f_c, sa_weights_f_c = self.self_attn_f_c(query= query_norm, key= f_c_kv, value= f_c_kv, 
                                                    attn_mask=query_mask, key_padding_mask=key_padding_mask)
        sa_d_c, sa_weights_d_c = [],[]
        for self_attn, d_c in zip(self.self_attn_d_c, d_c_kv):
            sa_, w_ = self_attn(query= query_norm, key= d_c, value= d_c,
                                         attn_mask=query_mask, key_padding_mask=key_padding_mask)
            sa_d_c.append(sa_)
            sa_weights_d_c.append(w_)
        
        # Merge
        sa = torch.cat([*sa_h_c, sa_f_c, *sa_d_c], dim= -1)
        sa_weights = (tuple(sa_weights_h_c), sa_weights_f_c, sa_weights_d_c)
        sa = self.merge2(self.dropout_merge1(self.activation_merge(self.merge1(self.norm_merge(sa)))))
        query = query + self.dropout_merge2(sa)
        
        # cross attn
        ca, ca_weights = self.cross_attn(query= self.norm1(query), key= memory, value= memory)
        query = query + self.dropout2(ca)

        # MLP
        out = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(query)))))
        query = query + self.dropout3(out)
        return query, sa_weights, ca_weights

    def forward(self, query:Tensor, context_h_c:Sequence[Tensor], context_f_c:Tensor, context_d_c:Sequence[Tensor], 
                memory:Tensor, query_mask: Optional[Tensor] = None, context_key_padding_mask: Optional[Tensor] = None):
        query_norm = self.norm_q(query)
        context_h_c_norm = [self.norm_ctx_h_c[i](context_h_c_i) for i, context_h_c_i in enumerate(context_h_c)]
        context_f_c_norm = self.norm_ctx_f_c(context_f_c)
        context_d_c_norm = [self.norm_ctx_d_c[i](context_d_c_i) for i, context_d_c_i in enumerate(context_d_c)]
        query = self.forward_stream(query, query_norm, context_h_c_norm, context_f_c_norm, context_d_c_norm,
                                    memory, query_mask, context_key_padding_mask)[0]
        return query

class Decoder(pl.LightningModule):
    __constants__ = ['norm']

    def __init__(self, d_model:int,nhead_self_attn:Sequence[Union[int,Sequence[int]]], nhead_cross_attn:int, 
                 dim_feedforward= 2048, dropout= 0.1, activation='GELU', 
                 layer_norm_eps=1e-5, num_h_c:int = 2, num_d_c:int = 2, num_layers:int = 1):
        super().__init__()
        self.layers = [DecoderLayer(d_model, nhead_self_attn= nhead_self_attn,
                                    nhead_cross_attn= nhead_cross_attn,
                                    dim_feedforward=dim_feedforward,
                                    dropout= dropout, activation= activation,
                                    layer_norm_eps=layer_norm_eps, 
                                    num_h_c = num_h_c, num_d_c= num_d_c) for _ in range(num_layers)]
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query:Tensor, context_h_c:Sequence[Tensor], context_f_c:Tensor, context_d_c:Sequence[Tensor],
                memory, query_mask: Optional[Tensor] = None, context_key_padding_mask: Optional[Tensor] = None):
        for layer in self.layers:
            query = layer(query, context_h_c, context_f_c, context_d_c, memory, query_mask, context_key_padding_mask)
        query = self.norm(query)
        return query

class TokenEmbedding(pl.LightningModule):

    def __init__(self, h_c_charset_size: int, f_c_charset_size:int, d_c_charset_size:int, 
                 embed_dim:int, num_h_c:int = 2, num_d_c:int = 2):
        super().__init__()
        self.h_c_embedding = [nn.Embedding(num_embeddings=h_c_charset_size, embedding_dim=embed_dim)
                                for _ in range(num_h_c)]
        self.f_c_embedding = nn.Embedding(num_embeddings=f_c_charset_size, embedding_dim=embed_dim)
        self.d_c_embedding = [nn.Embedding(num_embeddings=d_c_charset_size, embedding_dim=embed_dim)
                                for _ in range(num_d_c)]
        self.embed_dim = embed_dim

    def forward(self, h_c_tokens: Sequence[Tensor], f_c_tokens:Tensor, d_c_tokens:Sequence[Tensor]):
        h_c_emb = [math.sqrt(self.embed_dim) * self.h_c_embedding[i](h_c_t) for i,h_c_t in enumerate(h_c_tokens)]
        f_c_emb = math.sqrt(self.embed_dim) * self.f_c_embedding(f_c_tokens)
        d_c_emb = [math.sqrt(self.embed_dim) * self.d_c_embedding[i](d_c_t) for i, d_c_t in enumerate(d_c_tokens)]
        return h_c_emb, f_c_emb, d_c_emb