from torch import Tensor
from typing import Optional, Sequence
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

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='GELU',
                 layer_norm_eps=1e-5, num_h_c:int = 2, num_d_c:int = 2):
        """
        constructor for DecoderLayer
        Args:
        - d_model (int): dimension of the decoder feature vectors

        """
        super().__init__()
        self.num_h_c = num_h_c
        self.num_d_c = num_d_c
        self.self_attn_h_c = [nn.MultiheadAttention(d_model//(num_h_c * 3), nhead / self.num_h_c, 
                                                    dropout= dropout, batch_first= True) for _ in range(num_h_c)]
        for _ in range(self.num_h_c):
            # 0th index will correspond to h_c_n and ith index will correspond to h_c_(n-i)
            self.self_attn_h_c.append(
                                        nn.MultiheadAttention(d_model//(num_h_c * 3), nhead / self.num_h_c, 
                                                                dropout= dropout, batch_first= True)
                                    )
        
        self.self_attn_f_c = nn.MultiheadAttention(d_model // 3, nhead, dropout=dropout, batch_first=True)
        self.self_attn_d_c = [nn.MultiheadAttention(d_model // (num_d_c * 3), nhead / num_d_c,
                                                    dropout=dropout, batch_first=True) for _ in range(num_d_c)]
        self.cross_attn = nn.MultiheadAttention(d_model, nhead * 3, dropout=dropout, batch_first=True)

        # for merging representations
        self.merge1 = nn.Linear(d_model, dim_feedforward)
        self.activation_merge = get_activation(activation)()
        self.dropout_merge1 = nn.Dropout(dropout)
        self.merge2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_merge2 = nn.Dropout(dropout)
        self.norm_merge = nn.LayerNorm(d_model, eps= layer_norm_eps)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_contxt_h_c = [nn.LayerNorm(d_model, eps=layer_norm_eps) for _ in range(num_h_c)]
        self.norm_contxt_f_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_contxt_d_c = [nn.LayerNorm(d_model, eps=layer_norm_eps) for _ in range(num_d_c)]

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation(activation)()

    def forward_stream(self, tgt: Tensor, tgt_norm: Tensor, tgt_h_c_kv: tuple, tgt_f_c_kv:Tensor, tgt_d_c_kv:Tensor,
                       memory: Tensor, tgt_mask: Optional[Tensor], tgt_key_padding_mask: Optional[Tensor]):
        """
        Forward pass for a single stream (i.e. content or query)
        tgt_norm is just a LayerNorm'd tgt. Added as a separate parameter for efficiency.
        Both tgt_kv and memory are expected to be LayerNorm'd too.
        memory is LayerNorm'd by ViT.
        """
        # self-attn
        tgt2_h_c, sa_weights_h_c = [], []
        for self_attn, tgt_h_c in zip(self.self_attn_h_c, tgt_h_c_kv):
            tgt_, sa_ = self_attn(tgt_norm, tgt_h_c, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
            tgt2.append(tgt_)
            sa_weights_h_c.append(sa_)
        tgt2_f_c, sa_weights_f_c = self.self_attn_f_c(tgt_norm, tgt_f_c_kv, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt2_d_c, sa_weights_d_c = []
        for self_attn, tgt_d_c in zip(self.self_attn_d_c, tgt_d_c_kv):
            tgt_, sa_ = self.self_attn_d_c(tgt_norm, tgt_d_c, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
            tgt2_d_c.append(tgt_)
            sa_weights_d_c.append(sa_)
        
        # Merge
        tgt2 = torch.cat([*tgt2_h_c, tgt2_f_c, *tgt2_d_c], dim= -1)
        sa_weights = (tuple(sa_weights_h_c), sa_weights_f_c, sa_weights_d_c)
        tgt2 = self.merge2(self.dropout_merge1(self.activation_merge(self.merge1(self.norm_merge(tgt2)))))
        tgt = tgt + self.dropout_merge2(tgt2)
        
        # cross attn
        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)

        # MLP
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, sa_weights, ca_weights

    def forward(self, query:Tensor, context_h_c:Sequence[Tensor], context_f_c:Tensor, context_d_c:Sequence[Tensor], 
                memory:Tensor, query_mask: Optional[Tensor] = None, context_key_padding_mask: Optional[Tensor] = None):
        query_norm = self.norm_q(query)
        context_h_c_norm = (self.norm_context_h_c[i](context_h_c_i) for i, context_h_c_i in enumerate(context_h_c))
        context_f_c_norm = self.norm_contxt_f_c(context_f_c)
        context_d_c_norm = (self.norm_contxt_d_c[i](context_d_c_i) for i, context_d_c_i in enumerate(context_d_c))
        query = self.forward_stream(query, query_norm, context_h_c_norm, context_f_c_norm, context_d_c_norm,
                                    memory, query_mask, context_key_padding_mask)[0]
        return query

class Decoder(pl.LightningModule):
    __constants__ = ['norm']

    def __init__(self, d_model:int , nhead:int, dim_feedforward= 2048, dropout= 0.1, activation='GELU', 
                 layer_norm_eps=1e-5, num_h_c:int = 2, num_d_c:int = 2, num_layers:int = 1):
        super().__init__()
        self.layers = [DecoderLayer(d_model, nhead, dim_feedforward=dim_feedforward,
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
                 embed_dim: int, num_h_c:int = 2, num_d_c:int = 2):
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