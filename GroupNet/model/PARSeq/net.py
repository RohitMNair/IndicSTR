from model.base import HindiBaseSystem, MalayalamBaseSystem
from model.FocalSTR.encoder import FocalNetEncoder
from torch import Tensor
from model.head import get_activation
from typing import Optional, Sequence
from itertools import permutations
import lightning.pytorch as pl
import torch.nn as nn
import torch
import numpy as np
import math

class DecoderLayer(pl.LightningModule):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
       This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='GELU',
                 layer_norm_eps=1e-5, num_h_c:int = 2):
        """
        constructor for DecoderLayer
        Args:
        - d_model (int): dimension of the decoder feature vectors

        """
        super().__init__()
        self.num_h_c = num_h_c
        self.self_attn_h_c = []
        for _ in range(self.num_h_c):
            # 0th index will correspond to h_c_n and ith index will correspond to h_c_(n-i)
            print(d_model//(num_h_c * 3), nhead / self.num_h_c)
            self.self_attn_h_c.append(
                                        nn.MultiheadAttention(d_model//(num_h_c * 3), nhead / self.num_h_c, 
                                                                dropout= dropout, batch_first= True)
                                    )
        
        self.self_attn_f_c = nn.MultiheadAttention(d_model // 3, nhead, dropout=dropout, batch_first=True)
        self.self_attn_d_c = nn.MultiheadAttention(d_model // 3, nhead, dropout=dropout, batch_first=True)
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
        self.norm_contxt_d_c = nn.LayerNorm(d_model, eps=layer_norm_eps)

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
        tgt2_h_c = []
        sa_weights_h_c = []
        for self_attn, tgt_h_c in zip(self.self_attn_h_c, tgt_h_c_kv):
            tgt_, sa_ = self_attn(tgt_norm, tgt_h_c, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
            tgt2.append(tgt_)
            sa_weights_h_c.append(sa_)
        tgt2_f_c, sa_weights_f_c = self.self_attn_f_c(tgt_norm, tgt_f_c_kv, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt2_d_c, sa_weights_d_c = self.self_attn_d_c(tgt_norm, tgt_d_c_kv, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)

        # Merge
        tgt2 = torch.cat([*tgt2_h_c, tgt2_f_c, tgt2_d_c], dim= -1)
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

    def forward(self, query:Tensor, context_h_c:tuple, context_f_c:Tensor, context_d_c:Tensor, memory:Tensor,
                query_mask: Optional[Tensor] = None, context_key_padding_mask: Optional[Tensor] = None):
        query_norm = self.norm_q(query)
        context_h_c_norm = (self.norm_context_h_c[i](context_h_c_i) for i, context_h_c_i in enumerate(context_h_c))
        context_f_c_norm = self.norm_contxt_f_c(context_f_c)
        context_d_c_norm = self.norm_contxt_d_c(context_d_c)
        query = self.forward_stream(query, query_norm, context_h_c_norm, context_f_c_norm, context_d_c_norm,
                                    memory, query_mask, context_key_padding_mask)[0]
        return query

class Decoder(pl.LightningModule):
    __constants__ = ['norm']

    def __init__(self, d_model:int , nhead:int, dim_feedforward= 2048, dropout= 0.1, activation='GELU', 
                 layer_norm_eps=1e-5, num_h_c = 2, num_layers = 1):
        super().__init__()
        self.layers = nn.Sequential(
                        *[DecoderLayer(d_model, nhead, dim_feedforward=dim_feedforward,
                                    dropout= dropout, activation= activation,
                                    layer_norm_eps=layer_norm_eps, num_h_c = num_h_c) for _ in range(num_layers)]
                    )
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, context, memory, query_mask: Optional[Tensor] = None, context_mask: Optional[Tensor] = None,
                context_key_padding_mask: Optional[Tensor] = None):
        print(query.shape, context.shape, memory.shape)
        query = self.layers(query, context, memory, query_mask, context_mask, context_key_padding_mask)
        query = self.norm(query)
        return query

class TokenEmbedding(pl.LightningModule):

    def __init__(self, h_c_charset_size: int, f_c_charset_size:int, d_c_charset_size:int, embed_dim: int, num_h_c:int = 2):
        super().__init__()
        self.h_c_embedding = []
        for _ in range(num_h_c):
            self.h_c_embedding.append(nn.Embedding(num_embeddings=h_c_charset_size, embedding_dim=embed_dim))
            
        self.f_c_embedding = nn.Embedding(num_embeddings=f_c_charset_size, embedding_dim=embed_dim)
        self.d_c_embedding = nn.Embedding(num_embeddings=d_c_charset_size, embedding_dim=embed_dim)
        self.embed_dim = embed_dim

    def forward(self, h_c_tokens: Tensor, f_c_tokens:Tensor, d_c_tokens:Tensor):
        h_c_emb = []
        for i, h_c in enumerate(h_c_tokens):
            h_c_emb.append(self.h_c_embedding[i](h_c))
        f_c_emb = math.sqrt(self.embed_dim) * self.f_c_embedding(f_c_tokens)
        d_c_emb = math.sqrt(self.embed_dim) * self.d_c_embedding(d_c_tokens)
        return map(lambda x: math.sqrt(self.embed_dim) * x, h_c_emb), f_c_emb, d_c_emb


class PARSeq(HindiBaseSystem):

    def __init__(self, embed_dim: int = 96, depths:list= [2, 2, 6, 2],
                 focal_levels:list= [2, 2, 2, 2], focal_windows:list= [3, 3, 3, 3],
                 mlp_ratio: float= 4.0, drop_path_rate:float = 0.1, initializer_range: float = 0.02, 
                 layer_norm_eps: float = 1e-12, image_size: int = 224, patch_size: int = 16, 
                 num_channels: int = 3, dec_num_heads: int = 12, dec_mlp_ratio: int= 4, 
                 dec_depth: int= 1, perm_num:int= 25, perm_forward: bool= True, 
                 perm_mirrored: bool= True, decode_ar: bool= True, refine_iters: int= 1, 
                 dropout: float= 0.1, threshold:float= 0.5, max_grps:int = 25,
                 learning_rate: float= 1e-4, weight_decay: float= 1.0e-4, warmup_pct:float= 0.3,
                 ) -> None:
        self.save_hyperparameters()
        self.embed_dim = embed_dim
        self.hidden_sizes = [self.embed_dim * (2 ** i) for i in range(len(depths))] 
        super().__init__(max_grps= max_grps, hidden_size= self.hidden_sizes[-1], threshold= threshold,
                         learning_rate= learning_rate, weight_decay= weight_decay, warmup_pct= warmup_pct)
        self.depths = depths
        self.focal_levels = focal_levels
        self.focal_windows = focal_windows
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        # Decoder params
        self.dec_num_heads = dec_num_heads
        self.dec_mlp_ratio = dec_mlp_ratio
        self.dec_depth = dec_depth
        # train/val params
        self.perm_num = perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored
        self.decode_ar = decode_ar
        self.refine_iters = refine_iters       
        # Perm/attn mask
        self.rng = np.random.default_rng()
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num

        self.encoder = FocalNetEncoder(
            hidden_dropout_prob= self.dropout, 
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
    
        self.decoder = Decoder( 
                            d_model= self.hidden_sizes[-1], 
                            nhead= self.dec_num_heads / 3,
                            dim_feedforward= self.hidden_sizes[-1] * self.dec_mlp_ratio,
                            dropout= self.dropout,
                            activation="GELU",
                            layer_norm_eps= self.layer_norm_eps,
                            num_h_c= 2,
                            num_layers= 1,
                        )

        self.text_embed = TokenEmbedding(
                            h_c_charset_size= len(self.tokenizer.h_c_classes),
                            f_c_charset_size= len(self.tokenizer.f_c_classes),
                            d_c_charset_size= len(self.tokenizer.d_classes),
                            embed_dim= self.hidden_sizes[-1],
                            num_h_c= 2
                        )

        # +1 for <eos>
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_grps, self.hidden_sizes[-1]))
        self.h_c_ctx_dropout = [nn.Dropout(dropout) for _ in range(2)]
        self.f_c_ctx_dropout = nn.Dropout(dropout)
        self.d_c_ctx_dropout = nn.Dropout(dropout)
        self.pos_dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.pos_queries, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        param_names = {'text_embed.embedding.weight', 'pos_queries'}
        enc_param_names = {'encoder.' + n for n in self.encoder.no_weight_decay()}
        return param_names.union(enc_param_names)

    def decode(self, h_c_ctx:Tensor, f_c_ctx:Tensor, d_c_ctx:Tensor, memory:Tensor,
               ctx_mask: Optional[Tensor] = None, ctx_padding_mask: Optional[Tensor] = None,
               query: Optional[Tensor] = None, query_mask: Optional[Tensor] = None):
        N, L = h_c_ctx.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        h_c_null_ctx, f_c_null_ctx, d_c_null_ctx = self.text_embed(
                            h_c_tokens= [h_c[:,0] for h_c in h_c_ctx],
                            f_c_tokens = f_c_ctx[:, 0],
                            d_c_tokens = d_c_ctx[:, 0],
                        )
        h_c_ctx_emb, f_c_ctx_emb, d_c_ctx_emb  = self.text_embed(
                                h_c_tokens= [h_c[:,1:] for h_c in h_c_ctx],
                                f_c_tokens = f_c_ctx[:, 1:],
                                d_c_tokens = d_c_ctx[:, 1:],
                            )
        
        
        if query is None:
            query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.pos_dropout(tgt_query)
        # return self.decoder(tgt_query, tgt_emb, memory, query_mask, mask, padding_mask)
        return self.decoder(query, context, memory, query_mask, ctx_mask, ctx_padding_mask)

    def forward(self, images: Tensor) -> Tensor:
        bs = images.shape[0]
        memory = self.encoder(images)
        pos_queries = self.pos_queries.expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        context_mask = query_mask = torch.triu(torch.full((self.max_grps, self.max_grps), float('-inf'), device=self._device), 1)

        if self.decode_ar:
            h_c_context = []
            for _ in range(2):
                t_ = torch.full((bs, self.max_grps), self.tokenizer.pad_id, dtype=torch.long, device=self._device)
                t_[:, 0] = self.tokenizer.blank_id
                h_c_context.append(t_)

            f_c_context = torch.full((bs, self.max_grps), self.tokenizer.pad_id, dtype=torch.long, device=self._device)
            d_c_context = torch.full((bs, self.max_grps), self.tokenizer.pad_id, dtype=torch.long, device=self._device)
            f_c_context[:, 0] = d_c_context[:, 0] = self.tokenizer.blank_id # change with BOS later
            
            logits = []
            for i in range(self.max_grps):
                j = i + 1  # next token index
                # Efficient decoding:
                # Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                tgt_out = self.decode([:, :j], memory, context_mask[:j, :j], tgt_query=pos_queries[:, i:j],
                                      tgt_query_mask=query_mask[i:j, :j])
                # the next token probability is in the output's ith token position
                p_i = self.head(tgt_out)
                logits.append(p_i)
                if j < self.max_grps:
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)
                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                    if (tgt_in == self.eos_id).any(dim=-1).all():
                        break

            logits = torch.cat(logits, dim=1)
        else:
            # No prior context, so input is just <bos>. We query all positions.
            tgt_in = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            tgt_out = self.decode(tgt_in, memory, tgt_query=pos_queries)
            logits = self.head(tgt_out)

        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[torch.triu(torch.ones(self.max_grps, self.max_grps, dtype=torch.bool, device=self._device), 2)] = 0
            bos = torch.full((bs, 1), self.bos_id, dtype=torch.long, device=self._device)
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                tgt_in = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)
                tgt_padding_mask = ((tgt_in == self.eos_id).int().cumsum(-1) > 0)  # mask tokens beyond the first EOS token.
                tgt_out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask,
                                      tgt_query=pos_queries, tgt_query_mask=query_mask[:, :tgt_in.shape[1]])
                logits = self.head(tgt_out)

        return logits

    def gen_tgt_perms(self, tgt):
        """Generate shared permutations for the whole batch.
           This works because the same attention mask can be used for the shorter sequences
           because of the padding mask.
        """
        # We don't permute the position of BOS, we permute EOS separately
        max_num_chars = tgt.shape[1] - 2
        # Special handling for 1-character sequences
        if max_num_chars == 1:
            return torch.arange(3, device=self._device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=self._device)] if self.perm_forward else []
        # Additional permutations if needed
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        # For 4-char sequences and shorter, we generate all permutations and sample from the pool to avoid collisions
        # Note that this code path might NEVER get executed since the labels in a mini-batch typically exceed 4 chars.
        if max_num_chars < 5:
            # Pool of permutations to sample from. We only need the first half (if complementary option is selected)
            # Special handling for max_num_chars == 4 which correctly divides the pool into the flipped halves
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(list(permutations(range(max_num_chars), max_num_chars)), device=self._device)[selector]
            # If the forward permutation is always selected, no need to add it to the pool for sampling
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend([torch.randperm(max_num_chars, device=self._device) for _ in range(num_gen_perms - len(perms))])
            perms = torch.stack(perms)
        if self.perm_mirrored:
            # Add complementary pairs
            comp = perms.flip(-1)
            # Stack in such a way that the pairs are next to each other.
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        # NOTE:
        # The only meaningful way of permuting the EOS position is by moving it one character position at a time.
        # However, since the number of permutations = T! and number of EOS positions = T + 1, the number of possible EOS
        # positions will always be much less than the number of permutations (unless a low perm_num is set).
        # Thus, it would be simpler to just train EOS using the full and null contexts rather than trying to evenly
        # distribute it across the chosen number of permutations.
        # Add position indices of BOS and EOS
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        # Special handling for the reverse direction. This does two things:
        # 1. Reverse context for the characters
        # 2. Null context for [EOS] (required for learning to predict [EOS] in NAR mode)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)
        return perms

    def generate_attn_masks(self, perm):
        """Generate attention masks given a sequence permutation (includes pos. for bos and eos tokens)
        :param perm: the permutation sequence. i = 0 is always the BOS
        :return: lookahead attention masks
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = float('-inf')
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = float('-inf')  # mask "self"
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def training_step(self, batch, batch_idx):
        images, labels = batch
        tgt = self.tokenizer.encode(labels, self._device)

        # Encode the source sequence (i.e. the image codes)
        memory = self.encode(images)

        # Prepare the target sequences (input and output)
        tgt_perms = self.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        # The [EOS] token is not depended upon by any other token in any permutation ordering
        tgt_padding_mask = (tgt_in == self.pad_id) | (tgt_in == self.eos_id)

        loss = 0
        loss_numel = 0
        n = (tgt_out != self.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm)
            out = self.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
            logits = self.head(out).flatten(end_dim=1)
            loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
            loss_numel += n
            # After the second iteration (i.e. done with canonical and reverse orderings),
            # remove the [EOS] tokens for the succeeding perms
            if i == 1:
                tgt_out = torch.where(tgt_out == self.eos_id, self.pad_id, tgt_out)
                n = (tgt_out != self.pad_id).sum().item()
        loss /= loss_numel

        self.log('loss', loss)
        return loss


