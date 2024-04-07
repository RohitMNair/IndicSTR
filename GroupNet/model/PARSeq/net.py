from model.base import HindiBaseSystem, MalayalamBaseSystem
from model.FocalSTR.encoder import FocalNetEncoder
from torch import Tensor
from typing import Optional, Sequence
from itertools import permutations
from .decoder import Decoder, TokenEmbedding
import torch.nn as nn
import torch
import numpy as np
import math

class HindiPARSeq(HindiBaseSystem):

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
                            num_d_c= 2,
                            num_layers= 1,
                        )

        self.text_embed = TokenEmbedding(
                            h_c_charset_size= len(self.tokenizer.h_c_classes),
                            f_c_charset_size= len(self.tokenizer.f_c_classes),
                            d_c_charset_size= len(self.tokenizer.d_classes),
                            embed_dim= self.hidden_sizes[-1],
                            num_h_c= 2,
                            num_d_c= 2,
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

    def decode(self, h_c_ctx:Sequence[Tensor], f_c_ctx:Tensor, d_c_ctx:Sequence[Tensor], memory:Tensor,
                ctx_padding_mask: Optional[Tensor] = None, query:Optional[Tensor] = None, query_mask: Optional[Tensor] = None):
        N, L = f_c_ctx.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        h_c_null_ctx, f_c_null_ctx, d_c_null_ctx = self.text_embed(
                            h_c_tokens= [h_c[:,0] for h_c in h_c_ctx],
                            f_c_tokens = f_c_ctx[:, 0],
                            d_c_tokens = [d_c[:, 0] for d_c in d_c_ctx],
                        )
        h_c_ctx_emb, f_c_ctx_emb, d_c_ctx_emb  = self.text_embed(
                                h_c_tokens= [h_c[:,1:] for h_c in h_c_ctx],
                                f_c_tokens = f_c_ctx[:, 1:],
                                d_c_tokens = [d_c[:, 1:] for d_c in d_c_ctx],
                            )
        h_c_ctx_emb = [ctx + self.pos_queries[:,:L-1] for ctx in h_c_ctx_emb]
        f_c_ctx_emb += self.pos_queries[:,:L-1]
        d_c_ctx_emb += [ctx + self.pos_queries[:,:L-1] for ctx in d_c_ctx_emb]
        
        h_c_ctx_emb = self.h_c_ctx_dropout([torch.cat([null_ctx, ctx], dim= 1) for null_ctx, ctx in zip(h_c_null_ctx, h_c_ctx_emb)])
        f_c_ctx_emb = self.f_c_ctx_dropout(torch.cat([f_c_null_ctx, f_c_ctx_emb], dim= 1))
        d_c_ctx_emb = self.d_c_ctx_dropout([torch.cat([null_ctx, ctx], dim= 1) for null_ctx, ctx in zip(d_c_null_ctx, d_c_ctx_emb)])

        if query is None:
            query = self.pos_queries[:, :L].expand(N, -1, -1)
        query = self.pos_dropout(query)
        
        return self.decoder(query= query, context_h_c= h_c_ctx_emb, context_f_c= f_c_ctx_emb, context_d_c= d_c_ctx_emb,
                            memory= memory, query_mask= query_mask, context_key_padding_mask= ctx_padding_mask)

    def forward(self, images: Tensor) -> Tensor:
        bs = images.shape[0]
        memory = self.encoder(images)
        pos_queries = self.pos_queries.expand(bs, -1, -1)

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        query_mask = torch.triu(torch.full((self.max_grps, self.max_grps), float('-inf'), device=self.device), 1)

        if self.decode_ar:
            h_c_ctx = []
            for _ in range(2):
                t_ = torch.full((bs, self.max_grps), self.tokenizer.pad_id, dtype=torch.long, device=self.device)
                t_[:, 0] = self.tokenizer.blank_id
                h_c_ctx.append(t_)

            f_c_ctx = torch.full((bs, self.max_grps), self.tokenizer.pad_id, dtype=torch.long, device=self.device)
            f_c_ctx[:, 0] = self.tokenizer.blank_id # change with BOS later
            d_c_ctx = []
            for _ in range(2):
                t_ = torch.full((bs, self.max_grps), self.tokenizer.pad_id, dtype=torch.long, device=self.device)
                t_[:, 0] = self.tokenizer.blank_id
                d_c_ctx.append(t_)
          
            logits = []
            for i in range(self.max_grps):
                j = i + 1  # next token index
                # Efficient decoding:
                # Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                tgt_out = self.decode(query=pos_queries[:, i:j], h_c_ctx= [ctx[:,:j] for ctx in h_c_ctx], f_c_ctx= h_c_ctx,
                                      d_c_ctx= d_c_ctx, memory= memory, query_mask=query_mask[i:j, :j], context_key_padding_mask= None)
                # the next token probability is in the output's ith token position
                h_c_2_logit, h_c_1_logit, f_c_logit, d_c_logit = self.classifier(tgt_out)
                logits.append([h_c_2_logit, h_c_1_logit, f_c_logit, d_c_logit])
                if j < self.max_grps:
                    # greedy decode. add the next token index to the target input
                    h_c_ctx[0][:, j] = h_c_2_logit.squeeze().argmax(-1)
                    h_c_ctx[1][:, j] = h_c_1_logit.squeeze().argmax(-1)
                    f_c_ctx[:, j] = f_c_logit.squeeze().argmax(-1)
                    vals, indices = torch.topk(d_c_logit.squeeze(), k= 2)
                    d_c_ctx[0][:, j] = indices[:, 0]
                    d_c_ctx[1][:, j] = indices[:, 1]
                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                    # if (tgt_in == self.eos_id).any(dim=-1).all():
                    #     break

            # logits = torch.cat(logits, dim=1)
            logits = [
                        torch.cat([h_c_2_logit for h_c_2_logit in logits[0]], dim= 1), 
                        torch.cat([h_c_1_logit for h_c_1_logit in logits[1]], dim= 1),
                        torch.cat([f_c_logit for f_c_logit in logits[2]], dim= 1),
                        torch.cat([d_c_logit for d_c_logit in logits[3]], dim= 1),
                    ]
        else:
            # No prior context, so input is just <bos>. We query all positions.
            h_c_ctx = [torch.full((bs, 1), self.tokenizer.bos_id, dtype= torch.long, device=self.device) for _ in range(2)]
            f_c_ctx = torch.full((bs, 1), self.tokenizer.bos_id, dtype= torch.long, device= self.device)
            d_c_ctx = [torch.full((bs, 1), self.tokenizer.bos_id, dtype= torch.long, device=self.device) for _ in range(2)]
            tgt_out = self.decode(query=pos_queries, h_c_ctx= h_c_ctx, f_c_ctx= f_c_ctx, d_c_ctx= d_c_ctx, memory= memory)
            logits = self.classifier(tgt_out)

        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[torch.triu(torch.ones(self.max_grps, self.max_grps, dtype=torch.bool, device=self.device), 2)] = 0
            bos = torch.full((bs, 1), self.tokenizer.bos_id, dtype=torch.long, device=self.device)
            for i in range(self.refine_iters):
                # Prior context is the previous output.
                h_c_ctx = [torch.cat([bos, h_c_logits[:, :-1].argmax(-1)], dim= 1) for h_c_logits in logits[:2]]
                f_c_ctx = torch.cat([bos, logits[2][:, :-1].argmax(-1)], dim= 1)
                vals, indices = torch.topk(logits[-1])
                d_c_ctx = [torch.cat([bos, indices[:,:,i]]) for i in range(2)]

                # ctx_padding_mask = ((tgt_in == self.eos_id).int().cumsum(-1) > 0)  # mask tokens beyond the first EOS token.
                h_c_ctx_pad = sum([(ctx == self.tokenizer.eos_id).int().cumsum(-1) for ctx in h_c_ctx])
                f_c_ctx_pad = (f_c_ctx == self.tokenizer.eos_id).int().cumsum(-1)
                d_c_ctx_pad = sum([(ctx == self.tokenizer.eos_id).int().cumsum(-1) for ctx in d_c_ctx])
                ctx_padding_mask = (h_c_ctx_pad + f_c_ctx_pad + d_c_ctx_pad) > 0
                tgt_out = self.decode(query=pos_queries, h_c_ctx= h_c_ctx, f_c_ctx= f_c_ctx,
                                        memory= memory, query_mask=query_mask[:, :f_c_ctx.shape[1]],
                                        ctx_padding_mask= ctx_padding_mask,
                                      )
                logits = self.classifier(tgt_out)

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
            return torch.arange(3, device=self.device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=self.device)] if self.perm_forward else []
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
            perm_pool = torch.as_tensor(list(permutations(range(max_num_chars), max_num_chars)), device=self.device)[selector]
            # If the forward permutation is always selected, no need to add it to the pool for sampling
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend([torch.randperm(max_num_chars, device=self.device) for _ in range(num_gen_perms - len(perms))])
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
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self.device)
        return perms

    def generate_attn_masks(self, perm):
        """Generate attention masks given a sequence permutation (includes pos. for bos and eos tokens)
        :param perm: the permutation sequence. i = 0 is always the BOS
        :return: lookahead attention masks
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=self.device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = float('-inf')
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self.device)] = float('-inf')  # mask "self"
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        batch_size = len(labels)
        h_c_2_targets, h_c_1_targets, f_c_targets, d_c_targets = (
                                  torch.zeros(batch_size, self.max_grps, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps, self.num_d_classes, device= self.device),
                                )
        
        n_grps = [self.max_grps for i in range(batch_size)]
        for idx,label in enumerate(labels, start= 0):
            h_c_2_targets[idx], h_c_1_targets[idx], f_c_targets[idx], d_c_targets[idx], n_grps[idx] = self.tokenizer.label_encoder(label, device= self.device)

        # Encode the source sequence (i.e. the image codes)
        memory = self.encode(imgs)

        # Prepare the target sequences (input and output)
        tgt_perms = self.gen_tgt_perms(f_c_targets)
        # ignore BOS tag
        h_c_2_out, h_c_1_out, f_c_out, d_c_out = h_c_2_targets[:, 1:], h_c_1_targets[:,1:], f_c_targets[:,1:], d_c_targets[:,1:]
        # The [EOS] token is not depended upon by any other token in any permutation ordering
        ctx_padding_mask = (f_c_out == self.pad_id) | (f_c_out == self.eos_id) # pads and eos are same accross char grps

        loss = 0
        loss_numel = 0
        n = (f_c_out != self.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm)
            vals, indices = torch.topk(d_c_targets, k= 2)
            out = self.decode(h_c_ctx= [h_c_2_targets[:,:-1], h_c_1_targets[:,:-1]], f_c_ctx= f_c_targets[:,:-1], 
                              d_c_ctx= [indices[:,:-1,i] for i in range(2)], # ignore the additional target for EOS
                              memory= memory, query_mask=query_mask, ctx_padding_mask= ctx_padding_mask)
            
            (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits) = self.classifier(out)
            # loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
            # Get the flattened versions of the targets and the logits for grp level metrics
            ((flat_h_c_2_targets, flat_h_c_1_targets, flat_f_c_targets, flat_d_c_targets), 
            (flat_h_c_2_logits, flat_h_c_1_logits, flat_f_c_logits, flat_d_c_logits)) = self._get_flattened_non_pad(
                                                                                    targets= (h_c_2_out, h_c_1_out, f_c_out, d_c_out),
                                                                                    logits= (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits),
                                                                                )
            loss += self.h_c_2_loss(input= flat_h_c_2_logits, target= flat_h_c_2_targets) \
            + self.h_c_1_loss(input= flat_h_c_1_logits, target= flat_h_c_1_targets) \
            + self.f_c_loss(input= flat_f_c_logits, target= flat_f_c_targets) \
            + self.d_loss(input= flat_d_c_logits, target= flat_d_c_targets)
            loss *= n
            loss_numel += n
            # After the second iteration (i.e. done with canonical and reverse orderings),
            # remove the [EOS] tokens for the succeeding perms
            if i == 1:
                tgt_out = torch.where(f_c_out == self.eos_id, self.pad_id, f_c_out)
                n = (tgt_out != self.pad_id).sum().item()
        loss /= loss_numel

        self.log('loss', loss)
        return loss


