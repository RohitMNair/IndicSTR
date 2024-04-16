from model.base import HindiBaseSystem, MalayalamBaseSystem, PARSeqBaseSystem
from model.FocalSTR.encoder import FocalNetEncoder
from torch import Tensor
from typing import Optional, Sequence, Union, Tuple
from itertools import permutations
from .decoder import Decoder, TokenEmbedding
from model.head import HindiGrpClassifier
from data.tokenizer import HindiPARSeqTokenizer
from model.ViTSTR.encoder import ViTEncoder
import torch.nn as nn
import torch
import numpy as np
import math

class HindiPARSeq(HindiBaseSystem):

    def __init__(self, embed_dim: int = 96, depths:list= [2, 2, 6, 2],
                 focal_levels:list= [2, 2, 2, 2], focal_windows:list= [3, 3, 3, 3],
                 mlp_ratio: float= 4.0, drop_path_rate:float = 0.1, initializer_range: float = 0.02, 
                 layer_norm_eps: float = 1e-12, image_size: int = 128, patch_size: int = 8, 
                 num_channels: int = 3, dec_num_sa_heads:Sequence[Union[Sequence[int], int]] = [[2,2], 4, [2,2]], 
                 dec_num_ca_heads:int= 12, dec_mlp_ratio: int= 4, dec_depth: int= 1, perm_num:int= 25, 
                 perm_forward: bool= True, perm_mirrored: bool= True, decode_ar: bool= True,
                 refine_iters: int= 1, dropout: float= 0.1, threshold:float= 0.5, max_grps:int = 25,
                 learning_rate: float= 1e-4, weight_decay: float= 1.0e-4, warmup_pct:float= 0.3,
                 ) -> None:
        self.save_hyperparameters()
        self.embed_dim = embed_dim
        self.num_h_c = 2
        self.num_d_c = 2
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
        self.dec_num_sa_heads = dec_num_sa_heads
        self.dec_num_ca_heads = dec_num_ca_heads
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

        self.tokenizer = HindiPARSeqTokenizer(threshold= threshold, max_grps= max_grps)
        self.num_h_c_classes = len(self.tokenizer.h_c_classes)
        self.num_f_c_classes = len(self.tokenizer.f_c_classes)
        self.num_d_classes =  len(self.tokenizer.d_classes)

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
                            nhead_self_attn= dec_num_sa_heads,
                            nhead_cross_attn= dec_num_ca_heads,
                            dim_feedforward= self.hidden_sizes[-1] * self.dec_mlp_ratio,
                            dropout= dropout,
                            activation="GELU",
                            layer_norm_eps= self.layer_norm_eps,
                            num_h_c= self.num_h_c,
                            num_d_c= self.num_d_c,
                            num_layers= 1,
                        )

        self.text_embed = TokenEmbedding(
                            h_c_charset_size= self.num_h_c_classes,
                            f_c_charset_size= self.num_f_c_classes,
                            d_c_charset_size= self.num_d_classes,
                            embed_dim= self.hidden_sizes[-1],
                            num_h_c= self.num_h_c,
                            num_d_c= self.num_d_c,
                        )
        self.classifier = HindiGrpClassifier(
            hidden_size= self.hidden_sizes[-1], 
            num_half_character_classes= self.num_h_c_classes - 2, # we dont predict BOS or PAD
            num_full_character_classes= self.num_f_c_classes - 2,
            num_diacritic_classes= self.num_d_classes - 2,
        )
        self.h_c_2_loss = nn.CrossEntropyLoss(reduction= 'mean', ignore_index= self.tokenizer.pad_id_h_c)
        self.h_c_1_loss = nn.CrossEntropyLoss(reduction= 'mean', ignore_index= self.tokenizer.pad_id_h_c)
        self.f_c_loss = nn.CrossEntropyLoss(reduction= 'mean', ignore_index= self.tokenizer.pad_id_f_c)
        self.d_loss = nn.BCEWithLogitsLoss(reduction= 'mean')
        # +1 for <eos>
        self.pos_queries = nn.Parameter(torch.Tensor(1, max_grps + 1, self.hidden_sizes[-1])) # +1 for eos
        self.h_c_ctx_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(self.num_h_c)])
        self.f_c_ctx_dropout = nn.Dropout(dropout)
        self.d_c_ctx_dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(self.num_d_c)])
        self.pos_dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.pos_queries, std=.02)

    def _get_flattened_non_pad(self, targets: Tuple[Tensor, Tensor, Tensor, Tensor],
                                logits: Tuple[Tensor, Tensor, Tensor, Tensor]):
        """
        Function which returns a flattened version of the targets and logits, it flattens the group dimension
        Args:
        - targets (tuple(Tensor, Tensor, Tensor, Tensor)): A tuple consisting of half-char 2, half-char 1, full char, & diacritic targets
        - logits (tuple(Tensor, Tensor, Tensor, Tensor)): A tuple consisting of half-char 2, half-char 1, full char, & diacritic logits

        Returns:
        - tuple(tuple(Tensor, Tensor, Tensor, Tensor), 
            tuple(Tensor, Tensor, Tensor, Tensor)): (half-char 2, half-char 1, full char, & diacritic targets), 
                                                    (half-char 2, half-char 1, full char, & diacritic logits)
        """
        h_c_2_targets, h_c_1_targets, f_c_targets, d_targets = targets
        h_c_2_logits, h_c_1_logits, f_c_logits, d_logits = logits

        flat_h_c_2_targets = h_c_2_targets.reshape(-1)
        flat_h_c_1_targets = h_c_1_targets.reshape(-1)
        flat_f_c_targets = f_c_targets.reshape(-1)
        flat_d_targets = d_targets.reshape(-1, self.num_d_classes)
        # print(f"The Flattened Targets {flat_h_c_2_targets}\n{flat_h_c_1_targets}\n{flat_f_c_targets}\n{flat_d_targets}\n\n")

        flat_h_c_2_non_pad = (flat_h_c_2_targets != self.tokenizer.pad_id_h_c)
        flat_h_c_1_non_pad = (flat_h_c_1_targets != self.tokenizer.pad_id_h_c)
        flat_f_c_non_pad = (flat_f_c_targets != self.tokenizer.pad_id_f_c)
        d_pad = torch.zeros(self.num_d_classes, dtype = torch.float32, device= self.device)
        d_pad[self.tokenizer.pad_id_d_c] = 1.
        flat_d_non_pad = ~ torch.all(flat_d_targets == d_pad, dim= 1)
        assert torch.all((flat_h_c_2_non_pad == flat_h_c_1_non_pad) == (flat_f_c_non_pad == flat_d_non_pad)).item(), \
                f"Pads are not aligned properly {(flat_f_c_non_pad == flat_d_non_pad)} {(flat_h_c_2_non_pad == flat_h_c_1_non_pad)}"

        flat_h_c_2_targets = flat_h_c_2_targets[flat_h_c_2_non_pad]
        flat_h_c_1_targets = flat_h_c_1_targets[flat_h_c_2_non_pad]
        flat_f_c_targets = flat_f_c_targets[flat_h_c_2_non_pad]
        flat_d_targets = flat_d_targets[flat_h_c_2_non_pad]

        flat_h_c_2_logits = h_c_2_logits.reshape(-1, self.num_h_c_classes - 2)[flat_h_c_2_non_pad]
        flat_h_c_1_logits = h_c_1_logits.reshape(-1, self.num_h_c_classes - 2)[flat_h_c_2_non_pad]
        flat_f_c_logits = f_c_logits.reshape(-1, self.num_f_c_classes - 2)[flat_h_c_2_non_pad]
        flat_d_logits = d_logits.reshape(-1, self.num_d_classes - 2)[flat_h_c_2_non_pad]

        return ((flat_h_c_2_targets, flat_h_c_1_targets, flat_f_c_targets, flat_d_targets[:,:-2]), # dont send PAD and EOS for diacritic
                (flat_h_c_2_logits, flat_h_c_1_logits, flat_f_c_logits, flat_d_logits))
    
    def decode(self, h_c_ctx:Sequence[Tensor], f_c_ctx:Tensor, d_c_ctx:Sequence[Tensor], memory:Tensor,
                context_key_padding_mask: Optional[Tensor] = None, query:Optional[Tensor] = None, query_mask: Optional[Tensor] = None):
        N, L = f_c_ctx.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        h_c_null_ctx, f_c_null_ctx, d_c_null_ctx = self.text_embed(
                            h_c_tokens= [h_c[:,:1] for h_c in h_c_ctx],
                            f_c_tokens = f_c_ctx[:, :1],
                            d_c_tokens = [d_c[:, :1] for d_c in d_c_ctx],
                        )
        h_c_ctx_emb, f_c_ctx_emb, d_c_ctx_emb  = self.text_embed(
                                h_c_tokens= [h_c[:,1:] for h_c in h_c_ctx],
                                f_c_tokens = f_c_ctx[:, 1:],
                                d_c_tokens = [d_c[:, 1:] for d_c in d_c_ctx],
                            )
        
        h_c_ctx_emb = [ctx + self.pos_queries[:,:L-1] for ctx in h_c_ctx_emb]
        f_c_ctx_emb += self.pos_queries[:,:L-1]
        d_c_ctx_emb += [ctx + self.pos_queries[:,:L-1] for ctx in d_c_ctx_emb]

        h_c_ctx_emb = [self.h_c_ctx_dropout[i](torch.cat([null_ctx, ctx], dim= 1)) for i, (null_ctx, ctx) in enumerate(zip(h_c_null_ctx, h_c_ctx_emb))]
        f_c_ctx_emb = self.f_c_ctx_dropout(torch.cat([f_c_null_ctx, f_c_ctx_emb], dim= 1))
        d_c_ctx_emb = [self.d_c_ctx_dropout[i](torch.cat([null_ctx, ctx], dim= 1)) for i, (null_ctx, ctx) in enumerate(zip(d_c_null_ctx, d_c_ctx_emb))]

        if query is None:
            query = self.pos_queries[:, :L].expand(N, -1, -1)
        query = self.pos_dropout(query)
        
        return self.decoder(query= query, context_h_c= h_c_ctx_emb, context_f_c= f_c_ctx_emb, context_d_c= d_c_ctx_emb,
                            memory= memory, query_mask= query_mask, context_key_padding_mask= context_key_padding_mask)

    def forward(self, images: Tensor) -> Tensor:
        bs = images.shape[0]
        memory = self.encoder(images)
        pos_queries = self.pos_queries.expand(bs, -1, -1)
        num_steps = self.max_grps + 1 # for eos

        # Special case for the forward permutation. Faster than using `generate_attn_masks()`
        query_mask = torch.triu(torch.full((num_steps, num_steps), True, dtype=torch.bool, device=self.device), 1)

        if self.decode_ar:
            h_c_ctx = []
            for _ in range(self.num_h_c):
                t_ = torch.full((bs, num_steps), self.tokenizer.pad_id_h_c, dtype=torch.long, device=self.device)
                t_[:, 0] = self.tokenizer.bos_id_h_c
                h_c_ctx.append(t_)

            f_c_ctx = torch.full((bs, num_steps), self.tokenizer.pad_id_f_c, dtype=torch.long, device=self.device)
            f_c_ctx[:, 0] = self.tokenizer.bos_id_f_c
            d_c_ctx = []
            for _ in range(self.num_d_c):
                t_ = torch.full((bs, num_steps), self.tokenizer.pad_id_d_c, dtype=torch.long, device=self.device)
                t_[:, 0] = self.tokenizer.bos_id_d_c
                d_c_ctx.append(t_)

            logits = []
            for i in range(num_steps):
                j = i + 1  # next token index
                # Efficient decoding:
                # Input the context up to the ith token. We use only one query (at position = i) at a time.
                # This works because of the lookahead masking effect of the canonical (forward) AR context.
                # Past tokens have no access to future tokens, hence are fixed once computed.
                tgt_out = self.decode(query=pos_queries[:, i:j], h_c_ctx= [ctx[:,:j] for ctx in h_c_ctx], f_c_ctx= f_c_ctx[:,:j],
                                      d_c_ctx= [ctx[:,:j] for ctx in d_c_ctx], memory= memory, query_mask=query_mask[i:j, :j], 
                                      context_key_padding_mask= None)
                # the next token probability is in the output's ith token position
                h_c_2_logit, h_c_1_logit, f_c_logit, d_c_logit = self.classifier(tgt_out)
                logits.append([h_c_2_logit, h_c_1_logit, f_c_logit, d_c_logit])
                if j < num_steps:
                    # greedy decode. add the next token index to the target input
                    h_c_ctx[0][:, j] = h_c_2_logit.squeeze().argmax(-1)
                    h_c_ctx[1][:, j] = h_c_1_logit.squeeze().argmax(-1)
                    f_c_ctx[:, j] = f_c_logit.squeeze().argmax(-1)
                    vals, indices = torch.topk(d_c_logit.squeeze(), k= self.num_d_c)
                    d_c_ctx[0][:, j] = indices[:, 0]
                    d_c_ctx[1][:, j] = indices[:, 1]
                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                    # if (tgt_in == self.eos_id).any(dim=-1).all():
                    #     break

            # logits = torch.cat(logits, dim=1)
            logits = [
                        torch.cat([logit[0] for logit in logits], dim= 1), 
                        torch.cat([logit[1] for logit in logits], dim= 1),
                        torch.cat([logit[2] for logit in logits], dim= 1),
                        torch.cat([logit[3] for logit in logits], dim= 1),
                    ]
        else:
            # No prior context, so input is just <bos>. We query all positions.
            h_c_ctx = [torch.full((bs, 1), self.tokenizer.bos_id_h_c, dtype= torch.long, device=self.device) for _ in range(self.num_h_c)]
            f_c_ctx = torch.full((bs, 1), self.tokenizer.bos_id_f_c, dtype= torch.long, device= self.device)
            d_c_ctx = [torch.full((bs, 1), self.tokenizer.bos_id_d_c, dtype= torch.long, device=self.device) for _ in range(self.num_d_c)]
            tgt_out = self.decode(query=pos_queries, h_c_ctx= h_c_ctx, f_c_ctx= f_c_ctx, d_c_ctx= d_c_ctx, memory= memory)
            logits = self.classifier(tgt_out)

        if self.refine_iters:
            # For iterative refinement, we always use a 'cloze' mask.
            # We can derive it from the AR forward mask by unmasking the token context to the right.
            query_mask[torch.triu(torch.ones(num_steps, num_steps, dtype=torch.bool, device=self.device), 2)] = 0
            bos_h_c = torch.full((bs, 1), self.tokenizer.bos_id_h_c, dtype=torch.long, device=self.device)
            bos_f_c = torch.full((bs, 1), self.tokenizer.bos_id_f_c, dtype=torch.long, device=self.device)
            bos_d_c = torch.full((bs, 1), self.tokenizer.bos_id_d_c, dtype=torch.long, device=self.device)
            for i in range(self.refine_iters):
                # Prior context is the previous output. remove the last grp as it is for eos
                h_c_ctx = [torch.cat([bos_h_c, h_c_logits[:, :-1].argmax(-1)], dim= 1) for h_c_logits in logits[:self.num_h_c]]
                f_c_ctx = torch.cat([bos_f_c, logits[2][:, :-1].argmax(-1)], dim= 1)
                vals, indices = torch.topk(logits[-1][:, :-1], k=self.num_d_c)
                d_c_ctx = [torch.cat([bos_d_c, indices[:,:,i]], dim= -1) for i in range(self.num_d_c)]

                # ctx_padding_mask = ((tgt_in == self.eos_id).int().cumsum(-1) > 0)  # mask tokens beyond the first EOS token.
                h_c_ctx_pad = sum([(ctx == self.tokenizer.eos_id).int().cumsum(-1) for ctx in h_c_ctx])
                f_c_ctx_pad = (f_c_ctx == self.tokenizer.eos_id).int().cumsum(-1)
                d_c_ctx_pad = sum([(ctx == self.tokenizer.eos_id).int().cumsum(-1) for ctx in d_c_ctx])
                ctx_padding_mask = (h_c_ctx_pad + f_c_ctx_pad + d_c_ctx_pad) > 0
                tgt_out = self.decode(query=pos_queries, h_c_ctx= h_c_ctx, f_c_ctx= f_c_ctx,
                                        d_c_ctx= d_c_ctx, memory= memory, query_mask=query_mask[:, :f_c_ctx.shape[1]],
                                        context_key_padding_mask= ctx_padding_mask,
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
        mask = torch.zeros((sz, sz), device=self.device, dtype= torch.bool)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = True
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self.device)] = True  # mask "self"
        query_mask = mask[1:, :-1]
        return content_mask, query_mask
    
    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        batch_size = len(labels)
        h_c_2_targets, h_c_1_targets, f_c_targets, d_c_targets = ( # +2 for eos and bos
                                  torch.zeros(batch_size, self.max_grps + 2, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps + 2, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps + 2, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps + 2, self.num_d_classes, device= self.device), # we don't need PAD
                                )
        
        n_grps = [self.max_grps + 2 for _ in range(batch_size)]
        for idx,label in enumerate(labels, start= 0):
            h_c_2_targets[idx], h_c_1_targets[idx], f_c_targets[idx], d_c_targets[idx], n_grps[idx] = self.tokenizer.label_encoder(label, device= self.device)
        
        # Truncate grps to largest in batch
        num_grps = max(n_grps)
        # Encode the source sequence (i.e. the image codes)
        memory = self.encoder(imgs)

        # Prepare the target sequences (input and output)
        tgt_perms = self.gen_tgt_perms(f_c_targets[:,:num_grps])
        # ignore BOS tag
        (h_c_2_out, h_c_1_out, f_c_out, d_c_out) = (h_c_2_targets[:, 1:num_grps], h_c_1_targets[:, 1:num_grps], 
                                                                    f_c_targets[:, 1:num_grps], d_c_targets[:, 1:num_grps]) # we dont predict PAD or BOS
        (h_c_2_in, h_c_1_in, f_c_in, d_c_in) = (h_c_2_targets[:, :num_grps -1], h_c_1_targets[:, :num_grps -1], 
                                                                    f_c_targets[:, :num_grps -1], d_c_targets[:, :num_grps -1])
        # The [EOS] token is not depended upon by any other token in any permutation ordering
        # pads and eos are same accross char grps
        ctx_padding_mask = (f_c_in == self.tokenizer.pad_id_f_c) | (f_c_in == self.tokenizer.eos_id) 

        loss = 0
        loss_numel = 0
        n = (f_c_targets != self.tokenizer.pad_id_f_c).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm)
            vals, indices = torch.topk(d_c_in, k= self.num_d_c)
            out = self.decode(h_c_ctx= [h_c_2_in, h_c_1_in], f_c_ctx= f_c_in, 
                              d_c_ctx= [indices[:,:,i] for i in range(self.num_d_c)], # ignore the additional target for EOS
                              memory= memory, query_mask=query_mask, context_key_padding_mask= ctx_padding_mask)
            (h_c_2_logits, h_c_1_logits, f_c_logits, d_c_logits) = self.classifier(out)
            # loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=self.pad_id)
            # Get the flattened versions of the targets and the logits for grp level metrics
            ((flat_h_c_2_targets, flat_h_c_1_targets, flat_f_c_targets, flat_d_c_targets), 
            (flat_h_c_2_logits, flat_h_c_1_logits, flat_f_c_logits, flat_d_c_logits)) = self._get_flattened_non_pad(
                                                                                    targets= (h_c_2_out, h_c_1_out, f_c_out, d_c_out),
                                                                                    logits= (h_c_2_logits, h_c_1_logits, f_c_logits, d_c_logits),
                                                                                )
            loss += n * (self.h_c_2_loss(input= flat_h_c_2_logits, target= flat_h_c_2_targets) \
            + self.h_c_1_loss(input= flat_h_c_1_logits, target= flat_h_c_1_targets) \
            + self.f_c_loss(input= flat_f_c_logits, target= flat_f_c_targets) \
            + self.d_loss(input= flat_d_c_logits, target= flat_d_c_targets))
            loss_numel += n
            # After the second iteration (i.e. done with canonical and reverse orderings),
            # remove the [EOS] tokens for the succeeding perms
            if i == 1:
                h_c_2_out = torch.where(h_c_2_out == self.tokenizer.eos_id, self.tokenizer.pad_id_h_c, h_c_2_out)
                h_c_1_out = torch.where(h_c_1_out == self.tokenizer.eos_id, self.tokenizer.pad_id_h_c, h_c_1_out)
                f_c_out = torch.where(f_c_out == self.tokenizer.eos_id, self.tokenizer.pad_id_f_c, f_c_out)
                d_pad, d_eos = [torch.zeros(self.num_d_classes, device= self.device) for _ in range(2)]
                d_pad[self.tokenizer.pad_id_d_c] = d_eos[self.tokenizer.eos_id] = 1.
                d_c_eos = torch.all(d_c_out == d_eos, dim= -1)
                d_c_out[d_c_eos] = d_pad
                n = (f_c_out != self.tokenizer.pad_id_f_c).sum().item()
        loss /= loss_numel

        self.log('train_loss_step', loss, prog_bar= True, on_step= True, on_epoch= False, logger = True, sync_dist = True, batch_size= batch_size)
        self.log('train_loss_epoch', loss, prog_bar= False, on_step= False, on_epoch= True, logger = True, sync_dist = True, batch_size= batch_size)
        return loss

    def validation_step(self, batch, batch_no)-> None:
        # batch: img (BS x C x H x W), label (BS)
        imgs, labels = batch
        batch_size = len(labels)
        h_c_2_targets, h_c_1_targets, f_c_targets, d_c_targets = (
                                  torch.zeros(batch_size, self.max_grps + 2, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps + 2, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps + 2, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps + 2, self.num_d_classes, device= self.device),
                                )
        n_grps = [self.max_grps for i in range(batch_size)]
        for idx,label in enumerate(labels, start= 0):
            h_c_2_targets[idx], h_c_1_targets[idx], f_c_targets[idx], d_c_targets[idx], n_grps[idx] = self.tokenizer.label_encoder(label, device= self.device)
        # ignore BOS
        (h_c_2_out, h_c_1_out, f_c_out, d_c_out) = (h_c_2_targets[:, 1:], h_c_1_targets[:, 1:], 
                                                                    f_c_targets[:, 1:], d_c_targets[:, 1:]) # dont predict PAD or BOS

        (h_c_2_logits, h_c_1_logits, f_c_logits, d_c_logits) = self.forward(imgs)

        # Get the flattened versions of the targets and the logits for grp level metrics
        ((flat_h_c_2_targets, flat_h_c_1_targets, flat_f_c_targets, flat_d_targets), 
        (flat_h_c_2_logits, flat_h_c_1_logits, flat_f_c_logits, flat_d_c_logits)) = self._get_flattened_non_pad(
                                                                                targets= (h_c_2_out, h_c_1_out, f_c_out, d_c_out),
                                                                                logits= (h_c_2_logits, h_c_1_logits, f_c_logits, d_c_logits),
                                                                            )

        # compute the loss for each character category
        loss = self.h_c_2_loss(input= flat_h_c_2_logits, target= flat_h_c_2_targets) \
            + self.h_c_1_loss(input= flat_h_c_1_logits, target= flat_h_c_1_targets) \
            + self.f_c_loss(input= flat_f_c_logits, target= flat_f_c_targets) \
            + self.d_loss(input= flat_d_c_logits, target= flat_d_targets)
        # Grp level metrics
        self.val_h_c_2_acc(flat_h_c_2_logits, flat_h_c_2_targets)
        self.val_h_c_1_acc(flat_h_c_1_logits, flat_h_c_1_targets)
        self.val_comb_h_c_acc((flat_h_c_2_logits, flat_h_c_1_logits),\
                               (flat_h_c_2_targets, flat_h_c_1_targets))
        self.val_f_c_acc(flat_f_c_logits, flat_f_c_targets)
        self.val_d_acc(flat_d_c_logits, flat_d_targets)
        self.val_grp_acc(([flat_h_c_2_logits, flat_h_c_1_logits], flat_f_c_logits, flat_d_c_logits),\
                           ([flat_h_c_2_targets, flat_h_c_1_targets], flat_f_c_targets, flat_d_targets))
        # Word level metric
        self.val_wrr2(([h_c_2_logits, h_c_1_logits], f_c_logits, d_c_logits),\
                     ([h_c_2_out, h_c_1_out], f_c_out, d_c_out[:,:,:-2]), self.tokenizer.pad_id_f_c)
        # self.val_wrr(pred_strs= self.tokenizer.decode((h_c_2_logits, h_c_1_logits, f_c_logits, d_logits)), target_strs= labels)
        
        if batch_no % 100000 == 0:
            pred_labels = self.tokenizer.decode((h_c_2_logits, h_c_1_logits, f_c_logits, d_c_logits))            
            self._log_tb_images(imgs[:5], pred_labels= pred_labels[:5], gt_labels= labels[:5], mode= "val")

        # On epoch only logs
        log_dict_epoch = {
            "val_loss": loss,
            "val_half_character2_acc": self.val_h_c_2_acc,
            "val_half_character1_acc": self.val_h_c_1_acc,
            "val_combined_half_character_acc": self.val_comb_h_c_acc,
            "val_character_acc": self.val_f_c_acc,
            "val_diacritic_acc": self.val_d_acc,
            "val_wrr2": self.val_wrr2, 
            "val_grp_acc": self.val_grp_acc,
        }
        self.log_dict(log_dict_epoch, on_step = False, on_epoch = True, prog_bar = False, logger = True, sync_dist = True, batch_size= batch_size)

    def test_step(self, batch, batch_no)-> None:
        # batch: img (BS x C x H x W), label (BS)
        imgs, labels = batch
        batch_size = len(labels)
        h_c_2_targets, h_c_1_targets, f_c_targets, d_c_targets = (
                                  torch.zeros(batch_size, self.max_grps + 2, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps + 2, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps + 2, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps + 2, self.num_d_classes, device= self.device),
                                )
        n_grps = [self.max_grps for i in range(batch_size)]
        for idx,label in enumerate(labels, start= 0):
            h_c_2_targets[idx], h_c_1_targets[idx], f_c_targets[idx], d_c_targets[idx], n_grps[idx] = self.tokenizer.label_encoder(label, device= self.device)
        
        (h_c_2_out, h_c_1_out, f_c_out, d_c_out) = (h_c_2_targets[:, 1:], h_c_1_targets[:, 1:], 
                                                                    f_c_targets[:, 1:], d_c_targets[:, 1:]) # dont predict PAD or BOS
        (h_c_2_logits, h_c_1_logits, f_c_logits, d_c_logits) = self.forward(imgs)

        # Get the flattened versions of the targets and the logits for grp level metrics
        ((flat_h_c_2_targets, flat_h_c_1_targets, flat_f_c_targets, flat_d_targets), 
        (flat_h_c_2_logits, flat_h_c_1_logits, flat_f_c_logits, flat_d_c_logits)) = self._get_flattened_non_pad(
                                                                                targets= (h_c_2_out, h_c_1_out, f_c_out, d_c_out),
                                                                                logits= (h_c_2_logits, h_c_1_logits, f_c_logits, d_c_logits),
                                                                            )

        # compute the loss for each group
        loss = self.h_c_2_loss(input= flat_h_c_2_logits, target= flat_h_c_2_targets) \
            + self.h_c_1_loss(input= flat_h_c_1_logits, target= flat_h_c_1_targets) \
            + self.f_c_loss(input= flat_f_c_logits, target= flat_f_c_targets) \
            + self.d_loss(input= flat_d_c_logits, target= flat_d_targets)
        
        # Grp level metrics
        self.test_h_c_2_acc(flat_h_c_2_logits, flat_h_c_2_targets)
        self.test_h_c_1_acc(flat_h_c_1_logits, flat_h_c_1_targets)
        self.test_comb_h_c_acc((flat_h_c_2_logits, flat_h_c_1_logits),\
                                (flat_h_c_2_targets, flat_h_c_1_targets))
        self.test_f_c_acc(flat_f_c_logits, flat_f_c_targets)
        self.test_d_acc(flat_d_c_logits, flat_d_targets)
        self.test_grp_acc(([flat_h_c_2_logits, flat_h_c_1_logits], flat_f_c_logits, flat_d_c_logits),\
                           ([flat_h_c_2_targets, flat_h_c_1_targets], flat_f_c_targets, flat_d_targets))
        
        # Word level metric
        self.test_wrr2(([h_c_2_logits, h_c_1_logits], f_c_logits, d_c_logits),\
                      ([h_c_2_out, h_c_1_out], f_c_out, d_c_out[:,:,:-2]), self.tokenizer.pad_id_f_c)
        pred_labels= self.tokenizer.decode((h_c_2_logits, h_c_1_logits, f_c_logits, d_c_logits))
        self.test_wrr(pred_strs= pred_labels, target_strs= labels)        
        self.ned(pred_labels= pred_labels, target_labels= labels)
        self._log_tb_images(imgs, pred_labels= pred_labels, gt_labels= labels, mode= "test")
            
        # On epoch only logs
        log_dict_epoch = {
            "test_loss": loss,
            "test_half_character2_acc": self.test_h_c_2_acc,
            "test_half_character1_acc": self.test_h_c_1_acc,
            # "test_combined_half_character_acc": self.test_comb_h_c_acc,
            "test_character_acc": self.test_f_c_acc,
            "test_diacritic_acc": self.test_d_acc,
            "test_wrr": self.test_wrr, 
            "test_wrr2": self.test_wrr2,
            "test_grp_acc": self.test_grp_acc,
            "NED": self.ned,
        }
        self.log_dict(log_dict_epoch, on_step = False, on_epoch = True, prog_bar = False, logger = True, sync_dist = True, batch_size= batch_size)

class ViTPARSeq(PARSeqBaseSystem):
    def __init__(self, hidden_size: int = 768, num_hidden_layers: int = 12, num_attention_heads: int = 12,
                 mlp_ratio: float= 4.0, dropout:float= 0.0, initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12, image_size: int = 128, patch_size: int = 8, 
                 num_channels: int = 3, dec_num_sa_heads:Sequence[Union[Sequence[int], int]] = [[2,2], 4, [2,2]], 
                 dec_num_ca_heads:int= 12, dec_mlp_ratio: int= 4, dec_depth: int= 1, perm_num:int= 25, 
                 perm_forward: bool= True, perm_mirrored: bool= True, decode_ar: bool= True,
                 refine_iters: int= 1, num_h_c:int= 2, num_d_c:int= 2, tokenizer:str= 'HindiPARSeqTokenizer',
                 threshold:float= 0.5, max_grps:int = 25, learning_rate: float= 1e-4, 
                 weight_decay: float= 1.0e-4, warmup_pct:float= 0.3,
                 ) -> None:
        self.save_hyperparameters()
        super().__init__(hidden_size= hidden_size, num_h_c= num_h_c, num_d_c= num_d_c, tokenizer= tokenizer, 
                 dec_num_sa_heads= dec_num_sa_heads, dec_num_ca_heads= dec_num_ca_heads,
                 dec_mlp_ratio= dec_mlp_ratio, dec_depth= dec_depth, perm_num= perm_num, 
                 perm_forward= perm_forward, perm_mirrored= perm_mirrored, decode_ar= decode_ar,
                 refine_iters= refine_iters, dropout= dropout, threshold= threshold, max_grps = max_grps,
                 learning_rate= learning_rate, weight_decay= weight_decay, warmup_pct= warmup_pct)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.intermediate_size = int(mlp_ratio * hidden_size)
        self.encoder = ViTEncoder(
            hidden_size= self.hidden_size,
            num_hidden_layers= self.num_hidden_layers,
            num_attention_heads= self.num_attention_heads,
            intermediate_size= self.intermediate_size,
            hidden_act= "gelu",
            hidden_dropout_prob= self.dropout,
            attention_probs_dropout_prob= self.dropout,
            initializer_range= self.initializer_range,
            layer_norm_eps= self.layer_norm_eps,
            image_size= self.image_size,
            patch_size= self.patch_size,
            num_channels= self.num_channels,
            qkv_bias= True,
        )

class FocalPARSeq(PARSeqBaseSystem):
    def __init__(self, embed_dim: int = 96, depths:list= [2, 2, 6, 2],
                 focal_levels:list= [2, 2, 2, 2], focal_windows:list= [3, 3, 3, 3],
                 mlp_ratio: float= 4.0, drop_path_rate:float = 0.1, dropout:float= 0.1,
                 initializer_range: float = 0.02, layer_norm_eps: float = 1e-12, image_size: int = 128,
                 patch_size: int = 8, num_channels: int = 3, 
                 dec_num_sa_heads:Sequence[Union[Sequence[int], int]] = [[2,2], 4, [2,2]], 
                 dec_num_ca_heads:int= 12, dec_mlp_ratio: int= 4, dec_depth: int= 1, perm_num:int= 25, 
                 perm_forward: bool= True, perm_mirrored: bool= True, decode_ar: bool= True,
                 refine_iters: int= 1, num_h_c:int= 2, num_d_c:int= 2, tokenizer:str= 'HindiPARSeqTokenizer',
                 threshold:float= 0.5, max_grps:int = 25, learning_rate: float= 1e-4, 
                 weight_decay: float= 1.0e-4, warmup_pct:float= 0.3,
                 ) -> None:
        self.save_hyperparameters()
        self.embed_dim = embed_dim
        self.hidden_sizes = [self.embed_dim * (2 ** i) for i in range(len(depths))] 
        super().__init__(hidden_size= self.hidden_sizes[-1], num_h_c= num_h_c, num_d_c= num_d_c, tokenizer= tokenizer, 
                 dec_num_sa_heads= dec_num_sa_heads, dec_num_ca_heads= dec_num_ca_heads,
                 dec_mlp_ratio= dec_mlp_ratio, dec_depth= dec_depth, perm_num= perm_num, 
                 perm_forward= perm_forward, perm_mirrored= perm_mirrored, decode_ar= decode_ar,
                 refine_iters= refine_iters, dropout= dropout, threshold= threshold, max_grps = max_grps,
                 learning_rate= learning_rate, weight_decay= weight_decay, warmup_pct= warmup_pct)

        self.encoder = FocalNetEncoder(
                        hidden_dropout_prob= dropout, 
                        initializer_range = initializer_range,
                        image_size= image_size, 
                        patch_size= patch_size, 
                        num_channels = num_channels,
                        embed_dim= embed_dim,
                        hidden_sizes = self.hidden_sizes, 
                        depths = depths,
                        focal_levels= focal_levels,
                        focal_windows= focal_windows,
                        mlp_ratio= mlp_ratio,
                        drop_path_rate= drop_path_rate,
                        layer_norm_eps= layer_norm_eps,
                    )