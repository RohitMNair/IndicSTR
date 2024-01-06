from lightning.pytorch.utilities.types import STEP_OUTPUT
from encoder import ViTEncoder
from decoder import GroupDecoder
from utils.metrics import (DiacriticAccuracy, FullCharacterAccuracy, CharGrpAccuracy,
                   HalfCharacterAccuracy, CombinedHalfCharAccuracy, WRR)
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Tuple
from torch import Tensor
from utils.transforms import LabelTransform

import lightning.pytorch as pl
import torch
import torch.nn as nn

class GrpClassifier(pl.LightningModule):
    def __init__(self, hidden_size:int, half_character_classes:list,
                 full_character_classes:list, diacritic_classes:list,):
        super().__init__()
        self.hidden_size= hidden_size
        self.h_c_classes = half_character_classes
        self.f_c_classes = full_character_classes
        self.d_classes = diacritic_classes
        self.half_character2_head = nn.Linear(
                                in_features = self.hidden_size,
                                out_features = len(self.h_c_classes) + 1, # extra node for no half-char
                                bias = True
                            )
        self.half_character1_head = nn.Linear(
                                in_features = self.hidden_size,
                                out_features = len(self.h_c_classes) + 1,
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
    def __init__(self, emb_path:str, half_character_classes:list, full_character_classes:list,
                 diacritic_classes:list, halfer:str, hidden_size: int = 768, 
                 num_hidden_layers: int = 12, num_attention_heads: int = 12,
                 mlp_ratio: float= 4.0, hidden_act: str = "gelu", hidden_dropout_prob: float = 0.0,
                 attention_probs_dropout_prob: float = 0.0, initializer_range: float = 0.02,
                 layer_norm_eps: float = 1e-12, image_size: int = 224, patch_size: int = 16, 
                 num_channels: int = 3, qkv_bias: bool = True, max_grps: int = 25, threshold:float= 0.5,
                 learning_rate: float= 1e-4, weight_decay: float= 1.0e-4, warmup_pct:float= 0.3
                 ):
        super().__init__()
        self.emb_path = emb_path
        self.h_c_classes = ["[B]"] + half_character_classes
        self.f_c_classes = ["[B]"] + full_character_classes
        self.d_classes = diacritic_classes
        self.halfer = halfer
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
        self.threshold = threshold
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.pct_start = warmup_pct

        # non parameteric attributes
        self.h_c_2_emb, self.h_c_1_emb, self.f_c_emb, self.d_emb = self._extract_char_embeddings()
        self.intermediate_size = int(self.mlp_ratio * self.hidden_size)
        self.h_c_set = set(self.h_c_classes)
        self.f_c_set = set(self.f_c_classes)
        self.d_c_set = set(self.d_classes)
        # dict with class indexes as keys and characters as values
        self.h_c_label_map = {k:c for k,c in enumerate(self.h_c_classes, start = 1)}
        # 0 will be reserved for blank
        self.f_c_label_map = {k:c for k,c in enumerate(self.f_c_classes, start = 1)}
        # blank not needed for embedding as it is Binary classification of each diacritic
        self.d_c_label_map = {k:c for k,c in enumerate(self.d_classes, start = 0)}
        
        # dict with characters as keys and class indexes as values
        self.rev_half_char_label_map = {c:k for k,c in enumerate(self.h_c_classes, start = 1)}
        self.rev_char_label_map = {c:k for k,c in enumerate(self.f_c_classes, start = 1)}
        self.rev_diacritirc_label_map = {c:k for k,c in enumerate(self.d_classes, start = 0)}

        # initializations
        self.label_transform = LabelTransform(
            half_character_set= self.h_c_set,
            full_character_set= self.f_c_set, 
            diacritic_set= self.d_c_set, 
            halfer= self.halfer,
        )
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
            half_character_classes= self.h_c_classes,
            full_character_classes= self.f_c_classes,
            diacritic_classes= self.d_classes,
        )
       
        self.h_c_2_loss = nn.CrossEntropyLoss(reduction= 'mean')
        self.f_c_loss = nn.CrossEntropyLoss(reduction= 'mean')
        self.h_c_1_loss = nn.CrossEntropyLoss(reduction= 'mean')
        self.d_loss = nn.BCEWithLogitsLoss(reduction= 'mean')

        # Trainig Metrics
        self.train_h_c_2_acc = HalfCharacterAccuracy(threshold = self.threshold)
        self.train_h_c_1_acc = HalfCharacterAccuracy(threshold = self.threshold)
        self.train_comb_h_c_acc = CombinedHalfCharAccuracy(threshold= self.threshold)
        self.train_f_c_acc = FullCharacterAccuracy(threshold = self.threshold)
        self.train_d_acc = DiacriticAccuracy(threshold = self.threshold)
        self.train_grp_acc = CharGrpAccuracy(threshold= self.threshold)
        self.train_wrr = WRR(threshold= self.threshold)
        # Validation Metrics
        self.val_h_c_1_acc = HalfCharacterAccuracy(threshold = self.threshold)
        self.val_h_c_2_acc = HalfCharacterAccuracy(threshold = self.threshold)
        self.val_comb_h_c_acc = CombinedHalfCharAccuracy(threshold= self.threshold)
        self.val_f_c_acc = FullCharacterAccuracy(threshold = self.threshold)
        self.val_d_acc = DiacriticAccuracy(threshold = self.threshold)
        self.val_grp_acc = CharGrpAccuracy(threshold= self.threshold)
        self.val_wrr = WRR(threshold= self.threshold)
        # Testing Metrics
        self.test_h_c_1_acc = HalfCharacterAccuracy(threshold = self.threshold)
        self.test_h_c_2_acc = HalfCharacterAccuracy(threshold = self.threshold)
        self.test_comb_h_c_acc = CombinedHalfCharAccuracy(threshold= self.threshold)
        self.test_f_c_acc = FullCharacterAccuracy(threshold = self.threshold)
        self.test_d_acc = DiacriticAccuracy(threshold = self.threshold)
        self.test_grp_acc = CharGrpAccuracy(threshold= self.threshold)
        self.test_wrr = WRR(threshold= self.threshold)
    
    def _extract_char_embeddings(self)-> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Extracts the character embeddings from Img2Vec checkpoint
        Returns:
        - tuple(Tensor, Tensor, Tensor, Tensor): half-char 2, half-char 1, full-char
                                                and diacritic embeddings from checkpoint
        """
        loaded_dict = torch.load(self.emb_path)

        assert list(loaded_dict["h_c_classes"]) == self.h_c_classes,\
              "Embedding Half-character classes and model half-character classes do not match"
        assert list(loaded_dict["f_c_classes"]) == self.f_c_classes,\
              "Embedding Full-character classes and model Full-character classes do not match"
        assert list(loaded_dict["d_classes"]) == self.d_classes, \
              "Embedding diacritic classes and model diacritic classes do not match"
        
        return loaded_dict["h_c_2_emb"], loaded_dict["h_c_1_emb"], loaded_dict["f_c_emb"], loaded_dict["d_emb"]


    def grp_class_encoder(self, grp: str)-> Tuple[int, int, int, Tensor]:
        """
        Encodes the group into class labels for Half-character 2, 1 and full character;
        One-Hot encodes diacritics.
        Args:
        - grp (str): Character group

        Returns:
        - tuple(Tensor, Tensor, Tensor, Tensor): class label for half-character 2,
                                                half-character 1, full-character,
                                                and diacritics (one hot encoded)
        """
        # get the character groupings
        h_c_2_target, h_c_1_target, f_c_target, d_target = 0, 0, 0, torch.tensor([0. for i in range(len(self.d_classes))])
        for i, char in enumerate(grp):
            halfer_cntr = 0
            if char in self.h_c_set and i + 1 < len(grp) and grp[i+1] == self.halfer:
                    # for half character occurence
                    if h_c_1_target == 0:
                        # half_character1 will always track the half-character
                        # which is joined with the complete character
                        h_c_1_target = self.rev_half_char_label_map[char]
                    else: # there are more than 1 half-characters
                        # half_character2 will keep track of half-character 
                        # not joined with the complete character
                        h_c_2_target = h_c_1_target
                        # the first half-char will be joined to the complete character
                        h_c_1_target = self.rev_half_char_label_map[char]

            elif char in self.f_c_set:
                assert f_c_target == 0,\
                       f"2 full Characters have occured {grp}-{char}"
                f_c_target = self.rev_char_label_map[char]

            elif char in self.d_c_set:
                # for diacritic occurence
                assert d_target[self.rev_diacritirc_label_map[char]] == 0.0, \
                    f"2 same matras occured {grp}-{char}-{d_target}"
                
                d_target[self.rev_diacritirc_label_map[char]] = 1.

            elif char == self.halfer and halfer_cntr < 2:
                halfer_cntr += 1

            elif char == self.halfer and halfer_cntr >=2 :
                raise Exception(f"More than 2 half-characters occured")

            else:
                raise Exception(f"Character {char} not found in vocabulary")
        
        assert torch.sum(d_target, dim = -1) <= 2, f"More than 2 diacritics occured {grp}-{d_target}"

        return h_c_2_target, h_c_1_target, f_c_target, d_target
    
    def label_encoder(self, label:str)-> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Converts the text label into classes indexes for classification
        Args:
        - label (str): The label to be encoded

        Returns:
        - tuple(int, int, int, Tensor): half-char 2 class index, half-char 1
                                        class index, full char class index,
                                        diacritic one hot encoding
        """
        grps = self.label_transform.hindi_label_transform(label= label)
        h_c_2_target, h_c_1_target, f_c_target, d_target = (
                                                            torch.zeros(self.max_grps), 
                                                            torch.zeros(self.max_grps),
                                                            torch.zeros(self.max_grps),
                                                            torch.zeros(self.max_grps, len(self.d_classes))
                                                        )
        # truncate grps if grps exceed max_grps
        if len(grps) > self.max_grps:
            grps = grps[:self.max_grps]

        for idx,grp in enumerate(grps, start= 0):
            h_c_2_target[idx], h_c_1_target[idx], f_c_target[idx], d_target[idx] = self.grp_class_encoder(grp=grp)

        return h_c_2_target, h_c_1_target, f_c_target, d_target
    
    def forward(self, x:torch.Tensor)-> Tuple[Tuple[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]] :
        batch_size = x.shape[0]
        enc_x = self.encoder(x)
        dec_x, pos_vis_attn_weights, chr_grp_attn_weights = self.decoder(enc_x)
        h_c_2_logits, h_c_1_logits, f_c_logits, d_logits = self.classifier(dec_x.view(-1, self.hidden_size))
        return (
            h_c_2_logits.view(batch_size, self.max_grps, len(self.h_c_classes)),
            h_c_1_logits.view(batch_size, self.max_grps, len(self.h_c_classes)),
            f_c_logits.view(batch_size, self.max_grps, len(self.f_c_classes)),
            d_logits.view(batch_size, self.max_grps, len(self.d_classes)),
        ), (pos_vis_attn_weights, chr_grp_attn_weights)

    def training_step(self, batch, batch_no)-> Tensor:
        # batch: img Tensor(BS x C x H x W), label tuple(BS)
        imgs, labels = batch
        batch_size = imgs.shape[0]
        h_c_2_targets, h_c_1_targets, f_c_targets, d_targets = (
                                  torch.zeros(batch_size, self.max_grps, len(self.h_c_classes)),
                                  torch.zeros(batch_size, self.max_grps, len(self.h_c_classes)),
                                  torch.zeros(batch_size, self.max_grps, len(self.f_c_classes)),
                                  torch.zeros(batch_size, self.max_grps, len(self.d_classes)),
                                )
        for idx,label in enumerate(labels, start= 0):
            h_c_2_targets[idx], h_c_1_targets[idx], f_c_targets[idx], d_targets[idx] = self.label_encoder(label)

        (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits) = self.forward(imgs)[0]

        # get a flattened copy for grp level metrics
        flat_h_c_2_logits = h_c_2_logits.view(-1, self.hidden_size)
        flat_h_c_1_logits = h_c_1_logits.view(-1, self.hidden_size)
        flat_f_c_logits = f_c_logits.view(-1, self.hidden_size)
        flat_d_logits = d_logits.view(-1, self.hidden_size)

        flat_h_c_2_targets = h_c_2_targets.view(-1,1)
        flat_h_c_1_targets = h_c_1_targets.view(-1, 1)
        flat_f_c_targets = f_c_targets.view(-1, 1)
        flat_d_targets = d_targets.view(-1, 1)

        # compute the loss for each group
        loss = self.h_c_2_loss(input= flat_h_c_2_logits, target= flat_h_c_2_targets) \
            + self.h_c_1_loss(input= flat_h_c_1_logits, target= flat_h_c_1_targets) \
            + self.f_c_loss(input= flat_f_c_logits, target= flat_f_c_targets) \
            + self.d_loss(input= flat_d_logits, target= flat_d_targets)
        # Grp level metrics
        self.train_h_c_2_acc(flat_h_c_2_logits, flat_h_c_2_targets)
        self.train_h_c_1_acc(flat_h_c_1_logits, flat_h_c_1_targets)
        self.train_comb_h_c_acc((flat_h_c_2_logits, flat_h_c_1_logits),\
                                 (flat_h_c_2_targets, flat_h_c_1_targets))
        self.train_f_c_acc(flat_f_c_logits, flat_f_c_targets)
        self.train_d_acc(flat_d_logits, flat_d_targets)
        self.train_grp_acc((flat_h_c_2_logits, flat_h_c_1_logits, flat_f_c_logits, flat_d_logits),\
                           (flat_h_c_2_targets, flat_h_c_1_targets, flat_f_c_targets, flat_d_targets))
        # Word level metric
        self.train_wrr((h_c_2_logits, h_c_1_logits, f_c_logits, d_logits),\
                       (h_c_2_targets, h_c_1_targets, f_c_targets, d_targets))

        # On step logs for proggress bar display
        log_dict_step = {
            "train_loss_step": loss,
            "train_wrr_step": self.train_wrr,
            "train_grp_acc_step": self.train_grp_acc,
        }
        self.log_dict(log_dict_step, on_step = True, on_epoch = False, prog_bar = True, logger = True, sync_dist=True)

        # On epoch only logs
        log_dict_epoch = {
            "train_loss_epoch": loss,
            "train_half_character2_acc": self.train_h_c_2_acc,
            "train_half_character1_acc": self.train_h_c_1_acc,
            "train_combined_half_character_acc": self.train_comb_h_c_acc,
            "train_character_acc": self.train_f_c_acc,
            "train_diacritic_acc": self.train_d_acc,
            "train_wrr_epoch": self.train_wrr, 
            "train_grp_acc_epoch": self.train_grp_acc,
        }
        self.log_dict(log_dict_epoch, on_step = False, on_epoch = True, prog_bar = False, logger = True, sync_dist = True)  
        return loss

    def configure_optimizers(self)-> dict:
        optmizer = AdamW(params= self.parameters(), lr= self.lr, weight_decay= self.weight_decay)
        lr_scheduler = OneCycleLR(
            optimizer= optmizer,
            max_lr= self.lr,
            total_steps= int(self.trainer.estimated_stepping_batches), # gets the max training steps
            pct_start= self.pct_start,
            cycle_momentum= False,
        )
        return {
            'optimizer': optmizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',
            }
        }

    def validation_step(self, batch, batch_idx)-> None:
        # batch: img (BS x C x H x W), label (BS)
        imgs, labels = batch
        batch_size = imgs.shape[0]
        h_c_2_targets, h_c_1_targets, f_c_targets, d_targets = (
                                  torch.zeros(batch_size, self.max_grps, len(self.h_c_classes)),
                                  torch.zeros(batch_size, self.max_grps, len(self.h_c_classes)),
                                  torch.zeros(batch_size, self.max_grps, len(self.f_c_classes)),
                                  torch.zeros(batch_size, self.max_grps, len(self.d_classes)),
                                )
        for idx,label in enumerate(labels, start= 0):
            h_c_2_targets[idx], h_c_1_targets[idx], f_c_targets[idx], d_targets[idx] = self.label_encoder(label)

        (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits) = self.forward(imgs)[0]

        # get a flattened copy for grp level metrics
        flat_h_c_2_logits = h_c_2_logits.view(-1, self.hidden_size)
        flat_h_c_1_logits = h_c_1_logits.view(-1, self.hidden_size)
        flat_f_c_logits = f_c_logits.view(-1, self.hidden_size)
        flat_d_logits = d_logits.view(-1, self.hidden_size)

        flat_h_c_2_targets = h_c_2_targets.view(-1,1)
        flat_h_c_1_targets = h_c_1_targets.view(-1, 1)
        flat_f_c_targets = f_c_targets.view(-1, 1)
        flat_d_targets = d_targets.view(-1, 1)

        # compute the loss for each group
        loss = self.h_c_2_loss(input= flat_h_c_2_logits, target= flat_h_c_2_targets) \
            + self.h_c_1_loss(input= flat_h_c_1_logits, target= flat_h_c_1_targets) \
            + self.f_c_loss(input= flat_f_c_logits, target= flat_f_c_targets) \
            + self.d_loss(input= flat_d_logits, target= flat_d_targets)
        # Grp level metrics
        self.val_h_c_2_acc(flat_h_c_2_logits, flat_h_c_2_targets)
        self.val_h_c_1_acc(flat_h_c_1_logits, flat_h_c_1_targets)
        self.val_comb_h_c_acc((flat_h_c_2_logits, flat_h_c_1_logits),\
                               (h_c_2_targets, h_c_1_targets))
        self.val_f_c_acc(flat_f_c_logits, flat_f_c_targets)
        self.val_d_acc(flat_d_logits, flat_d_targets)
        self.val_grp_acc((flat_h_c_2_logits, flat_h_c_1_logits, flat_f_c_logits, flat_d_logits),\
                           (flat_h_c_2_targets, flat_h_c_1_targets, flat_f_c_targets, flat_d_targets))
        # Word level metric
        self.val_wrr((h_c_2_logits, h_c_1_logits, f_c_logits, d_logits),\
                     (h_c_2_targets, h_c_1_targets, f_c_targets, d_targets))

        # On epoch only logs
        log_dict_epoch = {
            "val_loss_epoch": loss,
            "val_half_character2_acc": self.train_h_c_2_acc,
            "val_half_character1_acc": self.train_h_c_1_acc,
            "val_combined_half_character_acc": self.train_comb_h_c_acc,
            "val_character_acc": self.train_f_c_acc,
            "val_diacritic_acc": self.train_d_acc,
            "val_wrr_epoch": self.train_wrr, 
            "val_grp_acc_epoch": self.train_grp_acc,
        }
        self.log_dict(log_dict_epoch, on_step = False, on_epoch = True, prog_bar = False, logger = True, sync_dist = True)

    def test_step(self, batch, batch_idx)-> None:
        # batch: img (BS x C x H x W), label (BS)
        imgs, labels = batch
        batch_size = imgs.shape[0]
        h_c_2_targets, h_c_1_targets, f_c_targets, d_targets = (
                                  torch.zeros(batch_size, self.max_grps, len(self.h_c_classes)),
                                  torch.zeros(batch_size, self.max_grps, len(self.h_c_classes)),
                                  torch.zeros(batch_size, self.max_grps, len(self.f_c_classes)),
                                  torch.zeros(batch_size, self.max_grps, len(self.d_classes)),
                                )
        for idx,label in enumerate(labels, start= 0):
            h_c_2_targets[idx], h_c_1_targets[idx], f_c_targets[idx], d_targets[idx] = self.label_encoder(label)

        (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits) = self.forward(imgs)[0]

        # get a flattened copy for grp level metrics
        flat_h_c_2_logits = h_c_2_logits.view(-1, self.hidden_size)
        flat_h_c_1_logits = h_c_1_logits.view(-1, self.hidden_size)
        flat_f_c_logits = f_c_logits.view(-1, self.hidden_size)
        flat_d_logits = d_logits.view(-1, self.hidden_size)

        flat_h_c_2_targets = h_c_2_targets.view(-1,1)
        flat_h_c_1_targets = h_c_1_targets.view(-1, 1)
        flat_f_c_targets = f_c_targets.view(-1, 1)
        flat_d_targets = d_targets.view(-1, 1)

        # compute the loss for each group
        loss = self.h_c_2_loss(input= flat_h_c_2_logits, target= flat_h_c_2_targets) \
            + self.h_c_1_loss(input= flat_h_c_1_logits, target= flat_h_c_1_targets) \
            + self.f_c_loss(input= flat_f_c_logits, target= flat_f_c_targets) \
            + self.d_loss(input= flat_d_logits, target= flat_d_targets)
        # Grp level metrics
        self.test_h_c_2_acc(flat_h_c_2_logits, flat_h_c_2_targets)
        self.test_h_c_1_acc(flat_h_c_1_logits, flat_h_c_1_targets)
        self.test_comb_h_c_acc((flat_h_c_2_logits, flat_h_c_1_logits),\
                                (h_c_2_targets, h_c_1_targets))
        self.test_f_c_acc(flat_f_c_logits, flat_f_c_targets)
        self.test_d_acc(flat_d_logits, flat_d_targets)
        self.test_grp_acc((flat_h_c_2_logits, flat_h_c_1_logits, flat_f_c_logits, flat_d_logits),\
                           (flat_h_c_2_targets, flat_h_c_1_targets, flat_f_c_targets, flat_d_targets))
        
        # Word level metric
        self.test_wrr((h_c_2_logits, h_c_1_logits, f_c_logits, d_logits),\
                      (h_c_2_targets, h_c_1_targets, f_c_targets, d_targets))

        # On epoch only logs
        log_dict_epoch = {
            "test_loss_epoch": loss,
            "test_half_character2_acc": self.train_h_c_2_acc,
            "test_half_character1_acc": self.train_h_c_1_acc,
            "test_combined_half_character_acc": self.train_comb_h_c_acc,
            "test_character_acc": self.train_f_c_acc,
            "test_diacritic_acc": self.train_d_acc,
            "test_wrr_epoch": self.train_wrr, 
            "test_grp_acc_epoch": self.train_grp_acc,
        }
        self.log_dict(log_dict_epoch, on_step = False, on_epoch = True, prog_bar = False, logger = True, sync_dist = True)

    def decode(self, batch):
        pass