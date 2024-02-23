from .encoder import FocalNetEncoder
from model.commons import GrpClassifier
from utils.metrics import (DiacriticAccuracy, FullCharacterAccuracy, CharGrpAccuracy, NED,
                   HalfCharacterAccuracy, CombinedHalfCharAccuracy, WRR, WRR2, ComprihensiveWRR)
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import OneCycleLR
from typing import Tuple, Optional
from torch import Tensor
from data.tokenizer import Tokenizer

import lightning.pytorch.loggers as pl_loggers
import lightning.pytorch as pl
import torch
import torch.nn as nn

class FocalSTR(pl.LightningModule):
    """
    Group implementation of ViTSTR but instead of ViT, we use FocalNet
    """
    def __init__(self, half_character_classes:list, full_character_classes:list,
                 diacritic_classes:list, halfer:str, embed_dim: int = 96, depths:list= [2, 2, 6, 2],
                 focal_levels:list= [2, 2, 2, 2], focal_windows:list= [3, 3, 3, 3],
                 mlp_ratio: float= 4.0, hidden_dropout_prob: float = 0.0,
                 drop_path_rate:float = 0.1, initializer_range: float = 0.02, 
                 layer_norm_eps: float = 1e-12, image_size: int = 224, patch_size: int = 16, 
                 num_channels: int = 3, threshold:float= 0.5,
                 learning_rate: float= 1e-4, weight_decay: float= 1.0e-4, warmup_pct:float= 0.3):
        super().__init__()
        self.save_hyperparameters()
        self.embed_dim = embed_dim
        self.depths = depths
        self.hidden_sizes = [self.embed_dim * (2 ** i) for i in range(len(depths))]
        self.focal_levels = focal_levels
        self.focal_windows = focal_windows
        self.mlp_ratio = mlp_ratio
        self.hidden_dropout_prob = hidden_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.threshold = threshold
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.pct_start = warmup_pct
        self.max_grps = 16 #Focal net returns 16 feature representations

        # non parameteric attributes
        self.tokenizer = Tokenizer(
            half_character_classes= half_character_classes,
            full_character_classes= full_character_classes,
            diacritic_classes= diacritic_classes,
            halfer= halfer,
            threshold= threshold,
            max_grps= self.max_grps,
        )
        
        self.num_h_c_classes = len(self.tokenizer.h_c_classes)
        self.num_f_c_classes = len(self.tokenizer.f_c_classes)
        self.num_d_classes =  len(self.tokenizer.d_classes)

        self.encoder = FocalNetEncoder(
            hidden_dropout_prob= self.hidden_dropout_prob, 
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

        self.classifier = GrpClassifier(
            hidden_size= self.hidden_sizes[-1],
            num_half_character_classes= self.num_h_c_classes,
            num_full_character_classes= self.num_f_c_classes,
            num_diacritic_classes= self.num_d_classes,
        )
        # #0th position will be [B], thus less weight for 0th index
        # self.h_c_class_weight = torch.tensor([.1] + [10. for _ in range(self.num_h_c_classes - 1)])
        # self.f_c_class_weight = torch.tensor([.1] + [10. for _ in range(self.num_f_c_classes - 1)])
        # self.d_pos_weight = torch.full([self.num_d_classes], 30.)
        self.h_c_2_loss = nn.CrossEntropyLoss(reduction= 'mean', ignore_index= self.tokenizer.pad_id)
        self.h_c_1_loss = nn.CrossEntropyLoss(reduction= 'mean', ignore_index= self.tokenizer.pad_id)
        self.f_c_loss = nn.CrossEntropyLoss(reduction= 'mean', ignore_index= self.tokenizer.pad_id)
        self.d_loss = nn.BCEWithLogitsLoss(reduction= 'mean')

        # Trainig Metrics
        self.train_h_c_2_acc = HalfCharacterAccuracy(threshold = self.threshold)
        self.train_h_c_1_acc = HalfCharacterAccuracy(threshold = self.threshold)
        self.train_comb_h_c_acc = CombinedHalfCharAccuracy(threshold= self.threshold)
        self.train_f_c_acc = FullCharacterAccuracy(threshold = self.threshold)
        self.train_d_acc = DiacriticAccuracy(threshold = self.threshold)
        self.train_grp_acc = CharGrpAccuracy(threshold= self.threshold)
        # self.train_wrr = ComprihensiveWRR(threshold= self.threshold)
        self.train_wrr2 = WRR2(threshold= self.threshold)
        # Validation Metrics
        self.val_h_c_1_acc = HalfCharacterAccuracy(threshold = self.threshold)
        self.val_h_c_2_acc = HalfCharacterAccuracy(threshold = self.threshold)
        self.val_comb_h_c_acc = CombinedHalfCharAccuracy(threshold= self.threshold)
        self.val_f_c_acc = FullCharacterAccuracy(threshold = self.threshold)
        self.val_d_acc = DiacriticAccuracy(threshold = self.threshold)
        self.val_grp_acc = CharGrpAccuracy(threshold= self.threshold)
        # self.val_wrr = ComprihensiveWRR(threshold= self.threshold)
        self.val_wrr2 = WRR2(threshold= self.threshold)
        # Testing Metrics
        self.test_h_c_1_acc = HalfCharacterAccuracy(threshold = self.threshold)
        self.test_h_c_2_acc = HalfCharacterAccuracy(threshold = self.threshold)
        self.test_comb_h_c_acc = CombinedHalfCharAccuracy(threshold= self.threshold)
        self.test_f_c_acc = FullCharacterAccuracy(threshold = self.threshold)
        self.test_d_acc = DiacriticAccuracy(threshold = self.threshold)
        self.test_grp_acc = CharGrpAccuracy(threshold= self.threshold)
        self.test_wrr = WRR()
        self.test_wrr2 = WRR2(threshold= self.threshold)
        self.ned = NED()

    def forward(self, x:torch.Tensor)-> Tuple[Tensor, Tensor, Tensor, Tensor]:
        enc_x = self.encoder(x)
        h_c_2_logits, h_c_1_logits, f_c_logits, d_logits = self.classifier(enc_x)
        return (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits)
    
    def _log_tb_images(self, viz_batch:Tensor, pred_labels:tuple, gt_labels:tuple, mode:str= "test") -> None:
        """
        Function to display a batch and its predictions to tensorboard
        Args:
        - viz_batch (Tensor): images of a batch to be visualized
        - labels (tuple): corresponding lables of images in the batch
        - mode (str): "test" | "train" | "val"

        Returns: None
        """
        assert mode in ("test", "train", "val"), "Invalid mode"
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
                raise ValueError('TensorBoard Logger not found')

        assert gt_labels is not None, "gt_labels cannot be none"
        assert pred_labels is not None, "pred_labels cannot be none"
        # Log the images (Give them different names)
        for img_idx in range(viz_batch.shape[0]):
            tb_logger.add_image(f"{mode}/{self.global_step}_pred-{pred_labels[img_idx]}_gt-{gt_labels[img_idx]}", viz_batch[img_idx], 0)
    
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

        flat_h_c_2_non_pad = (flat_h_c_2_targets != self.tokenizer.pad_id)
        flat_h_c_1_non_pad = (flat_h_c_1_targets != self.tokenizer.pad_id)
        flat_f_c_non_pad = (flat_f_c_targets != self.tokenizer.pad_id)
        d_pad = torch.zeros(self.num_d_classes, dtype = torch.float32, device= self.device)
        d_pad[self.tokenizer.pad_id] = 1.
        flat_d_non_pad = ~ torch.all(flat_d_targets == d_pad, dim= 1)
        assert torch.all((flat_h_c_2_non_pad == flat_h_c_1_non_pad) == (flat_f_c_non_pad == flat_d_non_pad)).item(), \
                "Pads are not aligned properly"

        flat_h_c_2_targets = flat_h_c_2_targets[flat_h_c_2_non_pad]
        flat_h_c_1_targets = flat_h_c_1_targets[flat_h_c_2_non_pad]
        flat_f_c_targets = flat_f_c_targets[flat_h_c_2_non_pad]
        flat_d_targets = flat_d_targets[flat_h_c_2_non_pad]

        flat_h_c_2_logits = h_c_2_logits.reshape(-1, self.num_h_c_classes)[flat_h_c_2_non_pad]
        flat_h_c_1_logits = h_c_1_logits.reshape(-1, self.num_h_c_classes)[flat_h_c_2_non_pad]
        flat_f_c_logits = f_c_logits.reshape(-1, self.num_f_c_classes)[flat_h_c_2_non_pad]
        flat_d_logits = d_logits.reshape(-1, self.num_d_classes)[flat_h_c_2_non_pad]

        return ((flat_h_c_2_targets, flat_h_c_1_targets, flat_f_c_targets, flat_d_targets), 
                (flat_h_c_2_logits, flat_h_c_1_logits, flat_f_c_logits, flat_d_logits))
    
    def training_step(self, batch, batch_no)-> Tensor:
        # batch: img Tensor(BS x C x H x W), label tuple(BS)
        imgs, labels = batch
        batch_size = len(labels)
        h_c_2_targets, h_c_1_targets, f_c_targets, d_targets = (
                                  torch.zeros(batch_size, self.max_grps, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps, self.num_d_classes, device= self.device),
                                )
        # print(f"The Targets:")
        n_grps = [self.max_grps for i in range(batch_size)]
        for idx,label in enumerate(labels, start= 0):
            h_c_2_targets[idx], h_c_1_targets[idx], f_c_targets[idx], d_targets[idx], n_grps[idx] = self.tokenizer.label_encoder(label, device= self.device)
            # print(f"The label:{label}; The Targets: {h_c_2_targets[idx]}\n{h_c_1_targets[idx]}\n{f_c_targets[idx]}\n{d_targets[idx]}\n{n_grps[idx]}\n\n")

        (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits) = self.forward(imgs)
        
        # Get the flattened versions of the targets and the logits for grp level metrics
        ((flat_h_c_2_targets, flat_h_c_1_targets, flat_f_c_targets, flat_d_targets), 
        (flat_h_c_2_logits, flat_h_c_1_logits, flat_f_c_logits, flat_d_logits)) = self._get_flattened_non_pad(
                                                                                targets= (h_c_2_targets, h_c_1_targets, f_c_targets, d_targets),
                                                                                logits= (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits),
                                                                            )

        # compute the loss for each group
        loss = self.h_c_2_loss(input= flat_h_c_2_logits, target= flat_h_c_2_targets) \
            + self.h_c_1_loss(input= flat_h_c_1_logits, target= flat_h_c_1_targets) \
            + self.f_c_loss(input= flat_f_c_logits, target= flat_f_c_targets) \
            + self.d_loss(input= flat_d_logits, target= flat_d_targets)
        # print(f"The loss: {loss}")
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
        self.train_wrr2((h_c_2_logits, h_c_1_logits, f_c_logits, d_logits),\
                       (h_c_2_targets, h_c_1_targets, f_c_targets, d_targets), self.tokenizer.pad_id)
        # self.train_wrr(pred_strs= self.tokenizer.decode((h_c_2_logits, h_c_1_logits, f_c_logits, d_logits)), target_strs= labels)

        if batch_no % 1000000 == 0:
            pred_labels = self.tokenizer.decode((h_c_2_logits, h_c_1_logits, f_c_logits, d_logits))            
            self._log_tb_images(imgs[:5], pred_labels= pred_labels[:5], gt_labels= labels[:5], mode= "train")
        # On step logs for proggress bar display
        log_dict_step = {
            "train_loss_step": loss,
            "train_wrr2_step": self.train_wrr2,
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
            "train_wrr2_epoch": self.train_wrr2, 
            "train_grp_acc_epoch": self.train_grp_acc,
        }
        self.log_dict(log_dict_epoch, on_step = False, on_epoch = True, prog_bar = False, logger = True, sync_dist = True)  

        return loss

    def validation_step(self, batch, batch_no)-> None:
        # batch: img (BS x C x H x W), label (BS)
        imgs, labels = batch
        batch_size = len(labels)
        h_c_2_targets, h_c_1_targets, f_c_targets, d_targets = (
                                  torch.zeros(batch_size, self.max_grps, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps, self.num_d_classes, device= self.device),
                                )
        n_grps = [self.max_grps for i in range(batch_size)]
        for idx,label in enumerate(labels, start= 0):
            h_c_2_targets[idx], h_c_1_targets[idx], f_c_targets[idx], d_targets[idx], n_grps[idx] = self.tokenizer.label_encoder(label, device= self.device)

        (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits) = self.forward(imgs)

        # Get the flattened versions of the targets and the logits for grp level metrics
        ((flat_h_c_2_targets, flat_h_c_1_targets, flat_f_c_targets, flat_d_targets), 
        (flat_h_c_2_logits, flat_h_c_1_logits, flat_f_c_logits, flat_d_logits)) = self._get_flattened_non_pad(
                                                                                targets= (h_c_2_targets, h_c_1_targets, f_c_targets, d_targets),
                                                                                logits= (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits),
                                                                            )

        # compute the loss for each group
        loss = self.h_c_2_loss(input= flat_h_c_2_logits, target= flat_h_c_2_targets) \
            + self.h_c_1_loss(input= flat_h_c_1_logits, target= flat_h_c_1_targets) \
            + self.f_c_loss(input= flat_f_c_logits, target= flat_f_c_targets) \
            + self.d_loss(input= flat_d_logits, target= flat_d_targets)
        
        # Grp level metrics
        self.val_h_c_2_acc(flat_h_c_2_logits, flat_h_c_2_targets)
        self.val_h_c_1_acc(flat_h_c_1_logits, flat_h_c_1_targets)
        self.val_comb_h_c_acc((flat_h_c_2_logits, flat_h_c_1_logits),\
                               (flat_h_c_2_targets, flat_h_c_1_targets))
        self.val_f_c_acc(flat_f_c_logits, flat_f_c_targets)
        self.val_d_acc(flat_d_logits, flat_d_targets)
        self.val_grp_acc((flat_h_c_2_logits, flat_h_c_1_logits, flat_f_c_logits, flat_d_logits),\
                           (flat_h_c_2_targets, flat_h_c_1_targets, flat_f_c_targets, flat_d_targets))
        # Word level metric
        self.val_wrr2((h_c_2_logits, h_c_1_logits, f_c_logits, d_logits),\
                     (h_c_2_targets, h_c_1_targets, f_c_targets, d_targets), self.tokenizer.pad_id)
        # self.val_wrr(pred_strs= self.tokenizer.decode((h_c_2_logits, h_c_1_logits, f_c_logits, d_logits)), target_strs= labels)
        
        if batch_no % 100000 == 0:
            pred_labels = self.tokenizer.decode((h_c_2_logits, h_c_1_logits, f_c_logits, d_logits))            
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
        self.log_dict(log_dict_epoch, on_step = False, on_epoch = True, prog_bar = False, logger = True, sync_dist = True)

    def test_step(self, batch, batch_no)-> None:
        # batch: img (BS x C x H x W), label (BS)
        imgs, labels = batch
        batch_size = imgs.shape[0]
        h_c_2_targets, h_c_1_targets, f_c_targets, d_targets = (
                                  torch.zeros(batch_size, self.max_grps, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps, device= self.device, dtype= torch.long),
                                  torch.zeros(batch_size, self.max_grps, self.num_d_classes, device= self.device),
                                )
        n_grps = [self.max_grps for i in range(batch_size)]
        for idx,label in enumerate(labels, start= 0):
            h_c_2_targets[idx], h_c_1_targets[idx], f_c_targets[idx], d_targets[idx], n_grps[idx] = self.tokenizer.label_encoder(label, device= self.device)

        (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits) = self.forward(imgs)

        # Get the flattened versions of the targets and the logits for grp level metrics
        ((flat_h_c_2_targets, flat_h_c_1_targets, flat_f_c_targets, flat_d_targets), 
        (flat_h_c_2_logits, flat_h_c_1_logits, flat_f_c_logits, flat_d_logits)) = self._get_flattened_non_pad(
                                                                            targets= (h_c_2_targets, h_c_1_targets, f_c_targets, d_targets),
                                                                            logits= (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits),
                                                                            )

        # compute the loss for each group
        loss = self.h_c_2_loss(input= flat_h_c_2_logits, target= flat_h_c_2_targets) \
            + self.h_c_1_loss(input= flat_h_c_1_logits, target= flat_h_c_1_targets) \
            + self.f_c_loss(input= flat_f_c_logits, target= flat_f_c_targets) \
            + self.d_loss(input= flat_d_logits, target= flat_d_targets)
        
        # Grp level metrics
        self.test_h_c_2_acc(flat_h_c_2_logits, flat_h_c_2_targets)
        self.test_h_c_1_acc(flat_h_c_1_logits, flat_h_c_1_targets)
        self.test_comb_h_c_acc((flat_h_c_2_logits, flat_h_c_1_logits),\
                                (flat_h_c_2_targets, flat_h_c_1_targets))
        self.test_f_c_acc(flat_f_c_logits, flat_f_c_targets)
        self.test_d_acc(flat_d_logits, flat_d_targets)
        self.test_grp_acc((flat_h_c_2_logits, flat_h_c_1_logits, flat_f_c_logits, flat_d_logits),\
                           (flat_h_c_2_targets, flat_h_c_1_targets, flat_f_c_targets, flat_d_targets))
        
        # Word level metric
        self.test_wrr2(logits= (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits),\
                      targets= (h_c_2_targets, h_c_1_targets, f_c_targets, d_targets), pad_id= self.tokenizer.pad_id)
        
        pred_labels= self.tokenizer.decode((h_c_2_logits, h_c_1_logits, f_c_logits, d_logits))
        self.test_wrr(pred_strs= pred_labels, target_strs= labels)        
        self.ned(pred_labels= pred_labels, target_labels= labels)           
        self._log_tb_images(imgs, pred_labels= pred_labels, gt_labels= labels, mode= "test")
            
        # On epoch only logs
        log_dict_epoch = {
            "test_loss": loss,
            "test_half_character2_acc": self.test_h_c_2_acc,
            "test_half_character1_acc": self.test_h_c_1_acc,
            "test_combined_half_character_acc": self.test_comb_h_c_acc,
            "test_character_acc": self.test_f_c_acc,
            "test_diacritic_acc": self.test_d_acc,
            "test_wrr": self.test_wrr,
            "test_wrr2": self.test_wrr2,
            "test_grp_acc": self.test_grp_acc,
            "NED": self.ned,
        }
        self.log_dict(log_dict_epoch, on_step = False, on_epoch = True, prog_bar = False, logger = True, sync_dist = True)

    def predict_step(self, batch):
        (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits) = self.forward(batch)
        pred_labels = self.tokenizer.decode((h_c_2_logits, h_c_1_logits, f_c_logits, d_logits))
        return pred_labels, (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits)