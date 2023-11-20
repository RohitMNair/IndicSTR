import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torchvision import models
from .metrics import CharGrpAccuracy, DiacriticAccuracy, HalfCharacterAccuracy, CharacterAccuracy

class Img2Vec(pl.LightningModule):
    """
    Model to create vector embeddings for Indic Characters, the model expects a single
    character group to be present in the image, the classification heads will act as 
    the vector embeddings for the characters
    Args:
        character_classes (int): Number of character classes
        diacritic_classes (int): Number of diacritics classes
        half_character_classes (int): Number of characters that can appear
                                        in half in a character group
        backbone (torch.nn.Model or lightning.pytorch.LightningModule): Initialized model
                                                                    to be used as a backbone
        model_out_dim (int): Dimension of the backbones output
        rep_dim (int): Dimension of the representation layer
        rep_layer_act (default: nn.RelU): activation to use in the representation layer
        optimizer (torch.optim): Optimizer to be used
        lr (float): Learning rate to be used
        threshold (float): Threshold to be used for classifying from sigmoid

    """
    def __init__(self, character_classes: list, diacritic_classes: list, half_character_classes:list,
                optimizer = torch.optim.Adam, lr= 1e-3, threshold = 0.5, backbone = None, rep_dim = 2048, 
                weight_decay = 0.01, activation = nn.ReLU()):
        super().__init__()
        self.character_classes = character_classes # 54 + 10 + 9 + 27 also counting numbers vowels and chinh
        self.diacritic_classes = diacritic_classes
        self.half_character_classes = half_character_classes # aadhe akshar
        self.rep_dim = rep_dim
        self.backbone = backbone
        self.activation = activation
        self.rep_layer = nn.Sequential(
                            self.activation,
                            nn.Linear(
                                in_features = self.backbone.model.fc.out_features,
                                out_features = self.rep_dim,
                                bias = True
                            ),
                            self.activation
                        )
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.character_head = nn.Linear(
                                in_features = self.rep_dim,
                                out_features = len(self.character_classes),
                                bias = False
                            )
        self.diacritic_head = nn.Linear( # multi-label classification
                                in_features = self.rep_dim,
                                out_features = len(self.diacritic_classes),
                                bias = False
                            )
        self.half_character_head = nn.Linear(
                                in_features = self.rep_dim,
                                out_features = len(self.half_character_classes) + 1,
                                bias = False
                            )
        self.character_loss = nn.CrossEntropyLoss()
        self.half_character_loss = nn.CrossEntropyLoss()
        self.diacritic_loss = nn.BCEWithLogitsLoss()
        self.threshold = threshold
        
        self.train_acc = CharGrpAccuracy(threshold= self.threshold)
        self.train_diac_acc = DiacriticAccuracy(threshold = self.threshold)
        self.train_half_char_acc = HalfCharacterAccuracy(threshold = self.threshold)
        self.train_char_acc = CharacterAccuracy(threshold = self.threshold)

        self.val_acc = CharGrpAccuracy(threshold= self.threshold)
        self.val_diac_acc = DiacriticAccuracy(threshold = self.threshold)
        self.val_half_char_acc = HalfCharacterAccuracy(threshold = self.threshold)
        self.val_char_acc = CharacterAccuracy(threshold = self.threshold)

    def forward(self, x):
        x = self.backbone(x)
        x = self.rep_layer(x)
        half_char_logits = self.half_character_head(x)
        char_logits = self.character_head(x)
        diac_logits = self.diacritic_head(x)
    
        return half_char_logits, char_logits, diac_logits
    
    def training_step(self, batch, batch_no):
        x, half_char, char, diac = batch
        half_char_logits, char_logits, diac_logits = self.forward(x)
        loss = self.half_character_loss(half_char_logits, half_char) + self.character_loss(char_logits, char) \
            + self.diacritic_loss(diac_logits, diac)

        self.train_acc((half_char_logits, char_logits, diac_logits), (half_char, char, diac))
        self.train_diac_acc(diac_logits, diac)
        self.train_half_char_acc(half_char_logits, half_char)
        self.train_char_acc(char_logits, char)

        # On step logs
        self.log("train_loss", loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        self.log("train_acc", self.train_acc, on_step = True, on_epoch = True, prog_bar = True, logger = True)

        # On epoch only logs
        log_dict = {
            "train_diacritic_acc": self.train_diac_acc,
            "train_half_character_acc": self.train_half_char_acc,
            "train_character_acc": self.train_char_acc
        }
        self.log_dict(
            log_dict,
            on_step = False, 
            on_epoch = True, 
            prog_bar = True,
        )
        
        return loss

    def configure_optimizers(self):
        return self.optimizer(
            self.parameters(), 
            lr = self.lr,
            weight_decay = self.weight_decay
            )
    
    def validation_step(self, batch, batch_no):
        x, half_char, char, diac = batch
        half_char_logits, char_logits, diac_logits = self.forward(x)
        val_loss = self.half_character_loss(half_char_logits, half_char) + self.character_loss(char_logits, char) \
            + self.diacritic_loss(diac_logits, diac)
        
        self.val_acc((half_char_logits, char_logits, diac_logits), (half_char, char, diac))
        
        self.val_diac_acc(diac_logits, diac)
        self.val_half_char_acc(half_char_logits, half_char)
        self.val_char_acc(char_logits, char)
        log_dict = {
            "val_loss": val_loss,
            "val_acc": self.val_acc,
            "val_diacritic_acc": self.val_diac_acc,
            "val_half_character_acc": self.val_half_char_acc,
            "val_character_acc": self.val_char_acc
        }
        self.log_dict(
            log_dict,
            on_epoch = True,
            prog_bar = True,
        )
