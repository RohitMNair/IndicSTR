import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchmetrics import Accuracy, Metric
import lightning.pytorch as pl

class CharGrpAccuracy(Metric):
    def __init__(self, threshold= 0.5):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.thresh = threshold

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update the metric
        Args:
            preds (tuple | list): half character logits, character logits, diacritics logits
            target (tuple | list): half character class number, character class number, diacritics 1 or 2 Hot vector
        """
        half_char_logits, char_logits, diac_logits = preds
        half_char, char, diac = target
        # logits will be B * N 
        half_char_preds = torch.argmax(half_char_logits, dim = 1)
        char_preds = torch.argmax(char_logits, dim = 1)

        assert diac_logits.shape == diac.shape
        sigmoid = nn.Sigmoid() # convert to probability scores
        diac_preds = sigmoid(diac_logits)
        # B x 1 bool output
        diac_bin_mask = torch.all((diac_preds > self.thresh) == (diac >= 1.), dim = 1)
        diac_bin_mask = torch.reshape(diac_bin_mask,shape = (-1,1))
        
        # print(half_char_preds.shape, half_char.shape, diac_bin_mask)
        assert half_char_preds.shape == half_char.shape
        half_char_bin_mask = half_char_preds == half_char
        half_char_bin_mask = torch.reshape(half_char_bin_mask,shape= (-1,1))

        assert char_preds.shape == char.shape
        char_bin_mask = char_preds == char
        char_bin_mask = torch.reshape(char_bin_mask, shape= (-1,1))
        
        grp_pred = torch.cat((half_char_bin_mask, char_bin_mask, diac_bin_mask), dim = 1)
        self.correct += torch.sum(torch.all(grp_pred, dim= 1))
        self.total += diac_bin_mask.numel()

    def compute(self):
        return self.correct.float() / self.total
    
    @property
    def is_better(self):
        self.higher_is_better = True

class GrpEmbed(pl.LightningModule):
    def __init__(self, character_classes = 100, diacritic_classes = 35, half_character_classes = 54,
                optimizer = None, lr= 1e-3, threshold = 0.5):
        super().__init__()
        self.character_classes = character_classes # 54 + 10 + 9 + 27 also counting numbers vowels and chinh
        self.diacritic_classes = diacritic_classes
        self.half_character_classes = half_character_classes # aadhe akshar
        self.model = models.resnet50()
        self.model.fc = nn.Sequential(
                            nn.Linear(
                                in_features = 2048,
                                out_features = 2048,
                                bias = True
                            ),
                            nn.ReLU()
                        )
        self.optimizer = optimizer
        self.lr = lr
        self.character_head = nn.Linear(
                                in_features = 2048,
                                out_features = self.character_classes,
                                bias = False
                            )
        self.diacritic_head = nn.Linear( # multi-label classification
                                in_features = 2048,
                                out_features = self.diacritic_classes,
                                bias = False
                            )
        self.half_character_head = nn.Linear(
                                in_features = 2048,
                                out_features = self.half_character_classes,
                                bias = False
                            )
        self.character_loss = nn.CrossEntropyLoss()
        self.half_character_loss = nn.CrossEntropyLoss()
        self.diacritic_loss = nn.BCEWithLogitsLoss()
        self.threshold = threshold
        self.train_acc = CharGrpAccuracy(threshold= self.threshold)
        self.val_acc = CharGrpAccuracy(threshold= self.threshold)

    def forward(self, x):
        x = self.model(x)
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
        self.log("train_loss", loss, on_step = True, on_epoch = True, prog_bar = True)
        self.log("train_acc", self.train_acc, on_step = True, on_epoch = True, prog_bar = True)
        return loss

    def configure_optimizers(self):
        # choose your optimizer
        return self.optimizer(self.parameters(), lr = self.lr)
    
    def validation_step(self, batch, batch_no):
        x, half_char, char, diac = batch
        half_char_logits, char_logits, diac_logits = self.forward(x)
        val_loss = self.half_character_loss(half_char_logits, half_char) + self.character_loss(char_logits, char) \
            + self.diacritic_loss(diac_logits, diac)
        
        self.val_acc((half_char_logits, char_logits, diac_logits), (half_char, char, diac))
        self.log("val_loss", val_loss, on_epoch = True)
        self.log("val_acc", self.val_acc, on_epoch = True)