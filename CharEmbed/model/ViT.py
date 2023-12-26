import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchmetrics import Accuracy
import lightning.pytorch as pl


class ViT(pl.LightningModule):
    """
    ViT model
    Args:
        num_classes (int): no. of classes
        version (str): one of "b_16", "b_32", "l_16", "l_32", and "h_14"
    """
    def __init__(self, num_classes: int, version: str, optimizer = None, lr= 1e-3):
        super().__init__()
        # define model and loss
        self.num_classes = num_classes

        if version.lower() == "b_16":
            self.model = models.vit_b_16(num_classes= self.num_classes)
        elif version.lower() == "b_32":
            self.model = models.vit_b_32(num_classes= self.num_classes)
        elif version == "l_16":
            self.model = models.vit_l_16(num_classes= self.num_classes)
        elif version.lower() == "l_32":
            self.model = models.vit_l_32(num_classes= self.num_classes)
        elif version.lower() == "h_14":
            self.model = models.vit_h_14(num_classes= self.num_classes)
        else:
            raise ModuleNotFoundError("Given ViT module does not exist")
        
        self.loss = nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.optimizer = optimizer
        self.lr = lr

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_no):
        # implement single training step
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, on_step = True, on_epoch = False, prog_bar = True)
        return loss
    
    def configure_optimizers(self):
        # choose your optimizer
        return self.optimizer(self.parameters(), lr=self.lr)
    
    def validation_step(self, batch, batch_no):
        # this is the validation loop
        x, y = batch
        logits = self.forward(x)
        val_loss = self.loss(logits, y)
        
        self.val_accuracy(logits, y)
        self.log("val_loss", val_loss, on_epoch = True)
        self.log("val_acc", self.val_accuracy, on_epoch = True)