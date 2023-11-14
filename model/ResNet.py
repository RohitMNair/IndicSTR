import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchmetrics import Accuracy
import lightning.pytorch as pl

class ResNet(pl.LightningModule):
    """
    ResNet18 model
    Args:
        num_classes (int): no. of classes
        version (int): one of 18 (resnet18), 50 (resnet50) and 101 (resnet101)
    """
    def __init__(self, num_classes:int, version:int, optimizer = None, lr= 1e-3):
        super().__init__()
        # define model and loss
        self.num_classes = num_classes

        if version == 18:
            in_features = 512
            self.model = models.resnet18()
        elif version == 50:
            in_features = 2048
            self.model = models.resnet50()
        elif version == 101:
            in_features = 4096
            self.model = models.resent101()
        else:
            raise ModuleNotFoundError("Given ResNet module does not exist")
        
        self.model.fc = torch.nn.Linear(
                            in_features= in_features,
                            out_features= self.num_classes,
                            bias = False
                        )
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
        return self.optimizer(self.parameters(), lr = self.lr)
    
    def validation_step(self, batch, batch_no):
        # this is the validation loop
        x, y = batch
        logits = self.forward(x)
        val_loss = self.loss(logits, y)
        
        self.val_accuracy(logits, y)
        self.log("val_loss", val_loss, on_epoch = True)
        self.log("val_acc", self.val_accuracy, on_epoch = True)


class ResNet50(pl.LightningModule):
    """
    ResNet50 model
    Args:
        num_classes (int): no. of classes
    """
    def __init__(self, num_classes):
        super().__init__()
        # define model and loss
        self.num_classes = num_classes
        self.model = models.resnet50(num_classes= self.num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

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
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def validation_step(self, batch, batch_no):
        # this is the validation loop
        x, y = batch
        logits = self.forward(x)
        val_loss = self.loss(logits, y)
        
        self.val_accuracy(logits, y)
        self.log("val_loss", val_loss, on_epoch = True)
        self.log("val_acc", self.val_accuracy, on_epoch = True)
  
