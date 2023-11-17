import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchvision import models
from torchmetrics import Accuracy

class ResNet(pl.LightningModule):
    """
    ResNet model
    Args:
        version (int): one of 18 (resnet18), 50 (resnet50) and 101 (resnet101)
        out_dim (int): dimensions of the fully connected last layer
        img_size (tuple or int): image dimensions
    """
    def __init__(self, version:int, out_features:int, img_size:tuple or int):
        super().__init__()
        if version == 18:
            self.model = models.resnet18(num_classes = out_features)
        elif version == 50:
            self.model = models.resnet50(num_classes = out_features)
        elif version == 101:
            self.model = models.resnet101(num_classes = out_features)
        else:
            raise ModuleNotFoundError("Given ResNet module does not exist")
        self.out_features = out_features
        self.img_size = img_size
        
    def forward(self, x):
        return self.model(x)


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
  
