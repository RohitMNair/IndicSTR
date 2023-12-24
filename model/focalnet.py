from transformers import FocalNetModel
from torch.nn import GELU
import lightning.pytorch as pl

class FocalNetBackbone(pl.LightningModule):
    """
    class to implement FocalNet as the backbone for Img2Vec model
    Args:
        config: a FocalNetConfig instance
    """
    def __init__(self, config, out_features:int):
        super().__init__()
        self.model = FocalNetModel(config, add_pooling_layer = True)
        self.out_features = out_features # Dimension of the last stage
        # self.activation = GELU(approximate='none')

    def forward(self, x):
        x = self.model(x, return_dict = True).pooler_output # we only need the pooling layer output
        return x

