import lightning.pytorch as pl
from torch import Tensor
import torch.nn as nn

class GrpClassifier(pl.LightningModule):
    def __init__(self, hidden_size:int, num_half_character_classes:int,
                 num_full_character_classes:int, num_diacritic_classes:int):
        super().__init__()
        self.hidden_size= hidden_size
        self.num_h_c_classes = num_half_character_classes
        self.num_f_c_classes = num_full_character_classes
        self.num_d_classes = num_diacritic_classes
        self.half_character2_head = nn.Linear(
                                in_features = self.hidden_size,
                                out_features = self.num_h_c_classes, # extra node for no half-char
                                bias = True
                            )
        self.half_character1_head = nn.Linear(
                                in_features = self.hidden_size,
                                out_features = self.num_h_c_classes,
                                bias = True
                            )
        self.character_head = nn.Linear(
                                in_features = self.hidden_size,
                                out_features = self.num_f_c_classes,
                                bias = True
                            )
        self.diacritic_head = nn.Linear( # multi-label classification hence no need for extra head
                                in_features = self.hidden_size,
                                out_features = self.num_d_classes,
                                bias = True
                            )
        
    def forward(self, x:Tensor):
        half_char2_logits = self.half_character2_head(x)
        half_char1_logits = self.half_character1_head(x)
        char_logits = self.character_head(x)
        diac_logits = self.diacritic_head(x)
        return half_char2_logits, half_char1_logits, char_logits, diac_logits 