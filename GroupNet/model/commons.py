from typing import Any, Tuple
import lightning.pytorch as pl
from torch import Tensor
import torch.nn as nn
import torch

class GrpClassifier(pl.LightningModule):
    """
    A multi-head-multi-label classifier to classify characters within a group
    """
    def __init__(self, hidden_size:int, num_half_character_classes:int,
                 num_full_character_classes:int, num_diacritic_classes:int):
        """
        Constructor for GrpClassifier
        Args:
        - hidden_size (int): the size of the input feature vectors
        - num_half_character_classes (int): Number of half characters
        - num_full_character_classes (int): Number of full characters
        - num_diacritic_classes (int): Number of diacritic classes
        """
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
        
    def forward(self, x:Tensor)-> Tuple[Tensor, Tensor, Tensor, Tensor] :
        half_char2_logits = self.half_character2_head(x)
        half_char1_logits = self.half_character1_head(x)
        char_logits = self.character_head(x)
        diac_logits = self.diacritic_head(x)
        return half_char2_logits, half_char1_logits, char_logits, diac_logits

    
class FixedGrpClassifier(pl.LightningModule):
    """
    Group classifier but instead of learnable parameters they are fixed, with the weights
    taken from the embedding models classification layer
    """
    def __init__(self, hidden_size:int, half_char1_embeddings:Tensor, half_char2_embeddings:Tensor, 
                 full_char_embeddings:Tensor, diacritic_embeddings:Tensor):
        """
        Constructor for FixedGrpClassifier
        Args:
        - hidden_size (int): Hidden size or the dimension of the feature vector
        - half_char1_embeddings (Tensor): the half-character 1 embeddings, will be the weights of the half-character 1
                                                classification layer
        - half_char2_embeddings (Tensor): the half-character 2 embeddings, will be the weights of the half-character 2
                                                classification layer
        - full_character_embeddings (Tensor): the full-character embeddings, will be the weights of the full-character
                                                classification layer
        - diacritic_embeddings (Tensor): the diacritic embeddings, will be the weights of the diacritic
                                            classification layer
        """
        # parametric args
        self.hidden_size = hidden_size
        # non-parameteric args
        self.embed_size = half_char1_embeddings.shape[-1]

        # project the feature dimension to the same as the embed. dim
        self.project = nn.Linear(in_features= hidden_size, out_features= self.embed_size)

        # weight matrix for classification
        self.half_character2_head = nn.Parameter(torch.permute(half_char1_embeddings, (1, 0)), requires_grad= False)
        self.half_character1_head = nn.Parameter(torch.permute(half_char1_embeddings, (1, 0)), requires_grad= False)
        self.character_head = nn.Parameter(torch.permute(full_char_embeddings, (1, 0)), requires_grad= False)
        self.diacritic_head = nn.Parameter(torch.permute(diacritic_embeddings, (1, 0)), requires_grad= False)
        # to handle [PAD] & [EOS] we need additional trainable parameters
        self.half_character2_special = nn.Linear(in_features = self.embed_size, out_features= 2) # we have BLANK in embeddings
        self.half_character1_special = nn.Linear(in_features= self.embed_size, out_features= 2)
        self.character_special = nn.Linear(in_features= self.embed_size, out_features= 2)
        self.diacritic_special = nn.Linear(self.embed_size, out_features= 2)

    def forward(self, x:Tensor)-> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.project(x) # bs x grps x embed dim

        # logits shape: bs x grps x # classes
        half_char1_logits = self.half_character2_head @ x
        half_char2_logts = self.half_character1_head @ x
        full_char_logits = self.character_head @ x
        diac_logits = self.diacritic_head @ x
        
        # special logits shape: bs x grps x 2
        hc2_special_logits = self.half_character2_special(x)
        hc1_special_logits = self.half_character1_special(x)
        fc_special_logits = self.character_special(x)
        diac_special_logits = self.diacritic_special(x)

                
        return