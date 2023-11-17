from torchmetrics import Metric
from torch import nn
import torch


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