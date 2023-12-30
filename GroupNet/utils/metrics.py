from torchmetrics import Metric
from torch import nn
from typing import Tuple
from torch import Tensor
import torch

class CharGrpAccuracy(Metric):
    """
    Metric to calculate the recognition accuracy of an entire character group
    Args:
        threshold: Threshold for classification
    """
    def __init__(self, threshold:float= 0.5):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.is_differentiable = False
        self.higher_is_better = True
        self.full_state_update = False
        self.thresh = threshold

    def update(self, preds: Tuple[Tensor, Tensor, Tensor, Tensor], 
            target: Tuple[Tensor, Tensor, Tensor, Tensor]):
        """
        Update the metric
        Args:
            preds (tuple | list): half character logits, character logits, diacritics logits
            target (tuple | list): half character class number, character class number, 
                                    diacritics 1 or 2 Hot vector
        """
        half_char2_logits, half_char1_logits,char_logits, diac_logits = preds
        half_char2, half_char1, char, diac = target
        # logits will be B * N 
        half_char2_preds = torch.argmax(half_char2_logits, dim = 1)
        half_char1_preds = torch.argmax(half_char1_logits, dim= 1)
        char_preds = torch.argmax(char_logits, dim = 1)

        assert diac_logits.shape == diac.shape
        sigmoid = nn.Sigmoid() # convert to probability scores
        diac_preds = sigmoid(diac_logits)
        # B element bool
        diac_bin_mask = torch.all((diac_preds >= self.thresh) == (diac >= 1.), dim = 1)
        # reshape the mask to a column tensor B x 1
        # So that we can append the masks of diacritic, half_char and char
        # column-wise, inorder to compute correct predictions
        diac_bin_mask = torch.reshape(diac_bin_mask,shape = (-1,1))
        
        assert half_char2_preds.shape == half_char2.shape
        half_char2_bin_mask = half_char2_preds == half_char2
        # Reshape to column tensor B x 1
        half_char2_bin_mask = torch.reshape(half_char2_bin_mask, shape = (-1,1))

        assert half_char1_preds.shape == half_char1.shape
        half_char1_bin_mask = half_char1_preds == half_char1
        # Reshape to column tensor B x 1
        half_char1_bin_mask = torch.reshape(half_char1_bin_mask,shape= (-1,1))

        assert char_preds.shape == char.shape
        char_bin_mask = char_preds == char
        # reshape to a column tensor B x 1
        char_bin_mask = torch.reshape(char_bin_mask, shape= (-1,1))

        # concat the column-vectors column-wise manner to get a B x 3 tensor
        grp_pred = torch.cat((half_char2_bin_mask, half_char1_bin_mask, char_bin_mask, diac_bin_mask), dim = 1)
        # reduce the tensor column-wise, where, if all the element of the 
        # row is True, the the value of that row will be true
        temp = torch.all(grp_pred, dim= 1) # B x 1 Matrix
        # Number of Trues in the matrix
        self.correct += torch.sum(temp)
        # Get the batch size
        self.total += diac_bin_mask.numel()

    def compute(self):
        return self.correct.float() / self.total

class DiacriticAccuracy(Metric):
    """
    Metric to calculate the accuracy of diacritic recognition in character groups
    Args:
        threshold (float): Threshold for classification 
    """
    def __init__(self, threshold):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.is_differentiable = False
        self.higher_is_better = True
        self.full_state_update = False
        self.sigmoid = nn.Sigmoid()
        self.thresh = threshold
    
    def update(self, diacritic_logits: torch.Tensor, diacritic_target: torch.Tensor):
        """
        Update the metric
        Args:
            diacritic_logits (tensor): Diacritic Logits
            diacritic_target (tensor): Corresponding Diacritic true labels 
        """
        assert diacritic_logits.shape == diacritic_target.shape
        d_probs = nn.functional.sigmoid(diacritic_logits)

        # get the probs greater than thresh and the targets, get the 
        # batch item bool mask where all the diacritics are predicted correctly
        d_bin_mask = torch.all((d_probs >= self.thresh) == (diacritic_target >= 1.), dim = 1)
        
        self.correct += d_bin_mask.sum()
        self.total += d_bin_mask.numel()
    
    def compute(self):
        return self.correct.float() / self.total

    @property
    def is_better(self):
        self.higher_is_better = True

class HalfCharacterAccuracy(Metric):
    """
    Metric to calculate the accuracy of Half Character recognition in character groups
    Args:
        threshold (float): Threshold for classification 
    """
    def __init__(self, threshold):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.is_differentiable = False
        self.higher_is_better = True
        self.full_state_update = False
        self.softmax = nn.Softmax(dim= 1)
        self.thresh = threshold
    
    def update(self, half_char_logits: torch.Tensor, half_char_target: torch.Tensor):
        """
        Update the metric
        Args:
            half_char_logits (tensor): Half Character Logits
            half_char_target (tensor): Corresponding Half Character true labels 
        """
        # get the probab.
        h_c_probs = self.softmax(half_char_logits)

        # get max. prob. value and the corresponding index
        h_c_max_probs, h_c_pred  = torch.max(half_char_logits, dim = 1)

        # get mask where max prob. is > thresh
        h_c_bin_mask = h_c_max_probs >= self.thresh

        # Only consider predictions where the max probability is greater than thresh
        h_c_pred_filtered = h_c_pred[h_c_bin_mask]
        h_c_target_filtered = half_char_target[h_c_bin_mask]

        assert h_c_pred_filtered.shape == h_c_target_filtered.shape,\
            "Half-Character prediction and target filtered shapes do not match"
        
        self.correct += (h_c_pred_filtered == h_c_target_filtered).sum()
        self.total += half_char_bin_mask.numel()
    
    def compute(self):
        return self.correct.float() / self.total


class CombinedHalfCharAccuracy(Metric):
    """
    Metric to calculate the accuracy of both half characters in character groups
    Args:
        threshold (float): Threshold for classification
    """
    def __init__(self, threshold:float= 0.5):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.is_differentiable = False
        self.higher_is_better = True
        self.full_state_update = False
        self.thresh = threshold
        self.softmax = nn.Softmax(dim = 1)

    def update(self, half_char_logits:Tuple[Tensor, Tensor], half_char_target:Tuple[Tensor, Tensor]):
        """
        Update the metric
        Args:
            (half_char2_logits, half_char1_logits) (tensor, tensor): Half-Character Logits
            (half_char2_target, half_char1_target) (tensor, tensor): Corresponding Half-Character true labels 
        """
        # Dim logits: Batch x # of classes
        # Dim target: Batch
        h_c_2_logits, h_c_1_logits = half_char_logits
        h_c_2_target, h_c_1_target = half_char_target

        assert h_c_2_target.shape == h_c_1_target.shape, \
            "Half character 1 and Half character 2 target shapes do not match"

        # get the probabilities
        h_c_2_probs = self.softmax(h_c_2_logits)
        h_c_1_probs = self.softmax(h_c_1_logits)

        # get the max. probab. and the index for each pred. in batch
        h_c_2_max_probs, h_c_2_pred = torch.max(h_c_2_probs, dim = 1)
        h_c_1_max_probs, h_c_1_pred = torch.max(h_c_1_probs, dim = 1)

        # get a binary mask for each item in batch
        # where the max probs are above threshold
        h_c_2_bin_mask = h_c_2_max_probs >= self.thresh
        h_c_1_bin_mask = h_c_1_max_probs >= self.thresh
        combined_bin_mask = h_c_2_bin_mask == h_c_1_bin_mask
        
        # only considere those predictions where bin mask is true
        h_c_2_pred_filtered = h_c_2_pred[combined_bin_mask]
        h_c_2_target_filtered = h_c_2_target[combined_bin_mask]
        h_c_1_pred_filtered = h_c_1_pred[combined_bin_mask]
        h_c_1_target_filtered = h_c_1_target[combined_bin_mask]

        # Ensure shapes match
        assert h_c_2_pred_filtered.shape == h_c_2_target_filtered.shape,\
                "shapes of half character 2 filtered and predicted do not match"
        assert h_c_1_pred_filtered.shape == h_c_1_target_filtered.shape, \
                "shapes of half character 1 filtered and predicted do not match"

        # compute correct predictions
        h_c_2_correct = h_c_2_pred_filtered == h_c_2_target_filtered
        h_c_1_correct = h_c_1_pred_filtered == h_c_1_target_filtered

        self.correct += (h_c_2_correct == h_c_1_correct).sum()
        self.total += h_c_2_target.numel() # batch size
    
    def compute(self):
        return self.correct.float() / self.total

    
class FullCharacterAccuracy(Metric):
    """
    Metric to calculate the accuracy of Full Character recognition in character groups
    Args:
        threshold (float): Threshold for classification 
    """
    def __init__(self, threshold):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.is_differentiable = False
        self.higher_is_better = True
        self.full_state_update = False
        self.thresh = threshold
        self.softmax = nn.Softmax(dim= 1)
    
    def update(self, full_char_logits: torch.Tensor, full_char_target: torch.Tensor):
        """
        Update the metric
        Args:
            full_char_logits (tensor): Character Logits
            full_char_target (tensor): Corresponding Character true labels 
        """
        
        # Get the softmax probabilities
        char_probs = self.softmax(char_logits)

        # Get the maximum probabilities along with their indices
        f_c_max_probs, f_c_preds = torch.max(char_probs, dim=1)

        # Check if the maximum probability is greater than 0.5
        f_c_bin_mask = f_c_max_probs >= self.thresh

        # Only consider predictions where the max probability is greater than thresh
        f_c_preds_filtered = f_c_preds[f_c_bin_mask]
        f_c_target_filtered = full_char_target[f_c_bin_mask]

        # Ensure shapes match
        assert f_c_preds_filtered.shape == f_c_target_filtered.shape

        # Compute accuracy
        self.correct += (f_c_preds_filtered == f_c_target_filtered).sum()
        self.total += char_preds.numel()
    
    def compute(self):
        return self.correct.float() / self.total


# class CRR(Metric):
#     """
#     Measures character recognition rate
#     threshold (measures)
#     """
    