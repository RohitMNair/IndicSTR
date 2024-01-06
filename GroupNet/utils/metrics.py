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
        self.softmax = nn.Softmax(dim = 1)
        self.sigmoid = nn.Sigmoid()
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
        h_c_2_logits, h_c_1_logits, f_c_logits, d_logits = preds
        h_c_2_target, h_c_1_target, f_c_target, d_target = target
        
        # calc probabs
        h_c_2_probs = self.softmax(h_c_2_logits)
        h_c_1_probs = self.softmax(h_c_1_logits)
        f_c_probs = self.softmax(f_c_logits)
        d_probs = self.sigmoid(d_logits)

        # get the max probab and correspondig index
        h_c_2_max_probs, h_c_2_preds= torch.max(h_c_2_probs, dim= 1)
        h_c_1_max_probs, h_c_1_preds = torch.max(h_c_1_probs, dim= 1)
        f_c_max_probs, f_c_preds = torch.max(f_c_probs, dim= 1)

        # bin mask for batch items having predictions greater than thresh
        h_c_2_bin_mask = h_c_2_max_probs >= self.thresh
        h_c_1_bin_mask = h_c_1_max_probs >= self.thresh
        f_c_bin_mask = f_c_max_probs >= self.thresh

        combined_mask = (h_c_2_bin_mask == h_c_1_bin_mask) == f_c_bin_mask

        # consider only those data items where max probab > thresh
        h_c_2_preds_filtered = h_c_2_preds[combined_mask]
        h_c_1_preds_filtered = h_c_1_preds[combined_mask]
        f_c_preds_filtered = f_c_preds[combined_mask]

        h_c_2_target_filtered = h_c_2_target[combined_mask]
        h_c_1_target_filtered = h_c_1_target[combined_mask]
        f_c_target_filtered = f_c_target[combined_mask]

        h_c_2_correct = h_c_2_preds_filtered == h_c_2_target_filtered
        h_c_1_correct = h_c_1_preds_filtered == h_c_1_target_filtered
        f_c_correct = f_c_preds_filtered == f_c_target_filtered   
        d_correct = torch.all((d_probs >= self.thresh) == (d_target >= 1.), dim = 1)
        combined_correct = torch.stack([h_c_2_correct, h_c_1_correct, f_c_correct, d_correct], dim= 0)

        self.correct += torch.all(combined_correct, dim= 0).sum()
        self.total += h_c_2_target.numel() # batch size

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
        self.total += half_char_target.numel()
    
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
        combined_correct = torch.stack([h_c_2_correct, h_c_1_correct], dim= 0)

        self.correct += torch.all(combined_correct, dim= 0).sum()
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
        char_probs = self.softmax(full_char_logits)

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
        self.total += full_char_target.numel()
    
    def compute(self):
        return self.correct.float() / self.total


class WRR(Metric):
    """
    Measures Word Recognition Rate(WRR)
    Args:
     threshold (float): float value between 0 and 1, will behave as threshold for
                        classification.
    """
    def __init__(self, threshold:float= 0.5):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.is_differentiable = False
        self.higher_is_better = True
        self.full_state_update = False
        self.thres = threshold
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim= 1)

    def update(self, logits:Tuple[Tensor, Tensor, Tensor, Tensor], 
               targets:Tuple[Tensor, Tensor, Tensor, Tensor]):
        # shape character logits: BS x Max Grps x # of classes
        # shape character target: BS x Max Grps
        # shape diacritic logits: BS x Max Grps x # of classes
        # shape diacritic target: BS x Max Grps x # of classes
        h_c_2_logits, h_c_1_logits, f_c_logits, d_logits = logits
        h_c_2_target, h_c_1_target, f_c_target, d_target = targets

        # check shapes
        assert len(h_c_2_logits.shape) == len(h_c_1_logits.shape) == len(f_c_logits.shape) == 3 and \
              len(h_c_2_target.shape) == len(h_c_1_target.shape) == len(f_c_target.shape) == 2, \
        "for half and full characters logits shape must be (Batch_Size, Max Groups, # of classes) and target shape \
        should be (Batch_size, Max Grps) containing the class number"

        assert len(d_logits.shape) == 3 and len(d_target.shape) == 3, \
        "For diacritics the logits and target must be of shape (Batch Size, Max Groups, # of classes)"
        
        # get probab values
        h_c_2_probabs = self.softmax(h_c_2_logits, dim= 2)
        h_c_1_probabs = self.softmax(h_c_1_logits, dim = 2)
        f_c_probabs = self.softmax(f_c_logits, dim= 2)
        d_probabs = self.sigmoid(d_logits)

        # get the max probab and the corresponding index
        h_c_2_max_probs, h_c_2_preds = torch.max(h_c_2_probabs, dim= 2)
        h_c_1_max_probs, h_c_1_preds = torch.max(h_c_1_probabs, dim= 2)
        f_c_max_probs, f_c_preds = torch.max(f_c_probabs, dim= 2)

        # batch mask where max. val is greater than thresh
        h_c_2_bin_mask = torch.all(h_c_2_max_probs >= self.thresh, dim= 1)
        h_c_1_bin_mask = torch.all(h_c_1_max_probs >= self.thresh, dim= 1)
        f_c_bin_mask = torch.all(f_c_max_probs >= self.thresh, dim= 1)

        # filter out batch items with low confidence predictions
        h_c_2_preds_filtered = h_c_2_preds[h_c_2_bin_mask]
        h_c_1_preds_filtered = h_c_1_preds[h_c_1_bin_mask]
        f_c_preds_filtered = f_c_preds[f_c_bin_mask]

        h_c_2_target_filtered = h_c_2_target[h_c_2_bin_mask]
        h_c_1_target_filtered = h_c_1_target[h_c_1_bin_mask]
        f_c_target_filtered = f_c_target[f_c_bin_mask]

        h_c_2_correct = torch.all(h_c_2_preds_filtered == h_c_2_target_filtered, dim= 1)
        h_c_1_correct = torch.all(h_c_1_preds_filtered == h_c_1_target_filtered, dim= 1)
        f_c_correct = torch.all(f_c_preds_filtered == f_c_target_filtered, dim= 1)
        d_correct = torch.all(torch.all((d_probabs >= self.thresh) == (d_target >= 1.), dim= 2), dim= 1)
        combined_correct = torch.all(torch.stack([h_c_2_correct, h_c_1_correct, f_c_correct, d_correct], dim= 0), dim= 0)
        
        self.correct += combined_correct.sum()
        self.total =+ h_c_2_target.shape[0]

    def compute(self):
        return self.correct.float() / self.total    