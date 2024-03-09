import torch.nn as nn
import torch
import unicodedata
from abc import ABC, abstractmethod

from typing import Tuple
from torch import Tensor

class BaseTokenizer(ABC):
    BLANK = "[B]"
    EOS = "[E]"
    PAD = "[P]"
    def __init__(self, half_character_classes:list, full_character_classes:list, diacritic_classes:list,
                halfer:str, threshold:float= 0.5, max_grps:int= 25):
        """
        Args:
        - half_character_set (list)
        - full_character_set (list)
        - diacritic_set (list)
        - halfer (str): diacritic used to represent a half-character
        - threshold (float): classification threshold (default: 0.5)
        """
        self.h_c_classes = [BaseTokenizer.EOS, BaseTokenizer.PAD, BaseTokenizer.BLANK] + half_character_classes
        self.f_c_classes = [BaseTokenizer.EOS, BaseTokenizer.PAD] + full_character_classes
        self.d_classes = [BaseTokenizer.EOS, BaseTokenizer.PAD] + diacritic_classes       
        self.pad_id = 1
        self.eos_id = 0
        self.blank_id = 2
        self._normalize_charset()
        self.h_c_set = set(self.h_c_classes)
        self.f_c_set = set(self.f_c_classes)
        self.d_c_set = set(self.d_classes)
        self.halfer = halfer
        self.thresh = threshold
        self.max_grps = max_grps

        # dict with class indexes as keys and characters as values
        self.h_c_label_map = {k:c for k,c in enumerate(self.h_c_classes, start = 0)}
        # 0 will be reserved for blank
        self.f_c_label_map = {k:c for k,c in enumerate(self.f_c_classes, start = 0)}
        # blank not needed for embedding as it is Binary classification of each diacritic
        self.d_c_label_map = {k:c for k,c in enumerate(self.d_classes, start = 0)}
        
        # dict with characters as keys and class indexes as values
        self.rev_h_c_label_map = {c:k for k,c in enumerate(self.h_c_classes, start = 0)}
        self.rev_f_c_label_map = {c:k for k,c in enumerate(self.f_c_classes, start = 0)}
        self.rev_d_label_map = {c:k for k,c in enumerate(self.d_classes, start= 0)}

    def _normalize_charset(self)-> None:
        """
        Function to normalize the charset provided and converts the charset from list to tuple
        """
        self.h_c_classes = tuple([unicodedata.normalize("NFKD", c) for c in self.h_c_classes])
        self.f_c_classes = tuple([unicodedata.normalize("NFKD", c) for c in self.f_c_classes])
        self.d_classes = tuple([unicodedata.normalize("NFKD", c) for c in self.d_classes])
    
    def _check_h_c(self, label:str, idx:int)-> bool:
        """
        Method to check if the character at index idx in label is a half-char or not
        Returns:
        - bool: True if the current index is a half character or not
        """
        # check if the current char is in h_c_set and next char is halfer
        return idx < len(label) and label[idx] in self.h_c_set \
            and idx + 1 < len(label) and label[idx + 1] == self.halfer

    def _check_f_c(self, label, idx)-> bool:
        """
        Method to check if the character at index idx in label is a full-char or not
        Returns:
        - bool: True if the current idx is a full character
        """
        # check if the current char is in f_c_set and 
        # it is the last char of label or the following char is not halfer
        return idx < len(label) and label[idx] in self.f_c_set \
            and (idx + 1 >= len(label) or label[idx + 1] != self.halfer)

    def _check_d_c(self, label, idx)-> bool:
        """
        Function to check if the character at index idx in label is a diacritic or not
        Returns:
        - bool: True if the current idx is a diacritic
        """
        # check if the char belongs in d_c_set
        return idx < len(label) and label[idx] in self.d_c_set

    @abstractmethod
    def grp_sanity(self, label:str, grps:tuple)-> bool:
        """
        Checks whether the groups are properly formed
        """
        pass

    @abstractmethod
    def label_transform(self, label:str)-> tuple:
        """
        Transform label into group
        """
        pass 

    @abstractmethod
    def grp_class_encoder(self, grp: str)-> tuple:
        """
        Encodes a group into tensors

        Returns:
        - tuple: Containing encodings of various components
        """
        pass

    @abstractmethod
    def label_encoder(self, label:str, device:torch.device)-> tuple:
        """
        Converts the text label into classes indexes for classification
        Args:
        - label (str): The label to be encoded
        - device (torch.device): Device in which the encodings should be saved

        Returns:
        - tuple: tuple of tensors representing various character components in the group
        """
        pass
    
    @abstractmethod
    def _decode_grp(self, **kwarts)-> str:
        """
        Method which takes in class predictions of a single group and decodes
        the group
        Returns:
        - str: the group formed
        """
        pass

    @abstractmethod          
    def decode(self, logits:tuple)-> tuple:
        """
        Method to decode the labels of a batch given the logits
        Args:-
        - logits (tuple): the logits of the model in the order
                                                        
        Returns:
        - tuple: the labels of each batch item
        """
        pass

class HindiTokenizer(BaseTokenizer):
    """
    Class for encoding and decoding Hindi labels
    """
    def grp_sanity(self, label:str, grps:tuple)-> bool:
        """
        Checks whether the groups are properly formed
        for Hindi, each group should contain:
            1) at most 2 half-characters
            2) at most 2 diacritics
            3) 1 full-character
        Args:
        - label (str): label for which groups are provided
        - gprs (tuple): tuple containing the groups of the label

        Returns:
        - bool: True if all groups pass the sanity check else raises an Exception
        """
        for grp in grps:
            # allowed character category counts
            h_c_count, f_c_count, d_c_count = 2, 1, 2
            d_seen = []
            i = 0
            while i < len(grp):
                if i + 1 < len(grp) and self.halfer == grp[i + 1] and grp[i] in self.h_c_set and h_c_count > 0:
                    h_c_count -= 1
                    i += 1
                elif grp[i] in self.f_c_set and f_c_count > 0:
                    f_c_count -= 1
                elif grp[i] in self.d_c_set and d_c_count == 2:
                    d_c_count -= 1
                    d_seen.append(grp[i])
                elif grp[i] in self.d_c_set and d_c_count != 2 and grp[i] not in d_seen:
                    d_c_count -= 1
                elif grp[i] in self.d_c_set and d_c_count != 2 and grp[i] in d_seen:
                    print(f"Duplicate Diacritic in group {grp} for label {label}")
                    return False
                else:
                    if grp[i] not in self.f_c_set or \
                        grp[i] not in self.d_c_set or grp[i] not in self.h_c_set:
                        print(f"Invalid {grp[i]} in group {grp} for label {label}")
                    if f_c_count == 1:
                        print(f"There are no full character in group {grp} for {label}")
                    if (h_c_count, f_c_count, d_c_count) == (2, 1, 2):
                        print(f"Invalid number of half {h_c_count}, full {f_c_count} \
                                        or diacritic characters {d_c_count} in {grp} for {label}")
                    return False  
                i += 1
    
            if f_c_count == 1 or (h_c_count, f_c_count, d_c_count) == (2, 1, 2):
                print(f"There are no full character in group {grp} for {label} OR")
                return False

        return True

    def label_transform(self, label:str)-> tuple:
        """
        Transform hindi labels into groups
        Args:
        - label (str): label to transform
        Returns:
        - tuple: groups of the label
        """
        grps = ()
        running_grp = ""
        idx = 0
        while(idx < len(label)):
            t = idx
            # the group starts with a half-char or a full-char
            if self._check_h_c(label, idx):
                # checks for half-characters
                running_grp += label[idx:idx+2]
                idx += 2 
                # there can be 2 half-char
                if self._check_h_c(label, idx):
                    running_grp += label[idx: idx+2]
                    idx += 2
            
            # half-char is followed by full char
            if self._check_f_c(label, idx):
                # checks for 1 full character
                running_grp += label[idx]
                idx += 1

            # diacritics need not be always present
            if self._check_d_c(label, idx):
                # checks for diacritics
                running_grp += label[idx]
                idx += 1
                # there can be 2 diacritics in a group
                if self._check_d_c(label, idx):
                    running_grp += (label[idx])
                    idx += 1

            if t == idx:
                print(f"Invalid label {label}-{t}")
                return ()
                
            grps = grps + (running_grp, )
            running_grp = ""

        return grps if self.grp_sanity(label, grps) else ()

    def grp_class_encoder(self, grp: str)-> Tuple[int, int, int, Tensor]:
        """
        Encodes the group into class labels for Half-character 2, 1 and full character;
        One-Hot encodes diacritics.
        Args:
        - grp (str): Character group

        Returns:
        - tuple(Tensor, Tensor, Tensor, Tensor): class label for half-character 2,
                                                half-character 1, full-character,
                                                and diacritics (one hot encoded)
        """
        # get the character groupings
        h_c_2_target, h_c_1_target, f_c_target, d_target = (
                                                            self.blank_id,
                                                            self.blank_id,
                                                            self.pad_id,
                                                            torch.tensor(
                                                                [0 for i in range(len(self.d_classes))],
                                                                  dtype= torch.long
                                                                )
                                                            )
        for i, char in enumerate(grp):
            halfer_cntr = 0
            if char in self.h_c_set and i + 1 < len(grp) and grp[i+1] == self.halfer:
                    # for half character occurence
                    if h_c_1_target == self.blank_id:
                        # half_character1 will always track the half-character
                        # which is joined with the complete character
                        h_c_1_target = self.rev_h_c_label_map[char]
                    else: # there are more than 1 half-characters
                        # half_character2 will keep track of half-character 
                        # not joined with the complete character
                        h_c_2_target = h_c_1_target
                        # the first half-char will be joined to the complete character
                        h_c_1_target = self.rev_h_c_label_map[char]

            elif char in self.f_c_set:
                assert f_c_target == self.pad_id,\
                       f"2 full Characters have occured {grp}-{char}"
                f_c_target = self.rev_f_c_label_map[char]

            elif char in self.d_c_set:
                # for diacritic occurence
                assert d_target[self.rev_d_label_map[char]] == 0, \
                    f"2 same matras occured {grp}-{char}-{d_target}"
                
                d_target[self.rev_d_label_map[char]] = 1.

            elif char == self.halfer and halfer_cntr < 2:
                halfer_cntr += 1

            elif char == self.halfer and halfer_cntr >=2 :
                raise Exception(f"More than 2 half-characters occured")

            else:
                raise Exception(f"Character {char} not found in vocabulary")
            
        assert f_c_target != self.pad_id, f"There is no full character {grp}"
        assert torch.sum(d_target, dim = -1) <= 2, f"More than 2 diacritics occured {grp}-{d_target}"

        return h_c_2_target, h_c_1_target, f_c_target, d_target

    def label_encoder(self, label:str, device:torch.device)-> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """
        Converts the text label into classes indexes for classification
        Args:
        - label (str): The label to be encoded
        - device (torch.device): Device in which the encodings should be saved

        Returns:
        - tuple(int, int, int, Tensor): half-char 2 class index, half-char 1
                                        class index, full char class index,
                                        diacritic one hot encoding
        """
        grps = self.label_transform(label= label)
        h_c_2_target, h_c_1_target, f_c_target, d_target = (
                                                            torch.full((self.max_grps,), self.pad_id, dtype= torch.long), 
                                                            torch.full((self.max_grps,), self.pad_id, dtype= torch.long),
                                                            torch.full((self.max_grps,), self.pad_id, dtype= torch.long),
                                                            torch.zeros(self.max_grps, len(self.d_classes), dtype= torch.long)
                                                        )
        d_target[:,self.pad_id] = 1.
        # truncate grps if grps exceed max_grps
        if len(grps) < self.max_grps:
            eos_idx = len(grps)
            # assign eos after the last group
            h_c_2_target[eos_idx], h_c_1_target[eos_idx], f_c_target[eos_idx] = self.eos_id, self.eos_id, self.eos_id
            d_target[eos_idx, self.eos_id] = 1
                                                                                        
        else:
            grps = grps[:self.max_grps]
       
        for idx,grp in enumerate(grps, start= 0):
            h_c_2_target[idx], h_c_1_target[idx], f_c_target[idx], d_target[idx] = self.grp_class_encoder(grp=grp)

        return h_c_2_target.to(device), h_c_1_target.to(device), f_c_target.to(device), d_target.to(device), len(grps)
    
    def _decode_grp(self, h_c_2_pred:Tensor,h_c_1_pred:Tensor, f_c_pred:Tensor, 
                    d_pred:Tensor, d_max:Tensor)-> str:
        """
        Method which takes in class predictions of a single group and decodes
        the group
        Args:
        - h_c_2_pred (Tensor): Index of max logit (prediction) of half-char 2; shape torch.Size(1)
        - h_c_1_pred (Tensor): Index of max logit (prediction) of half-char 1; shape torch.Size(1)
        - f_c_pred (Tensor): Index of max logit (prediction) of full-char; shape torch.Size(1)
        - d_pred (Tensor): Index of 2 probability values > threshold; shape torch.Size(2)
        - d_max (Tensor): The corresponding values of d_pred in binary mask

        Returns:
        - str: the group formed
        """
        assert len(d_pred) == len(d_max) == 2, \
            "Diacritic preds and max must contain 2 elements for 2 diacritics"
        
        grp = ""
        grp += self.h_c_label_map[int(h_c_2_pred.item())] + self.halfer if self.h_c_label_map[int(h_c_2_pred.item())] != HindiTokenizer.BLANK \
            and self.h_c_label_map[int(h_c_2_pred.item())] != HindiTokenizer.PAD else ""
        grp += self.h_c_label_map[int(h_c_1_pred.item())] + self.halfer if self.h_c_label_map[int(h_c_1_pred.item())] != HindiTokenizer.BLANK \
            and self.h_c_label_map[int(h_c_1_pred.item())] != HindiTokenizer.PAD else ""
        grp += self.f_c_label_map[int(f_c_pred.item())]
        grp += self.d_c_label_map[int(d_pred[0].item())] if d_max[0] else ""
        grp += self.d_c_label_map[int(d_pred[1].item())] if d_max[1] else ""
                
        return grp.replace(HindiTokenizer.BLANK, "").replace(HindiTokenizer.PAD, "") # remove all [B], [P] occurences
                
    def decode(self, logits:Tuple[Tensor, Tensor, Tensor, Tensor])-> tuple:
        """
        Method to decode the labels of a batch given the logits
        Args:-
        - logits (tuple(Tensor, Tensor, Tensor, Tensor)): the logits of the model in the order
                                                        half-char 2, half-char 1, full-char, diacritic
                                                        logits
        Returns:
        - tuple: the labels of each batch item
        """
        # logits shape: BS x Max Grps x # of classes
        (h_c_2_logits, h_c_1_logits, f_c_logits, d_logits) = logits
        batch_size = h_c_2_logits.shape[0]

        # greedy decoding for half and full chars
        h_c_2_preds = torch.argmax(h_c_2_logits, dim= 2)
        h_c_1_preds = torch.argmax(h_c_1_logits, dim= 2)
        f_c_preds = torch.argmax(f_c_logits, dim= 2)

        # threshold filter for diacritic
        d_bin_mask = nn.functional.sigmoid(d_logits) > self.thresh
        d_max, d_preds = torch.topk(d_bin_mask.int(), k= 2, dim= 2, largest= True) # top k requires scalar

        # get the predictions
        pred_labels = []    
        for i in range(batch_size):
            label = ""
            for j in range(self.max_grps):
                grp = self._decode_grp(
                                h_c_2_pred= h_c_2_preds[i,j],
                                h_c_1_pred= h_c_1_preds[i,j],
                                f_c_pred= f_c_preds[i,j],
                                d_pred= d_preds[i,j],
                                d_max= d_max[i,j]
                            )
                if HindiTokenizer.EOS in grp:
                    break
                else:
                    label += grp
                    
            pred_labels.append(label)
            label = ""
        
        return tuple(pred_labels)

class MalayalamTokenizer(BaseTokenizer):
    """
    Class for encoding and decoding labels
    """
    def __init__(self, chill:list, special_matra:list, half_character_classes:list, full_character_classes:list, diacritic_classes:list,
                halfer:str, threshold:float= 0.5, max_grps:int= 25):
        """
        Args:
        - chill (list): chillaksharam list
        - special_matra (list): special diacritics
        - half_character_set (list)
        - full_character_set (list)
        - diacritic_set (list)
        - halfer (str): diacritic used to represent a half-character
        - threshold (float): classification threshold (default: 0.5)
        """     
        super().__init__(half_character_classes= half_character_classes, full_character_classes= full_character_classes,
                         diacritic_classes= diacritic_classes, halfer= halfer, threshold= threshold, max_grps= max_grps)
        self.chill= chill
        self.special_matra= special_matra
    
    def _normalize_charset(self)-> None:
        """
        Function to normalize the charset provided and converts the charset from list to tuple
        """
        self.h_c_classes = tuple([unicodedata.normalize("NFKD", c) for c in self.h_c_classes])
        self.f_c_classes = tuple([unicodedata.normalize("NFKD", c) for c in self.f_c_classes])
        self.d_classes = tuple([unicodedata.normalize("NFKD", c) for c in self.d_classes])

    def _check_h_c(self, label:str, idx:int)-> bool:
        """
        Method to check if the character at index idx in label is a half-char or not
        Returns:
        - bool: True if the current index is a half character or not
        """
        # check if the current char is in h_c_set and next char is halfer
        return idx < len(label) and label[idx] in self.h_c_set \
                and idx + 1 < len(label) and label[idx + 1] == self.halfer

    def _check_f_c(self, label, idx):
        """
        Method to check if the character at index idx in label is a full-char or not
        Returns:
        - bool: True if the current idx is a full character
        """
        # check if the current char is in f_c_set and 
        # it is the last char of label or the following char is not halfer
        return idx < len(label) and label[idx] in self.f_c_set \
            and (idx + 1 >= len(label) or label[idx + 1] != self.halfer)

    def _check_d_c(self, label, idx):
        """
        Function to check if the character at index idx in label is a diacritic or not
        Returns:
        - bool: True if the current idx is a diacritic
        """
        # check if the char belongs in d_c_set
        return idx < len(label) and label[idx] in self.d_c_set
    
    def grp_sanity(self, label:str, grps:tuple)-> bool:
        """
        Checks whether the groups are properly formed
        for Malayalam, each group should contain:
            1) at most 3 half-characters
            2) at most 2 diacritics
            3) at most 1 full-character
        Args:
        - label (str): label for which groups are provided
        - gprs (tuple): tuple containing the groups of the label

        Returns:
        - bool: True if all groups pass the sanity check else raises an Exception
        """
        for grp in grps:
            # allowed character category counts
            h_c_count, f_c_count, d_c_count = 3, 1, 2
            d_seen = []
            i = 0
            while i < len(grp):
                if grp[i] in self.chill and i+1 != len(grp):
                    print(f"chill not the ending in the group {grp} for label {label}")
                    return False
                else:
                    if i + 1 < len(grp) and self.halfer == grp[i + 1] and grp[i] in self.h_c_set and h_c_count > 0:
                        h_c_count -= 1
                        i += 1
                    elif grp[i] in self.f_c_set and f_c_count > 0:
                        f_c_count -= 1
                    elif grp[i] in self.d_c_set and d_c_count == 2:
                        d_c_count -= 1
                        d_seen.append(grp[i])
                    elif grp[i] in self.d_c_set and d_c_count != 2 and grp[i] not in d_seen:
                        d_c_count -= 1
                    elif grp[i] in self.d_c_set and d_c_count != 2 and grp[i] in d_seen:
                        print(f"Duplicate Diacritic in group {grp} for label {label}")
                        return False
                    elif grp[i]== '്' and grp[i-1]== 'ു':
                        i+=1
                    else:
                        if grp[i] not in self.f_c_set or \
                            grp[i] not in self.d_c_set or grp[i] not in self.h_c_set:
                            print(f"Invalid {grp[i]} in group {grp} for label {label}")
                        if (h_c_count, f_c_count, d_c_count) == (3, 1, 2):
                            print(f"Invalid number of half {h_c_count}, full {f_c_count} \
                                            or diacritic characters {d_c_count} in {grp} for {label}")
                        return False  
                    i += 1
    
            if f_c_count == 1 or (h_c_count, f_c_count, d_c_count) == (2, 1, 2):
                if grp[len(grp)-1:] == self.halfer:
                    return True
                print(f"There are no full character in group {grp} for {label} at {grps} OR")
                return False

        return True

    def label_transform(self, label:str)-> tuple:
        """
        Transform Malayalam labels into groups
        Args:
        - label (str): label to transform
        Returns:
        - tuple: groups of the label
        """
        grps = ()
        running_grp = ""
        idx = 0
        while(idx < len(label)):
            t = idx
            # the group starts with a half-char or a full-char
            if label[idx] in self.chill:
                running_grp+=label[idx]
                idx += 1
            else:
                if self._check_h_c(label, idx):
                    # checks for half-characters
                    running_grp += label[idx:idx+2]
                    idx += 2
                    # there can be 3 half-char
                    if self._check_h_c(label, idx):
                        running_grp += label[idx: idx+2]
                        idx += 2
                        if self._check_h_c(label, idx):
                            running_grp += label[idx: idx+2]
                            idx += 2
                
                # half-char is followed by full char
                if self._check_f_c(label, idx): # suppose matra occurs after chill then check in sanity-check
                    # checks for 1 full character  
                    running_grp += label[idx]
                    idx += 1
                # print("yes",running_grp)
                # diacritics need not be always present
                if self._check_d_c(label, idx):
                    # checks for diacritics
                    running_grp += label[idx]
                    idx += 1
                    if idx < len(label) and label[idx-1]=='ു' and label[idx]=='്':
                        running_grp += (label[idx])
                        idx += 1
                    # there can be 1 diacritics in a group + (am or aha ) + 
                    if idx < len(label) and label[idx] in self.special_matra:
                        running_grp += (label[idx])
                        idx += 1
                    if idx < len(label) :
                        if label[idx-1] == 'െ':
                            if label[idx] in ['െ', 'ൗ','ാ']:
                                running_grp += (label[idx])
                                idx += 1
                        elif label[idx-1] == 'േ' and label[idx] == 'ാ':
                            running_grp += (label[idx])
                            idx += 1
                if t == idx:
                    print(f"  {label} is Invalid label because of {label[t]} after {label[t-1]}  at index {t} ")
                    return ()
            if running_grp!="":
                grps = grps + (running_grp, ) 
                # print(grps)
            running_grp = ""
        return grps if self.grp_sanity(label, grps) else ()