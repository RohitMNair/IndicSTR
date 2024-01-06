from torchvision import transforms
from torch import Tensor

import torch

class RescaleTransform:
    """
    Rescale the image in a sample to a given size.
    """
    def __init__(self, output_size):
        """
        Initialize a new Rescale transform
        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size. If int, smaller of image edges is matched
                to output_size keeping aspect ratio the same.
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image)-> Tensor:
            h, w = image.shape[1:]
            if isinstance(self.output_size, int):
                new_h, new_w = self.output_size, self.output_size
                aspect_ratio = w / h
                if w > h:
                    new_h = int(new_w / aspect_ratio)
                else:
                    new_w = int(new_h * aspect_ratio)
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)
            image = transforms.Resize((new_h, new_w), antialias=True)(image)
            return image

class PadTransform:
    """
    Pad the image in a sample to a given size.
    """
    def __init__(self, output_size):
        """
        Initializes a new pad transform
        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size. If int, the pad will be a square.
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, img)-> Tensor:
        if isinstance(self.output_size, int):
          background = torch.zeros((3, self.output_size, self.output_size))
          background[:,0:img.shape[1],0:img.shape[2]] += img
        else:
          channels, height, width = 3, self.output_size[0], self.output_size[1]
          background = torch.zeros((channels, height, width))
          background[:, 0:img.shape[1], 0:img.shape[2]] += img
        return background
    
class LabelTransform:
    """
    Transforms the labels for Hindi, Malayalam into character groups
    """
    def __init__(self, half_character_set: set, full_character_set: set, diacritic_set:set, halfer:str):
        """
        Transforms words into character groups
        Args:
            half_character_set (set): Half character classes set
            full_character_set (set): Full character classes set
            diacritic_set (set): Diacritic classes set
            halfer (str): Diacritic used for representing Half-character
        """
        self.h_c_set = half_character_set
        self.f_c_set = full_character_set
        self.d_c_set = diacritic_set
        self.halfer = halfer

    def hindi_grp_sanity(self, label:str, grps:tuple)-> bool:
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
            for char in grp:

                if self.halfer == char[1] and char[0] in self.h_c_set and h_c_count > 0:
                    h_c_count -= 1
                elif char in self.f_c_set and f_c_count > 0:
                    f_c_count -= 1
                elif char in self.d_c_set and d_c_count > 0:
                    d_c_count -= 1
                else:
                    if char not in self.f_c_set or \
                        char not in self.d_c_set or char not in self.h_c_set:
                        raise Exception(f"Invalid {char} in group {grp} for label {label}")
                    elif f_c_count == 1:
                        raise Exception(f"There are no full character in group {grp} for {label}")
                    else:
                        raise Exception(f"Invalid number of half {h_c_count}, full {f_c_count} \
                                        or diacritic characters {d_c_count} in {grp} for {label}")      
        return True

    def check_h_c(self, label, idx):
        """
        Returns:
        - bool: True if the current index is a half character or not
        """
        # check if the current char is in h_c_set and next char is halfer
        return idx < len(label) and label[idx] in self.h_c_set \
            and idx + 1 < len(label) and label[idx + 1] == self.halfer

    def check_f_c(self, label, idx):
        """
        Returns:
        - bool: True if the current idx is a full character
        """
        # check if the current char is in f_c_set and 
        # it is the last char of label or the following char is not halfer
        return idx < len(label) and label[idx] in self.f_c_set \
            and (idx + 1 >= len(label) or label[idx + 1] != self.halfer)

    def check_d_c(self, label, idx):
        """
        Returns:
        - bool: True if the current idx is a diacritic
        """
        # check if the char belongs in d_c_set
        return idx < len(label) and label[idx] in self.d_c_set

    def hindi_label_transform(self, label:str)-> tuple:
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
            # the group starts with a half-char or a full-char
            if self.check_h_c(label, idx):
                # checks for half-characters
                running_grp += label[idx:idx+2]
                idx += 2 
                # there can be 2 half-char
                if self.check_h_c(label, idx):
                    running_grp += label[idx: idx+2]
                    idx += 2
            
            # half-char is followed by full char
            if self.check_f_c(label, idx):
                # checks for 1 full character
                running_grp += label[idx]
                idx += 1

            # diacritics need not be always present
            if self.check_d_c(label, idx):
                # checks for diacritics
                running_grp += label[idx]
                idx += 1
                # there can be 2 diacritics in a group
                if self.check_d_c(label, idx):
                    running_grp += (label[idx])
                    idx += 1
                
            grps = grps + (running_grp, )
            running_grp = ""

        return grps if self.hindi_grp_sanity(label, grps) else ()
