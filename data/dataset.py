import torch
import torchvision
import pandas as pd
import cv2 as cv
import numpy as np
import lightning.pytorch as pl
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms, utils

class DevanagariDataset(Dataset):
    """
    Devanagari Character Group Dataset, expects the input ground truth
    file to contain a single Devanagari character group
    Args:
        img_dir: the path to the image directory w.r.t which the groud truth file
                image paths
        gt_file: path to the ground truth file
        false_sample_dir (optional): path to negative samples
        false_sample_gt (optional): path to negative samples file names text file
        charset (list): Full character list
        diacritics (list): Diacritics ist
        halfer (str): the halfer character (eg. Halanth in Devanagari)
        half_charset (list): List of Half characters
        separator: delimiter for the ground truth txt file
        transfroms: Image transformations
    """
    def __init__(self, img_dir: str, gt_file: str, charset:list or tuple, diacritics:list or tuple, 
                halfer:str, half_charset:list or tuple, transforms: transforms.Compose, separator:str = ' ',
                false_sample_dir: str = None, false_sample_gt:str = None):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.gt_file = pd.read_csv(Path(gt_file), delimiter = separator, header = None)
        self.transforms = transforms
        self.characters = charset
        self.charset = set(self.characters)
        self.diacritics = diacritics
        self.diac_set = set(self.diacritics)
        self.halfer = halfer
        self.half_characters = half_charset
        self.half_charset = set(self.half_characters)

        if false_sample_dir and false_sample_gt:
            self.false_sample_dir = Path(false_sample_dir)
            self.false_samples = pd.read_csv(Path(false_sample_gt), header = None)
            self.false_samples[1] = "" # create a new column for empty labels
            self.gt_file = pd.concat([self.gt_file, self.false_samples], axis = 0) # concat the 2 dfs
            self.gt_file = self.gt_file.sample(frac = 1).reset_index(drop = True) # shuffle the df

        # dict with class indexes as keys and characters as values
        self.char_label_map = {k:c for k,c in enumerate(self.characters, start = 1)}
        # blank not needed for embedding as it is Binary classification of each diacritic
        self.diacritic_label_map = {k:c for k,c in enumerate(self.diacritics, start = 0)}
        # 0 will be reserved for blank
        self.half_char_label_map = {k:c for k,c in enumerate(self.half_characters, start = 1)}

        # dict with characters as keys and class indexes as values
        self.rev_char_label_map = {c:k for k,c in enumerate(self.characters, start = 1)}
        self.rev_diacritirc_label_map = {c:k for k,c in enumerate(self.diacritics, start = 0)}
        self.rev_half_char_label_map = {c:k for k,c in enumerate(self.half_characters, start = 1)}
        print("Class Maps",self.char_label_map, self.diacritic_label_map, self.half_char_label_map, self.halfer, flush = True)

    def __getitem__(self, idx):
        img_name, label = self.gt_file.iloc[idx]
        image_path = self.img_dir / f"{img_name}"
        img = read_image(image_path.as_posix()) / 255.

        if self.transforms is not None:
            img = self.transforms(img)
        
        # get the character groupings
        character, half_character1, half_character2, diacritic = 0, 0, 0, torch.tensor([0. for i in range(len(self.diacritics))])
        for i, char in enumerate(label):
            halfer_cntr = 0
            if char in self.half_charset and i + 1 < len(label) and label[i+1] == self.halfer:
                    # for half character occurence
                    if half_character1 == 0:
                        # half_character1 will always track the half-character
                        # which is joined with the complete character
                        half_character1 = self.rev_half_char_label_map[char]
                    else: # there are more than 1 half-characters
                        # half_character2 will keep track of half-character 
                        # not joined with the complete character
                        half_character2 = half_character1
                        # the second half-char will be joined to the complete character
                        half_character1 = self.rev_half_char_label_map[char]

            elif char in self.charset:
                assert character == 0,\
                       f"2 full Characters have occured {label}-{char}-{character}-{half_character1}-{half_character2}"
                character = self.rev_char_label_map[char]

            elif char in self.diac_set:
                # for diacritic occurence
                assert diacritic[self.rev_diacritirc_label_map[char]] == 0.0, f"2 same matras occured {label}-{char}-{diacritic}"
                diacritic[self.rev_diacritirc_label_map[char]] = 1.

            elif char == self.halfer and halfer_cntr < 2:
                halfer_cntr += 1

            elif char == self.halfer and halfer_cntr >=2 :
                raise Exception(f"More than 2 half-characters occured")

            else:
                raise Exception(f"Character {char} not found in vocabulary")
        
        assert torch.sum(diacritic, dim = -1) <= 2, f"More than 2 diacritics occured {label}-{diacritic}"

        return img, half_character2, half_character1, character, diacritic
                       
    def __len__(self):
        return self.gt_file.shape[0]