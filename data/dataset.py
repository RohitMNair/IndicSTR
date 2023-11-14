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
        sperator: delimiter for the ground truth txt file
        transfroms: Image transformations
        normalize: whether to divide the image by 255
    """
    def __init__(self, img_dir: str, gt_file: str, charset:list, diacritics:list, halfer:str,
                transforms: transforms.Compose = None, seperator:str = ' ', normalize = False):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.gt_file = pd.read_csv(Path(gt_file), delimiter = seperator, header = None)
        self.transforms = transforms
        self.normalize = normalize
        self.characters = charset
        self.charset = set(self.characters)
        self.diacritics = diacritics
        self.halfer = halfer
        self.diac_set = set(self.diacritics)

        # dict with class indexes as keys and characters as values
        self.char_label_map = {k:c for k,c in enumerate(self.characters)}
        self.diacritic_label_map = {k:c for k,c in enumerate(self.diacritics)}

        # dict with characters as keys and class indexes as values
        self.rev_char_label_map = {c:k for k,c in enumerate(self.characters)}
        self.rev_diacritirc_label_map = {c:k for k,c in enumerate(self.diacritics)}
        
    def __getitem__(self, idx):
        img_name, label = self.gt_file.iloc[idx]
        image_path = self.img_dir / f"{img_name}"
        img = read_image(image_path.as_posix())

        if self.normalize:
            img = img / 255

        if self.transforms is not None:
            img = self.transforms(img)
        
        # get the character groupings
        character, half_character, diacritic = 0, 0, torch.tensor([0. for i in self.diacritics])
        for i, char in enumerate(label):
            if char in self.charset:
                if i + 1 < len(label) and label[i+1] == self.halfer:
                    # for half character occurence
                    half_character = self.rev_char_label_map[char]
                else:
                    character = self.rev_char_label_map[char]
            
            elif char in self.diac_set:
                # for diacritic occurence
                assert diacritic[self.rev_diacritirc_label_map[char]] != 1, "2 same matras occured"
                diacritic[self.rev_diacritirc_label_map[char]] = 1.
            
            elif char in self.charset:
                # for character
                character = self.rev_char_label_map[char]
            else:
                raise Exception(f"Character {char} not found in vocabulary")
        return img, half_character, character, diacritic
                       
    def __len__(self):
        return self.gt_file.shape[0]
    

class VyanjanDataset(Dataset):
    """
    Custom Dataset loader, which load the images and perform transforms on them
    Load their corresponding labels
    Args:
        img_dir: the path to the image directory w.r.t which the groud truth file
                image paths
        gt_file: path to the ground truth file
        sperator: delimiter for the ground truth txt file
        transfroms: Image transformations
        normalize: whether to divide the image by 255
    """
    def __init__(self, img_dir: str, gt_file: str, transforms = None,  
                seperator: str = ' ', normalize = True):
        super(VyanjanDataset, self).__init__()
        self.img_dir = Path(img_dir)
        self.gt_file = pd.read_csv(Path(gt_file), delimiter = seperator, header = None)
        self.transforms = transforms
        self.normalize = normalize
        self.charset = (
            'क', 'क़', 'ख', 'ख़', 'ग़', 'ज़', 'ग', 'घ', 'ङ', 'ड़', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट',
            'ठ', 'ड', 'ढ', 'ढ़', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'ऩ', 'प', 'फ', 'फ़',
            'ब', 'भ', 'म', 'य', 'य़', 'र', 'ऱ', 'ल', 'ळ', 'ऴ', 'व', 'श', 'ष',
            'स', 'ह', 'ऋ', 'ॠ', 'ॸ', 'ॹ', 'ॺ', 'ॻ', 'ॼ', 'ॾ', 'ॿ',
        )

        self.label_map = {k: self.charset[k] for k in range(len(self.charset))}
        self.rev_label_map = {self.charset[k]:k for k in range(len(self.charset))}

    def __getitem__(self, idx):
        img_name, label = self.gt_file.iloc[idx]
        image_path = self.img_dir / f"{img_name}"
        img = read_image(image_path.as_posix())

        if self.normalize:
            img = img / 255

        if self.transforms:
          img = self.transforms(img)

        return img, self.rev_label_map[label]

    def __len__(self):
        return self.gt_file.shape[0]