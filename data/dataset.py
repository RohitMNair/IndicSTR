import torch
import pandas as pd
import lightning.pytorch as pl
import lmdb
import io
import random
from typing import Optional
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from PIL import Image

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
        false_sample_dir (str): Path to false samples image directory
        false_sample_gt (str): Path to false sample ground truth .txt file containing image names
        false_weight (int): amount of false samples, false_weight x false_samples
    """
    def __init__(self, img_dir: str, gt_file: str, charset:list or tuple, diacritics:list or tuple, 
                halfer:str, half_charset:list or tuple, transforms: transforms.Compose, separator:str = ' ',
                false_sample_dir:Optional[str] = None, false_sample_gt:Optional[str] = None, false_weight = 1):
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
        self.false_weight = false_weight

        if false_sample_dir and false_sample_gt:
            self.false_sample_dir = Path(false_sample_dir)
            self.false_samples = pd.read_csv(Path(false_sample_gt), header = None)
            self.false_samples[1] = "" # create a new column for empty labels
            self.gt_file = pd.concat([self.gt_file] + self.false_weight*[self.false_samples], axis = 0) # concat the dfs
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
        if len(label) == 0:
            image_path = self.false_sample_dir / f"{img_name}"
        else:
            image_path = self.img_dir / f"{img_name}"
        img = read_image(image_path.as_posix()) / 255.

        if len(label) == 0:
            # it is a negative sample, therefore random crop
            resized_crop = transforms.RandomResizedCrop(
                size = (img.shape[1], img.shape[2]), # keep the image size
                scale = (0.1, 1),
                ratio = (0.5, 1.5),
                antialias = True
            )
            img = resized_crop(img)

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

class LMDBDevanagariDataset(Dataset):
    """
    Devanagari Character Group Dataset, expects the input ground truth
    file to contain a single Devanagari character group
    Args:
        img_dir: the path to the image LMDB directory
        false_sample_dir (optional): path to negative samples LMDB directory
        charset (list): Full character list
        diacritics (list): Diacritics ist
        halfer (str): the halfer character (eg. Halanth in Devanagari)
        half_charset (list): List of Half characters
        separator: delimiter for the ground truth txt file
        transfroms: Torchvision image transforms image transformations
        false_sample_dir (str): false samples directory
    """
    def __init__(self, img_dir: str, charset:list or tuple, diacritics:list or tuple, 
                halfer:str, half_charset:list or tuple, transforms: transforms.Compose,
                false_sample_dir: Optional[str] = None, false_weight = 1):
        self._t_env = None
        self._f_env = None
        self.root = img_dir
        self.transforms = transforms
        self.characters = charset
        self.charset = set(self.characters)
        self.diacritics = diacritics
        self.diac_set = set(self.diacritics)
        self.halfer = halfer
        self.half_characters = half_charset
        self.half_charset = set(self.half_characters)
        self.false_dir = false_sample_dir
        self.false_weight = false_weight
        self.items = []
        self.num_samples = self._preprocess_labels()

        
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
        
    def __del__(self):
        if self._t_env is not None:
            self._t_env.close()
            self._env = None

        if self._f_env is not None:
            self._f_env.close()
            self._f_env = None

    def _create_env(self, root):
        return lmdb.open(root, max_readers=1, readonly=True, create=False,
                         readahead=False, meminit=False, lock=False)

    @property
    def t_env(self):
        if self._t_env is None:
            self._t_env = self._create_env(self.root)
        return self._t_env
    
    @property
    def f_env(self):
        if self._f_env is None and self.false_dir is not None:
            self._f_env = self._create_env(self.false_dir)
        return self._f_env

    def _preprocess_labels(self):
        with self._create_env(self.root) as env, env.begin() as txn:
            num_samples = int(txn.get('num-samples'.encode()))

            for index in range(num_samples):
                index += 1  # lmdb starts with 1
                label_key = f'label-{index:09d}'.encode()
                label = txn.get(label_key).decode()
                label = label.strip()
                if len(label) > 7:
                    # raise Exception("Goup longer than 7 characters encountered")
                    print("Goup longer than 7 characters encountered", label)
                    continue
                self.items.append((True, index, label))

        if self.false_dir is not None:
            with self._create_env(self.false_dir) as env, env.begin() as txn:
                for i in range(self.false_weight):
                    num_samples = int(txn.get('num-samples'.encode()))
                    for index in range(num_samples):
                        index += 1
                        label_key = f'label-{index:09d}'.encode()
                        label = txn.get(label_key).decode()
                        label = label.strip()
                        self.items.append((False, index, label))
        random.shuffle(self.items)
        print("Length of Labels", len(self.items))
        return len(self.items)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        item = self.items[index]
        img, label = None, None
        if item[0]:
            img_key = f'image-{item[1]:09d}'.encode()
            with self.t_env.begin() as txn:
                imgbuf = txn.get(img_key)
            buf = io.BytesIO(imgbuf)
            img = Image.open(buf).convert('RGB')
            to_tensor = transforms.ToTensor()
            img = to_tensor(img)
            if self.transforms is not None:
                img = self.transforms(img)

            label = item[2]
        
        else:
            img_key = f'image-{item[1]:09d}'.encode()
            with self.f_env.begin() as txn:
                imgbuf = txn.get(img_key)
            buf = io.BytesIO(imgbuf)
            img = Image.open(buf).convert('RGB')
            to_tensor = transforms.ToTensor()
            img = to_tensor(img)
            if self.transforms is not None:
                img = self.transforms(img)
            label = item[2]
        
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
        