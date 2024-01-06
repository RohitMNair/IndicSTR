import lightning.pytorch as pl
import lmdb
import torch
import unicodedata
import io

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
from pathlib import Path
from typing import Tuple
from utils.transforms import LabelTransform

class HindiLMDBDataset(Dataset):
    def __init__(self, data_dir: str, charset:list or tuple, diacritics:list or tuple, 
                halfer:str, half_charset:list or tuple, transforms: transforms.Compose,
                max_grps:int= 25):
        super().__init__()
        self._t_env = None
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.f_c = ["[B]"] + charset
        self.f_c_set = set(self.f_c)
        self.d_c = diacritics
        self.d_c_set = set(self.d_c)
        self.halfer = halfer
        self.h_c = ["[B]"] + half_charset
        self.h_c_set = set(self.h_c)
        self.max_grps = max_grps
        self.items = []
        self.num_samples = self._preprocess_labels()
        self.processed_indexes = []

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
            self._t_env = self._create_env(self.data_dir)
        return self._t_env
    
    def _preprocess_labels(self):
        with self._create_env(self.data_dir) as env, env.begin() as txn:
            num_samples = int(txn.get('num-samples'.encode()))

            for index in range(num_samples):
                index += 1  # lmdb starts with 1
                label_key = f'label-{index:09d}'.encode()
                label = txn.get(label_key).decode()
                label = label.strip()
                label = ''.join(label.split()) # remove any white-spaces
                # normalize unicode to remove redundant representations
                label = unicodedata.normalize('NFKD', label)
                # save the label and corresponding index
                self.items.append(label)
                self.processed_indexes.append(index)
        print("Length of labels ", len(self.items))
        return len(self.items)

    def __getitem__(self, index)-> Tuple[Tensor, str]:
        label = self.items[index] 
        # get corresponding index for the label groups
        index = self.processed_indexes[index] 
        img = None
        to_tensor = transforms.ToTensor()
        # keys assigned as per create_lmdb.py
        img_key = f'image-{index:09d}'.encode()
        
        with self.t_env.begin() as txn:
            imgbuf = txn.get(img_key)
            buf = io.BytesIO(imgbuf)
            img = Image.open(buf).convert('RGB')
            img = to_tensor(img)
            if self.transforms is not None:
                img = self.transforms(img)
        return img, label