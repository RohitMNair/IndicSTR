import lightning.pytorch as pl
import lmdb
import torch
import unicodedata
import io

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple
from .tokenizer import HindiTokenizer, MalayalamTokenizer, DevanagariTokenizer

class HindiLMDBDataset(Dataset):
    def __init__(self, data_dir: str, transforms: transforms.Compose,
                 half_character_classes:list, full_character_classes:list, 
                 diacritic_classes:list, halfer:str, remove_unseen:bool= False):
        super().__init__()
        self._env = None
        self.data_dir = data_dir
        self.transforms = transforms
        self.remove_unseen = remove_unseen
        self.items = []
        self.processed_indexes = []
        self.tokenizer = HindiTokenizer(
            half_character_classes= half_character_classes,
            full_character_classes= full_character_classes,
            diacritic_classes= diacritic_classes,
            halfer= halfer,
            threshold= 0.5,
            max_grps= 50,
        )
        self.num_samples = self._preprocess_labels()

    def __del__(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _create_env(self, root):
        return lmdb.open(root, max_readers=1, readonly=True, create=False,
                         readahead=False, meminit=False, lock=False)

    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env(self.data_dir)
        return self._env

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
                # remove other characters
                if self.remove_unseen:
                    label = ''.join(c for c in label if c in (self.tokenizer.f_c_classes + \
                                    self.tokenizer.h_c_classes + self.tokenizer.d_classes + (self.tokenizer.halfer, )))
                if index % 100000 == 0:
                    print(f"Processed {index} number of labels", flush = True)

                if len(self.tokenizer.label_transform(label)) == 0:
                    # The label is does not follow the group convention
                    continue
                else:
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
        
        with self.env.begin() as txn:
            imgbuf = txn.get(img_key)
            buf = io.BytesIO(imgbuf)
            img = Image.open(buf).convert('RGB')
            img = to_tensor(img)
            if self.transforms is not None:
                img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.items)

class DevanagariLMDBDataset(Dataset):
    def __init__(self, data_dir: str, transforms: transforms.Compose,
                 svar:list, vyanjan:list, matras:list, ank:list, chinh:list,
                nukthas:list, halanth:str, remove_unseen:bool= False):
        super().__init__()
        self._env = None
        self.data_dir = data_dir
        self.transforms = transforms
        self.remove_unseen = remove_unseen
        self.items = []
        self.processed_indexes = []
        self.tokenizer = DevanagariTokenizer(
                            svar = svar,
                            vyanjan= vyanjan,
                            matras= matras,
                            ank= ank,
                            chinh= chinh,
                            nukthas= nukthas,
                            halanth= halanth,
                        )
        self.num_samples = self._preprocess_labels()

    def __del__(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _create_env(self, root):
        return lmdb.open(root, max_readers=1, readonly=True, create=False,
                         readahead=False, meminit=False, lock=False)

    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env(self.data_dir)
        return self._env

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
                # remove other characters
                if self.remove_unseen:
                    label = ''.join(c for c in label if c in (self.tokenizer.f_c_classes + \
                                    self.tokenizer.h_c_classes + self.tokenizer.d_classes + (self.tokenizer.halfer, )))
                if index % 100000 == 0:
                    print(f"Processed {index} number of labels", flush = True)

                if len(self.tokenizer.label_transform(label)) == 0:
                    # The label is does not follow the group convention
                    continue
                else:
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
        
        with self.env.begin() as txn:
            imgbuf = txn.get(img_key)
            buf = io.BytesIO(imgbuf)
            img = Image.open(buf).convert('RGB')
            img = to_tensor(img)
            if self.transforms is not None:
                img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.items)