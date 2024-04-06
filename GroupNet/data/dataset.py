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
from .tokenizer import MalayalamTokenizer, DevanagariTokenizer, HindiTokenizer

class LMDBDataset(Dataset):
    def __init__(self, data_dir: str, transforms: transforms.Compose,
                 language:str, remove_unseen:bool= False):
        super().__init__()
        self._env = None
        self.data_dir = data_dir
        self.language = language.lower()
        self.transforms = transforms
        self.remove_unseen = remove_unseen
        self.items = []
        self.processed_indexes = []
        if self.language== 'devanagari':
            self.tokenizer = DevanagariTokenizer()
        elif self.language == 'hindi':
            self.tokenizer = HindiTokenizer()
        elif self.language == 'malayalam':
            self.tokenizer = MalayalamTokenizer()
        else:
            raise NotImplementedError(f"Language {self.language} not implemented")
        
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
                    label = ''.join(c for c in label if c in (self.tokenizer.get_charset()))
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
            if self.transforms is not None:
                img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.items)
