import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import DevanagariLMDBDataset, HindiLMDBDataset
from typing import Optional

class LMDBDevanagariDataModule(pl.LightningDataModule):
    """
    Data Module for Devanagari Character Groups
    Args:
        train_dir (str): Path to Train Images LMDB directory
        val_dir (str): Path to Validation Images LMDB directory
        test_dir (Optional[str]): Path to Test LMDB Directory
        transforms (Torchvision.transforms.Compose): Transforms to be applied on the Images
        batch_size (int): Batch Size
        num_workers (int): Number of processes for the DataSet and DataLoader
        drop_last (bool): Whether to drop the last batch if its not divisible by the batch size
    """
    def __init__(self, train_dir:Optional[str], val_dir:Optional[str], test_dir:Optional[str],
                svar:list, vyanjan:list, matras:list, ank:list, chinh:list,
                halanth:str, transforms: transforms.Compose, 
                test_transforms: transforms.Compose, batch_size:int =32, num_workers:int= 4,
                drop_last:bool= False, remove_unseen:bool= False
                ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.transforms = transforms
        self.test_transforms = test_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.svar = svar
        self.vyanjan= vyanjan
        self.matras= matras
        self.ank= ank
        self.chinh= chinh
        self.halanth= halanth
        self.remove_unseen = remove_unseen

        #non-parametric attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage):
        if self.train_dir:
            self.train_dataset = DevanagariLMDBDataset(
                                    data_dir= self.train_dir,
                                    transforms= self.transforms,
                                    svar= self.svar,
                                    vyanjan= self.vyanjan,
                                    matras= self.matras,
                                    ank= self.ank,
                                    chinh= self.chinh,
                                    halanth= self.halanth,
                                    remove_unseen = self.remove_unseen,
                                )
            
        if self.val_dir:
            self.val_dataset = DevanagariLMDBDataset(
                                    data_dir= self.val_dir,
                                    transforms= self.transforms,
                                    svar= self.svar,
                                    vyanjan= self.vyanjan,
                                    matras= self.matras,
                                    ank= self.ank,
                                    chinh= self.chinh,
                                    halanth= self.halanth,
                                    remove_unseen = self.remove_unseen,
                                )
            
        if self.test_dir:
            self.test_dataset = DevanagariLMDBDataset(
                                data_dir= self.test_dir,
                                transforms= self.test_transforms,
                                svar= self.svar,
                                vyanjan= self.vyanjan,
                                matras= self.matras,
                                ank= self.ank,
                                chinh= self.chinh,
                                halanth= self.halanth,
                                remove_unseen = self.remove_unseen,
                            )

    def train_dataloader(self):
        if self.train_dataset:
            return DataLoader(
                        dataset = self.train_dataset,
                        batch_size=self.batch_size, 
                        num_workers = self.num_workers,
                        drop_last = self.drop_last,
                        shuffle=True,
                    )
        else:
            raise Exception("train_dir not provided")

    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(
                        dataset= self.val_dataset, 
                        batch_size= self.batch_size, 
                        num_workers = self.num_workers,
                        drop_last = self.drop_last,   
                    )
        else:
            raise Exception("val_dir not provided")

    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(
                        self.test_dataset, 
                        batch_size= self.batch_size, 
                        num_workers = self.num_workers,
                        drop_last = self.drop_last,
                    )
        else:
            raise Exception("test_dir not provided")

class LMDBHindiDataModule(pl.LightningDataModule):
    """
    Data Module for Devanagari Character Groups
    Args:
        train_dir (str): Path to Train Images LMDB directory
        val_dir (str): Path to Validation Images LMDB directory
        test_dir (Optional[str]): Path to Test LMDB Directory
        transforms (Torchvision.transforms.Compose): Transforms to be applied on the Images
        batch_size (int): Batch Size
        num_workers (int): Number of processes for the DataSet and DataLoader
        drop_last (bool): Whether to drop the last batch if its not divisible by the batch size
    """
    def __init__(self, train_dir:Optional[str], val_dir:Optional[str], test_dir:Optional[str],
                svar:list, vyanjan:list, matras:list, ank:list, chinh:list,
                halanth:str, transforms: transforms.Compose, 
                test_transforms: transforms.Compose, batch_size:int =32, num_workers:int= 4,
                drop_last:bool= False, remove_unseen:bool= False
                ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.transforms = transforms
        self.test_transforms = test_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.svar = svar
        self.vyanjan= vyanjan
        self.matras= matras
        self.ank= ank
        self.chinh= chinh
        self.halanth= halanth
        self.remove_unseen = remove_unseen

        #non-parametric attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage):
        if self.train_dir:
            self.train_dataset = HindiLMDBDataset(
                                    data_dir= self.train_dir,
                                    transforms= self.transforms,
                                    svar= self.svar,
                                    vyanjan= self.vyanjan,
                                    matras= self.matras,
                                    ank= self.ank,
                                    chinh= self.chinh,
                                    halanth= self.halanth,
                                    remove_unseen = self.remove_unseen,
                                )
            
        if self.val_dir:
            self.val_dataset = HindiLMDBDataset(
                                    data_dir= self.val_dir,
                                    transforms= self.transforms,
                                    svar= self.svar,
                                    vyanjan= self.vyanjan,
                                    matras= self.matras,
                                    ank= self.ank,
                                    chinh= self.chinh,
                                    halanth= self.halanth,
                                    remove_unseen = self.remove_unseen,
                                )
            
        if self.test_dir:
            self.test_dataset = HindiLMDBDataset(
                                data_dir= self.test_dir,
                                transforms= self.test_transforms,
                                svar= self.svar,
                                vyanjan= self.vyanjan,
                                matras= self.matras,
                                ank= self.ank,
                                chinh= self.chinh,
                                halanth= self.halanth,
                                remove_unseen = self.remove_unseen,
                            )

    def train_dataloader(self):
        if self.train_dataset:
            return DataLoader(
                        dataset = self.train_dataset,
                        batch_size=self.batch_size, 
                        num_workers = self.num_workers,
                        drop_last = self.drop_last,
                        shuffle=True,
                    )
        else:
            raise Exception("train_dir not provided")

    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(
                        dataset= self.val_dataset, 
                        batch_size= self.batch_size, 
                        num_workers = self.num_workers,
                        drop_last = self.drop_last,   
                    )
        else:
            raise Exception("val_dir not provided")

    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(
                        self.test_dataset, 
                        batch_size= self.batch_size, 
                        num_workers = self.num_workers,
                        drop_last = self.drop_last,
                    )
        else:
            raise Exception("test_dir not provided")