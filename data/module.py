import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import DevanagariDataset, LMDBDevanagariDataset
from typing import Optional

class DevanagariDataModule(pl.LightningDataModule):
    """
    Data Module for Devanagari Character Groups
    Args:
        train_img_dir (str): Path to Train Images w.r.t ground truth (gt) labels file
        train_gt (str) : Path to Labels for the train images
        val_img_dir (str): Path to Validation Images w.r.t ground truth (gt) labels file
        val_gt (str): Path to Labels for the Validation images
        transforms (Torchvision.transforms.Compose): Transforms to be applied on the Images
        batch_size (int): Batch Size
        num_workers (int): Number of processes for the DataSet and DataLoader
        drop_last (bool): Whether to drop the last batch if its not divisible by the batch size
        false_sample_dir (str): path to false sample image director w.r.t the ground truth file
        false_sample_gt (str): path to false sample ground truth file
        false_train
    """
    def __init__(self, train_img_dir: str, train_gt: str, val_img_dir: str, val_gt: str, 
                charset:list or tuple, diacritics:list or tuple, halfer:str, half_charset: list or tuple,
                test_img_dir:Optional[str] = None, test_gt:Optional[str] = None,
                batch_size: int = 64, normalize = True, num_workers: int = 0,
                transforms: Optional[transforms.Compose or transforms] = None, delimiter:str = ' ', drop_last = False,
                false_sample_dir:Optional[str] = None, false_sample_gt:Optional[str] = None,
                false_train_wt:int = 1, false_val_wt:int = 1):
        super().__init__()
        self.train_img_dir = train_img_dir
        self.train_gt = train_gt
        self.false_sample_dir = false_sample_dir
        self.false_sample_gt = false_sample_gt
        self.val_img_dir = val_img_dir
        self.val_gt = val_gt
        self.test_img_dir = test_img_dir
        self.test_gt = test_gt
        self.charset = charset
        self.diacritics = diacritics
        self.halfer = halfer
        self.half_charset = half_charset
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.delimiter = delimiter
        self.normalize = normalize
        self.false_train_wt = false_train_wt
        self.false_val_wt = false_val_wt

    def setup(self, stage):
        self.train_dataset = DevanagariDataset(
                                img_dir = self.train_img_dir, 
                                gt_file = self.train_gt, 
                                charset= self.charset,
                                diacritics= self.diacritics,
                                halfer= self.halfer,
                                half_charset = self.half_charset,
                                transforms = self.transforms,
                                separator = self.delimiter,
                                false_sample_dir = self.false_sample_dir,
                                false_sample_gt = self.false_sample_gt,
                                false_weight = self.false_train_wt,
                            )
        self.val_dataset = DevanagariDataset(
                                img_dir = self.val_img_dir, 
                                gt_file = self.val_gt,
                                charset= self.charset,
                                diacritics= self.diacritics,
                                halfer= self.halfer,
                                half_charset = self.half_charset,
                                transforms = self.transforms,
                                separator = self.delimiter,
                                false_sample_dir = self.false_sample_dir,
                                false_sample_gt = self.false_sample_gt,
                                false_weight = self.false_val_wt,
                            )
        if self.test_gt is not None:
            self.test_dataset = DevanagariDataset(
                                img_dir = self.test_img_dir, 
                                gt_file = self.test_gt,
                                charset= self.charset,
                                diacritics= self.diacritics,
                                halfer= self.halfer,
                                half_charset = self.half_charset,
                                transforms = self.transforms,
                                separator = self.delimiter,
                            )
        else:
            self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
                    dataset = self.train_dataset,
                    batch_size=self.batch_size, 
                    num_workers = self.num_workers,
                    drop_last = self.drop_last
                )

    def val_dataloader(self):
        return DataLoader(
                    self.val_dataset, 
                    batch_size= self.batch_size, 
                    num_workers = self.num_workers,
                    drop_last = self.drop_last    
                )
    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(
                        self.test_dataset, 
                        batch_size= self.batch_size, 
                        num_workers = self.num_workers,
                        drop_last = self.drop_last    
                    )
        else:
            raise Exception("Test data not provided")


class LMDBDevanagariDataModule(pl.LightningDataModule):
    """
    Data Module for Devanagari Character Groups
    Args:
        train_img_dir (str): Path to Train Images LMDB directory
        val_img_dir (str): Path to Validation Images LMDB directory
        transforms (Torchvision.transforms.Compose): Transforms to be applied on the Images
        batch_size (int): Batch Size
        num_workers (int): Number of processes for the DataSet and DataLoader
        drop_last (bool): Whether to drop the last batch if its not divisible by the batch size
    """
    def __init__(self, train_dir: str, val_dir: str, charset:list or tuple, diacritics:list or tuple,
                halfer:str, half_charset: list or tuple, test_dir:Optional[str] = None,
                batch_size: int = 64, normalize = True, num_workers: int = 0,
                transforms: Optional[transforms.Compose] = None, delimiter:str = ' ', drop_last = False,
                false_sample_dir:Optional[str] = None, false_train_wt:int = 1, false_val_wt:int = 1):
        super().__init__()
        self.train_dir = train_dir
        self.false_sample_dir = false_sample_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.charset = charset
        self.diacritics = diacritics
        self.halfer = halfer
        self.half_charset = half_charset
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.delimiter = delimiter
        self.normalize = normalize
        self.false_train_wt = false_train_wt
        self.false_val_wt = false_val_wt

    def setup(self, stage):
        self.train_dataset = LMDBDevanagariDataset(
                                img_dir = self.train_dir, 
                                charset= self.charset,
                                diacritics= self.diacritics,
                                halfer= self.halfer,
                                half_charset = self.half_charset,
                                transforms = self.transforms,
                                false_sample_dir = self.false_sample_dir,
                                false_weight = self.false_train_wt,
                            )
        self.val_dataset = LMDBDevanagariDataset(
                                img_dir = self.val_dir,
                                charset= self.charset,
                                diacritics= self.diacritics,
                                halfer= self.halfer,
                                half_charset = self.half_charset,
                                transforms = self.transforms,
                                false_sample_dir = self.false_sample_dir,
                                false_weight = self.false_val_wt,
                            )
        if self.test_dir is not None:
            self.test_dataset = LMDBDevanagariDataset(
                                img_dir = self.test_dir, 
                                charset= self.charset,
                                diacritics= self.diacritics,
                                halfer= self.halfer,
                                half_charset = self.half_charset,
                                transforms = self.transforms,
                            )
        else:
            self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
                    dataset = self.train_dataset,
                    batch_size=self.batch_size, 
                    num_workers = self.num_workers,
                    drop_last = self.drop_last,
                    shuffle=True,
                )

    def val_dataloader(self):
        return DataLoader(
                    self.val_dataset, 
                    batch_size= self.batch_size, 
                    num_workers = self.num_workers,
                    drop_last = self.drop_last,
                    shuffle=True,    
                )
    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(
                        self.test_dataset, 
                        batch_size= self.batch_size, 
                        num_workers = self.num_workers,
                        drop_last = self.drop_last,
                        shuffle=True,    
                    )
        else:
            raise Exception("Test data not provided")