import torch
import torchvision
import pandas as pd
import cv2 as cv
import numpy as np
import lightning.pytorch as pl
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from .dataset import DevanagariDataset

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
    """
    def __init__(self, train_img_dir: str, train_gt: str, val_img_dir: str, val_gt: str, 
                charset:list or tuple, diacritics:list or tuple, halfer:str, half_charset: list or tuple,
                test_img_dir:str = None, test_gt:str = None,
                batch_size: int = 64, normalize = True, num_workers: int = 0,
                transforms: transforms.Compose = None, delimiter:str = ' ', drop_last = False):
        super().__init__()
        self.train_img_dir = train_img_dir
        self.train_gt = train_gt
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

    def setup(self, stage):
        self.train_dataset = DevanagariDataset(
                                img_dir = self.train_img_dir, 
                                gt_file = self.train_gt, 
                                charset= self.charset,
                                diacritics= self.diacritics,
                                halfer= self.halfer,
                                half_charset = self.half_charset,
                                transforms = self.transforms,
                                seperator = self.delimiter,
                                normalize= self.normalize
                            )
        self.val_dataset = DevanagariDataset(
                                img_dir = self.val_img_dir, 
                                gt_file = self.val_gt,
                                charset= self.charset,
                                diacritics= self.diacritics,
                                halfer= self.halfer,
                                half_charset = self.half_charset,
                                transforms = self.transforms,
                                seperator = self.delimiter,
                                normalize= self.normalize
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
                                seperator = self.delimiter,
                                normalize= self.normalize
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


class VyanjanDataModule(pl.LightningDataModule):
    """
    A Lightning DataModule for Vyanjan Glyph dataset
    Args:
        train_img_dir (str): Path to Train Images w.r.t ground truth (gt) labels file
        train_gt (str) : Path to Labels for the train images
        val_img_dir (str): Path to Validation Images w.r.t ground truth (gt) labels file
        val_gt (str): Path to Labels for the Validation images
        transforms (Torchvision.transforms.Compose): Transforms to be applied on the Images
        batch_size (int): Batch Size
        num_workers (int): Number of processes for the DataSet and DataLoader
        drop_last (bool): Whether to drop the last batch if its not divisible by the batch size
    """
    def __init__(self, train_img_dir: str, train_gt: str, val_img_dir: str, val_gt: str, 
                batch_size: int = 64, num_workers: int = 0,
                transforms: transforms.Compose = None, seperator:str = ' ', drop_last = False):
        super().__init__()
        self.train_img_dir = train_img_dir
        self.val_img_dir = val_img_dir
        self.train_gt = train_gt
        self.val_gt = val_gt
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.seperator = seperator

    def setup(self, stage):
        self.train_dataset = VyanjanDataset(
                                img_dir = self.train_img_dir, 
                                gt_file = self.train_gt, 
                                transforms = self.transforms,
                                seperator = self.seperator
                            )
        self.charset = self.train_dataset.charset
        self.val_dataset = VyanjanDataset(
                                img_dir = self.val_img_dir, 
                                gt_file = self.val_gt, 
                                transforms = self.transforms,
                                seperator = self.seperator
                            )

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
  
