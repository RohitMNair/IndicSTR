import lightning.pytorch as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import HindiLMDBDataset
from typing import Optional, Union

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
    def __init__(self, train_dir:str, val_dir:str, test_dir:Optional[str],
                half_character_classes:list, full_character_classes:list, 
                diacritic_classes:list, halfer:str,
                transforms: transforms.Compose, batch_size:int =32, 
                num_workers:int= 4, drop_last:bool= False, 
                
                ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.half_character_classes = half_character_classes
        self.full_character_classes = full_character_classes
        self.diacritic_classes = diacritic_classes
        self.halfer = halfer

    def setup(self, stage):
        self.train_dataset = HindiLMDBDataset(
                                data_dir= self.train_dir,
                                transforms= self.transforms,
                                half_character_classes= self.half_character_classes,
                                full_character_classes= self.full_character_classes,
                                diacritic_classes= self.diacritic_classes,
                                halfer= self.halfer,
                            )
        self.val_dataset = HindiLMDBDataset(
                                data_dir= self.val_dir,
                                transforms= self.transforms,
                                half_character_classes= self.half_character_classes,
                                full_character_classes= self.full_character_classes,
                                diacritic_classes= self.diacritic_classes,
                                halfer= self.halfer,
                            )
        if self.test_dir is not None:
            self.test_dataset = HindiLMDBDataset(
                                data_dir= self.test_dir,
                                transforms= self.transforms,
                                half_character_classes= self.half_character_classes,
                                full_character_classes= self.full_character_classes,
                                diacritic_classes= self.diacritic_classes,
                                halfer= self.halfer,
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
                    dataset= self.val_dataset, 
                    batch_size= self.batch_size, 
                    num_workers = self.num_workers,
                    drop_last = self.drop_last,   
                )

    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(
                        self.test_dataset, 
                        batch_size= self.batch_size, 
                        num_workers = self.num_workers,
                        drop_last = self.drop_last,
                    )
        else:
            raise Exception("Test data not provided")