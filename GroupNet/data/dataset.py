import torch
import lightning.pytorch as pl

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms

class HindiDataset(Dataset):
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
        

