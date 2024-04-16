import torch
import hydra
from torchvision import transforms
from utils.transforms import RescaleTransform, PadTransform

from omegaconf import DictConfig
from hydra.utils import instantiate
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')

print(f"Device: {device}")

@hydra.main(version_base=None, config_path="configs", config_name="test")
def main(cfg: DictConfig):
    """
    Function to train character embeddings
    """
    model = instantiate(cfg.model)

    test_composed = transforms.Compose([
        transforms.ToTensor(),
        RescaleTransform(model.hparams.image_size),
        PadTransform(model.hparams.image_size),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        # transforms.Normalize(0.5, 0.5),
        ])

    datamodule = instantiate(
                        cfg.datamodule, 
                        train_dir= None,
                        val_dir= None,
                        transforms = None, 
                        test_transforms = test_composed,
                        )
    
    csv_logger = instantiate(cfg.csv_logger)
    tensorboard_logger = instantiate(cfg.tensorboard_logger)

    trainer = instantiate(
                    cfg.training, 
                    devices= 1,
                    strategy= 'auto',
                    logger = [csv_logger, tensorboard_logger]
                )

    trainer.test(model= model, datamodule = datamodule)

if __name__ == "__main__":
    main()