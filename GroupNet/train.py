import torch
import hydra
from torchvision import transforms
from utils.transforms import RescaleTransform, PadTransform
from lightning.pytorch.callbacks import StochasticWeightAveraging, LearningRateMonitor
from omegaconf import DictConfig
from hydra.utils import instantiate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')

print(f"Device: {device}")

@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    """
    Function to train character embeddings
    """
    composed = transforms.Compose([
        transforms.RandomRotation(
            degrees= cfg.transforms.rotation, 
            expand = True, 
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        RescaleTransform(cfg.transforms.img_size),
        PadTransform(cfg.transforms.img_size),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ])
    test_composed = transforms.Compose([
        RescaleTransform(cfg.transforms.img_size),
        PadTransform(cfg.transforms.img_size),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ])

    datamodule = instantiate(cfg.datamodule, transforms = composed, test_transforms = test_composed)

    csv_logger = instantiate(cfg.csv_logger)
    tensorboard_logger = instantiate(cfg.tensorboard_logger)

    checkpoint_callback = instantiate(cfg.model_checkpoint)

    swa = StochasticWeightAveraging(swa_lrs=1e-2)

    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum = True)
    trainer = instantiate(
                    cfg.training, 
                    callbacks = [checkpoint_callback, swa, lr_monitor],
                    logger = [csv_logger, tensorboard_logger]
                )
    
    if cfg.restart_training and cfg.model_load is not None:
        # just load model weights from the ckpt
        model = instantiate(cfg.model_load)
        # restart the training
        trainer.fit(model, datamodule)
    else: 
        model = instantiate(cfg.model)
        # continue training by loading training state from the checkpoint
        trainer.fit(model, datamodule, ckpt_path = cfg.ckpt_path)
    

    if cfg.datamodule.test_dir is not None:
        trainer.test(datamodule = datamodule)
    
if __name__ == "__main__":
    main()