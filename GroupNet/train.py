import torch
import hydra
import signal
import math
from data.augment import rand_augment_transform
from torchvision import transforms
from utils.transforms import RescaleTransform, PadTransform
from lightning.pytorch.callbacks import StochasticWeightAveraging, LearningRateMonitor
from omegaconf import DictConfig
from hydra.utils import instantiate
from lightning.pytorch.plugins.environments import SLURMEnvironment

# torch.set_printoptions(profile="full")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')
print(f"Device: {device}")
# Copied from OneCycleLR
def _annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = math.cos(math.pi * pct) + 1
    return end + (start - end) / 2.0 * cos_out

def get_swa_lr_factor(warmup_pct, swa_epoch_start, div_factor=25, final_div_factor=1e4) -> float:
    """Get the SWA LR factor for the given `swa_epoch_start`. Assumes OneCycleLR Scheduler."""
    total_steps = 1000  # Can be anything. We use 1000 for convenience.
    start_step = int(total_steps * warmup_pct) - 1
    end_step = total_steps - 1
    step_num = int(total_steps * swa_epoch_start) - 1
    pct = (step_num - start_step) / (end_step - start_step)
    return _annealing_cos(1, 1 / (div_factor * final_div_factor), pct)

@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    """
    Function to train character embeddings
    """
    composed = transforms.Compose([
        rand_augment_transform(),
        transforms.ToTensor(),
        transforms.RandomRotation(
            degrees= cfg.transforms.rotation, 
            expand = True, 
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        RescaleTransform(cfg.transforms.img_size),
        PadTransform(cfg.transforms.img_size),
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # ),
        transforms.Normalize(0.5, 0.5)
        ])
    test_composed = transforms.Compose([
        transforms.ToTensor(),
        RescaleTransform(cfg.transforms.img_size),
        PadTransform(cfg.transforms.img_size),
        transforms.Normalize(0.5, 0.5),
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # ),
        ])

    datamodule = instantiate(cfg.datamodule, transforms = composed, test_transforms = test_composed)

    csv_logger = instantiate(cfg.csv_logger)
    tensorboard_logger = instantiate(cfg.tensorboard_logger)

    checkpoint_callback = instantiate(cfg.model_checkpoint)
    swa_epoch_start = 0.75
    swa_lr = cfg.model.learning_rate * get_swa_lr_factor(cfg.model.warmup_pct, swa_epoch_start)
    swa = StochasticWeightAveraging(swa_lr, swa_epoch_start)
    # swa = StochasticWeightAveraging(swa_lrs=1e-2)

    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum = True)
    trainer = instantiate(
                    cfg.training, 
                    callbacks = [checkpoint_callback, swa, lr_monitor],
                    logger = [csv_logger, tensorboard_logger],
                    plugins=SLURMEnvironment(auto_requeue= True, requeue_signal=signal.SIGUSR1),
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