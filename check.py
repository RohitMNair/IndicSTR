import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    print(type(cfg.data.charset))
    print(cfg.data.charset)

main()