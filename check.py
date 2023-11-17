import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    model = hydra.utils.instantiate(cfg.Img2Vec)
    print(model)


main()