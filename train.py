import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="ripplenerf")
def main(cfg: DictConfig) -> None:
    pass


if __name__ == "__main__":
    main()
