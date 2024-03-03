import hydra
from omegaconf import DictConfig
import torch
import matplotlib.pyplot as plt

from datasets import NerfOnlineDataModule
from models import NerfModule
from utils.predictor import build_image


CHECKPOINT_PATH = "weights/nerf.ckpt"


# Temporary; it will be enhanced later
@hydra.main(version_base=None, config_path="config", config_name="nerf")
def main(cfg: DictConfig) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_module = NerfOnlineDataModule.from_cfg(cfg)
    nerf_module = (
        NerfModule.from_cfg(cfg, checkpoint_path=CHECKPOINT_PATH).to(device).eval()
    )

    data_module.setup("test")

    img = build_image(
        nerf_module,
        cfg.rendering.near_plane_distance,
        cfg.rendering.far_plane_distance,
        data_module.test_dataset,  # type: ignore
        10,
        img_index=12,
        nb_bins=cfg.rendering.num_bins,
        height=400,
        width=400,
    )

    plt.figure()
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()
