import hydra
from omegaconf import DictConfig
from models import NerfModule

from utils.trainer import initialize_trainer
from datasets import NerfOnlineDataModule


@hydra.main(version_base=None, config_path="config", config_name="nerf")
def main(cfg: DictConfig) -> None:
    data_module = NerfOnlineDataModule.from_cfg(cfg)
    nerf_module = NerfModule.from_cfg(cfg)

    trainer = initialize_trainer(
        cfg.model_name,
        save_folder=cfg.learning.save_folder,
        num_devices=cfg.learning.num_devices,
        beta_ema=cfg.learning.beta_ema,
        gradient_clip=cfg.learning.gradient_clip,
        max_epochs=cfg.learning.max_epochs,
        loss_monitor=cfg.learning.loss_monitor,
        use_wandb=cfg.learning.use_wandb,
        wandb_project_name=cfg.learning.wandb_project_name,
        use_amp=cfg.learning.use_amp,
    )

    trainer.fit(nerf_module, data_module)


if __name__ == "__main__":
    main()
