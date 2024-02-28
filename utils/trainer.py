import os
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    ModelSummary,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, Logger

from .ema import EMA


# Function body...
def initialize_trainer(
    model_name: str,
    save_folder: str = "saved/",
    num_devices: int = 0,
    beta_ema: float = 0.9999,
    gradient_clip: float = 0.1,
    max_epochs: int = 100,
    loss_monitor: str = "validation total loss",
    use_wandb: bool = False,
    wandb_project_name: str = "",
    use_amp: bool = False,
) -> L.Trainer:
    """
    Initializes a trainer for training a model.

    Args:
    *   model_name (str): The name of the model.
    *   save_folder (str, optional): The folder to save the model checkpoints. Defaults to "saved/".
    *   num_devices (int, optional): The number of devices to use for training. Defaults to 0.
    *   beta_ema (float, optional): The beta value for exponential moving average. Defaults to 0.9999.
    *   gradient_clip (float, optional): The value to clip gradients. Defaults to 0.1.
    *   max_epochs (int, optional): The maximum number of epochs for training. Defaults to 100.
    *   loss_monitor (str, optional): The metric to monitor for saving the best model checkpoint.
        Defaults to "validation total loss".
    *   use_wandb (bool, optional): Whether to use wandb for logging. Defaults to False.
    *   wandb_project_name (str, optional): The name of the wandb project. Defaults to the model_name parameter.
    *   use_amp (bool, optional): Whether to use automatic mixed precision. Defaults to False.

    Returns:
    *   L.Trainer: The initialized trainer object.
    """

    # Set device
    accelerator = "cpu" if num_devices == 0 else "gpu"

    # Configure trainer
    ema = EMA(beta_ema)
    learning_rate_monitor = LearningRateMonitor(logging_interval="step")
    tensorboard_logger = TensorBoardLogger(save_dir=save_folder, name=model_name)
    loggers: list[Logger] = [tensorboard_logger]

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_folder, model_name),
        filename=f"{model_name}_best.ckpt",
        save_weights_only=True,
        save_top_k=1,
        monitor=loss_monitor,
    )
    model_last_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(save_folder, model_name),
        filename=f"{model_name}_last.ckpt",
        save_last=True,
        save_weights_only=True,
        save_top_k=0,
    )

    # Initialize wandb if needed
    if use_wandb:
        if wandb_project_name == "":
            wandb_project_name = model_name
        wandb_logger = WandbLogger(project=wandb_project_name, log_model="all")
        loggers.append(wandb_logger)

    # AMP
    precision = 16 if use_amp else 32

    model_summary = ModelSummary(max_depth=3)
    trainer = L.Trainer(
        gradient_clip_val=gradient_clip,
        logger=loggers,
        callbacks=[
            model_checkpoint_callback,
            model_last_checkpoint_callback,
            model_summary,
            learning_rate_monitor,
            ema,
        ],
        devices="auto",
        max_epochs=max_epochs,
        log_every_n_steps=1,
        precision=precision,
        accelerator=accelerator,
    )

    return trainer
