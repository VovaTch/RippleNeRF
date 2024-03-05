from __future__ import annotations
from typing import Any
from typing_extensions import Self

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT

import models
from utils.geometry import render_rays
from utils.ripplenet import RippleLinear


class NerfLinearModel(nn.Module):
    """
    NeRF model from the minimal implementation in https://github.com/MaximeVandegar/"""

    def __init__(
        self,
        embedding_dim_pos: int = 10,
        embedding_dim_direction: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        """
        Initializes the NerfLinearModel.

        Args:
            embedding_dim_pos (int): The embedding dimension for position.
            embedding_dim_direction (int): The embedding dimension for direction.
            hidden_dim (int): The hidden dimension for the linear layers.
        """
        super(NerfLinearModel, self).__init__()

        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # density estimation
        self.block2 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1),
        )
        # color estimation
        self.block3 = nn.Sequential(
            nn.Linear(embedding_dim_direction * 6 + hidden_dim + 3, hidden_dim // 2),
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_direction = embedding_dim_direction
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x: torch.Tensor, embedding_dim: int) -> torch.Tensor:
        """
        Apply positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.
            embedding_dim (int): The dimension of the positional encoding.

        Returns:
            torch.Tensor: The tensor with positional encoding applied.
        """
        out = [x]
        for j in range(embedding_dim):
            out.append(torch.sin(2**j * x))
            out.append(torch.cos(2**j * x))
        return torch.cat(out, dim=1)

    def forward(
        self, origin: torch.Tensor, direction: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            origin (torch.Tensor): Tensor representing the origin coordinates.
            direction (torch.Tensor): Tensor representing the direction vectors.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the color tensor and the sigma tensor.
        """
        emb_x = self.positional_encoding(
            origin, self.embedding_dim_pos
        )  # emb_x dimensions: BS x EDim x 6
        emb_d = self.positional_encoding(
            direction, self.embedding_dim_direction
        )  # emb_d dimensions BS x EDim x 6
        hidden = self.block1(emb_x)  # h: BS x h
        tmp = self.block2(
            torch.cat((hidden, emb_x), dim=1)
        )  # tmp: [batch_size, hidden_dim + 1]
        hidden, sigma = tmp[:, :-1], self.relu(
            tmp[:, -1]
        )  # h: [batch_size, hidden_dim], sigma: [batch_size]
        hidden = self.block3(
            torch.cat((hidden, emb_d), dim=1)
        )  # h: [batch_size, hidden_dim // 2]
        color = self.block4(hidden)  # c: [batch_size, 3]
        return {"color": color, "sigma": sigma}


class RippleNerf(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        cast_dim: int,
        num_ripple_layers: int,
        coordinate_multiplier: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cast_dim = cast_dim
        self.num_ripple_layers = num_ripple_layers
        self.coordinate_multiplier = coordinate_multiplier

        if num_ripple_layers < 1:
            raise ValueError(
                f"The number of inner ripple layers must be at least 1, got {num_ripple_layers}"
            )

        self.in_linear = nn.Linear(5, cast_dim)

        if num_ripple_layers == 1:
            inner_ripple_layers = [RippleLinear(cast_dim, 4)]
        else:
            inner_ripple_layers = [RippleLinear(cast_dim, hidden_dim), nn.GELU()]
            for layer_idx in range(num_ripple_layers - 1):
                out_dim = hidden_dim if layer_idx < num_ripple_layers - 2 else 4
                inner_ripple_layers += [
                    nn.GELU(),
                    RippleLinear(hidden_dim, out_dim),
                ]
        self.bypass = RippleLinear(5, 4)
        self.res_block = nn.Sequential(*inner_ripple_layers)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, origin: torch.Tensor, direction: torch.Tensor
    ) -> dict[str, torch.Tensor]:

        origin *= self.coordinate_multiplier
        # direction *= self.coordinate_multiplier
        yaw = (
            torch.atan2(direction[:, 0], direction[:, 1]).unsqueeze(-1)
            * self.coordinate_multiplier
        )
        pitch = (
            torch.atan2(
                direction[:, 2], torch.sqrt(direction[:, 0] ** 2 + direction[:, 1] ** 2)
            ).unsqueeze(-1)
            * self.coordinate_multiplier
        )

        x = torch.cat((origin, yaw, pitch), dim=1)
        bypass = self.bypass(x)
        x = self.in_linear(x)
        x = F.gelu(x)
        x = self.res_block(x) + bypass

        return {"color": self.sigmoid(x[:, :3]), "sigma": self.sigmoid(x[:, 3])}


class NerfModule(L.LightningModule):
    """
    Module containing the training logic for the underlying NeRF model. The model
    is separated from the module via dependency injection.
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler_cfg: DictConfig | None = None,
        optimizer_cfg: DictConfig | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        save_folder: str = "saved/",
        scheduler_interval: str = "epoch",
        loss_monitor: str = "validation total loss",
        scheduler_frequency: int = 1,
        near_plane_distance: float = 0.0,
        far_plane_distance: float = 1.0,
        num_bins: int = 192,
    ) -> None:
        """
        Initialize the model.

        Args:
            model (nn.Module): The neural network model.
            loss (nn.Module | None, optional): The loss function. Defaults to None.
            scheduler_cfg (DictConfig | None, optional): The configuration for the scheduler. Defaults to None.
            optimizer_cfg (DictConfig | None, optional): The configuration for the optimizer. Defaults to None.
            learning_rate (float, optional): The learning rate. Defaults to 1e-3.
            weight_decay (float, optional): The weight decay. Defaults to 1e-6.
            save_folder (str, optional): The folder to save the model. Defaults to "saved/".
            scheduler_interval (str, optional): The interval at which the scheduler is updated. Defaults to "epoch".
            loss_monitor (str, optional): The loss monitor. Defaults to "validation total loss".
            scheduler_frequency (int, optional): The frequency at which the scheduler is updated. Defaults to 1.
            near_plane_distance (float, optional): The near plane distance. Defaults to 0.0.
            far_plane_distance (float, optional): The far plane distance. Defaults to 1.0.
            num_bins (int, optional): The number of bins. Defaults to 192.
        """
        super().__init__()
        self.model = model
        self.scheduler = (
            getattr(torch.optim.lr_scheduler, scheduler_cfg.type)(
                **scheduler_cfg.params
            )
            if scheduler_cfg
            else None
        )
        self.optimizer = (
            getattr(torch.optim, optimizer_cfg.type)(
                lr=learning_rate, weight_decay=weight_decay, **optimizer_cfg.params
            )
            if optimizer_cfg
            else torch.optim.AdamW(
                self.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        )
        self.save_folder = save_folder
        self.scheduler_interval = scheduler_interval
        self.scheduler_frequency = scheduler_frequency
        self.loss_monitor = loss_monitor
        self.near_plane_distance = near_plane_distance
        self.far_plane_distance = far_plane_distance
        self.num_bins = num_bins

    def configure_optimizers(
        self,
    ) -> OptimizerLRScheduler:
        """
        Configures the optimizers and learning rate schedulers for the model.

        Returns:
            Tuple[List[Optimizer], List[Dict[str, Any]]]: A tuple containing the list of optimizers and a
            list of scheduler settings.
        """
        if self.scheduler is None:
            return [self.optimizer]
        else:
            scheduler_settings = {
                "scheduler": self.scheduler,
                "interval": self.scheduler_interval,
                "frequency": self.scheduler_frequency,
                "monitor": self.loss_monitor,
            }
        return [self.optimizer], [scheduler_settings]

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x (dict[str, torch.Tensor]): Input dictionary containing 'origin' and 'direction' tensors.

        Returns:
            dict[str, torch.Tensor]: Output dictionary containing the model's predictions.
        """
        return self.model(x["origin"], x["direction"])

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        """
        Perform a single training step.

        Args:
            batch (dict[str, Any]): The input batch for the training step.
            batch_idx (int): The index of the current batch.

        Returns:
            STEP_OUTPUT: The output of the training step.
        """

        return self._step(batch, "training")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> STEP_OUTPUT:
        """
        Perform a validation step.

        Args:
            batch (dict[str, Any]): The input batch for validation.
            batch_idx (int): The index of the current batch.

        Returns:
            STEP_OUTPUT: The output of the validation step.
        """

        return self._step(batch, "validation")

    def _step(self, batch: dict[str, torch.Tensor], phase: str) -> STEP_OUTPUT:
        """
        Perform a single step of the training/validation process.

        Args:
            batch (dict[str, torch.Tensor]): A dictionary containing the input batch data.
            phase (str): The current phase of the process (e.g., "training", "validation").

        Returns:
            STEP_OUTPUT: The loss value for the step.
        """
        regenerated_pixel_values = render_rays(
            self,
            batch["origin"],
            batch["direction"],
            self.near_plane_distance,
            self.far_plane_distance,
            self.num_bins,
        )
        loss = F.mse_loss(
            batch["pixel_value"], regenerated_pixel_values, reduction="sum"
        )
        self.log(f"{phase}_total_loss", loss.item(), prog_bar=True)
        return loss

    @classmethod
    def from_cfg(cls, cfg: DictConfig, checkpoint_path: str | None = None) -> Self:
        """
        Create an instance of the class from a configuration object.

        Args:
            cfg (DictConfig): The configuration object.

        Returns:
            NerfModule: An instance of the class.
        """
        model = getattr(models, cfg.model.type)(**cfg.model.params)
        scheduler_cfg = None if "scheduler" not in cfg else cfg.scheduler
        optimizer_cfg = None if "optimizer" not in cfg else cfg.optimizer

        model_params = {
            "model": model,
            "scheduler_cfg": scheduler_cfg,
            "optimizer_cfg": optimizer_cfg,
            "learning_rate": cfg.learning.learning_rate,
            "weight_decay": cfg.learning.weight_decay,
            "save_folder": cfg.learning.save_folder,
            "scheduler_interval": cfg.learning.scheduler_interval,
            "loss_monitor": cfg.learning.loss_monitor,
            "scheduler_frequency": cfg.learning.scheduler_frequency,
            "near_plane_distance": cfg.rendering_train.near_plane_distance,
            "far_plane_distance": cfg.rendering_train.far_plane_distance,
            "num_bins": cfg.rendering_train.num_bins,
        }

        if checkpoint_path is None:
            return cls(**model_params)
        else:
            return cls.load_from_checkpoint(checkpoint_path, **model_params)
