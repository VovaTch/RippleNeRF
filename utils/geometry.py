from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np


@dataclass
class Ray:
    """
    Describes a ray in 3D space.

    Fields:
    - origin (np.ndarray): 3D point
    - direction (np.ndarray): 3D vector
    """

    origin: np.ndarray
    direction: np.ndarray

    @property
    def ray_tensor(self) -> torch.Tensor:
        """
        Returns a tensor representation of the ray.

        Returns:
            torch.Tensor: A tensor representing the ray, consisting of the origin, yaw angle, and pitch angle.
        """
        origin_tensor = torch.tensor(self.origin)
        direction_tensor = torch.tensor(self.direction)
        yaw_angle = torch.atan2(torch.tensor(direction_tensor[1], direction_tensor[0]))
        pitch_angle = torch.atan2(
            torch.tensor(direction_tensor[2]),
            torch.norm(direction_tensor[1], direction_tensor[0]),
        )
        return torch.cat([origin_tensor, yaw_angle, pitch_angle])


def compute_accumulated_transmittance(alphas: torch.Tensor) -> torch.Tensor:
    """
    Compute the accumulated transmittance for each pixel in the input tensor.

    Args:
        alphas (torch.Tensor): Input tensor of shape (batch_size, num_pixels) representing the alpha values.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, num_pixels) representing the accumulated transmittance.
    """
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat(
        (
            torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
            accumulated_transmittance[:, :-1],
        ),
        dim=-1,
    )


def render_rays(
    nerf_model: nn.Module,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near_plane_distance: float = 0.0,
    far_plane_distance: float = 0.5,
    num_bins: int = 192,
) -> torch.Tensor:
    """
    Renders rays using the NeRF model.

    Args:
        nerf_model (nn.Module): The NeRF model used for rendering.
        ray_origins (torch.Tensor): Tensor of shape (batch_size, 3) containing the origins of the rays.
        ray_directions (torch.Tensor): Tensor of shape (batch_size, 3) containing the directions of the rays.
        near_plane_distance (float, optional): Near plane distance. Defaults to 0.0.
        far_plane_distance (float, optional): Far plane distance. Defaults to 0.5.
        num_bins (int, optional): Number of bins for sampling along each ray. Defaults to 192.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, 3) containing the rendered colors for each ray.
    """
    device = ray_origins.device

    t = torch.linspace(
        near_plane_distance, far_plane_distance, num_bins, device=device
    ).expand(ray_origins.shape[0], num_bins)
    # Perturb sampling along each ray.
    mid = (t[:, :-1] + t[:, 1:]) / 2.0
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    delta = torch.cat(
        (
            t[:, 1:] - t[:, :-1],
            torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1),
        ),
        -1,
    )

    # Compute the 3D points along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(
        1
    )  # [batch_size, nb_bins, 3]
    # Expand the ray_directions tensor to match the shape of x
    ray_directions = ray_directions.expand(
        num_bins, ray_directions.shape[0], 3
    ).transpose(0, 1)

    outputs = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors, sigma = outputs["color"], outputs["sigma"]
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)  # [batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(
        2
    ) * alpha.unsqueeze(2)
    # Compute the pixel values as a weighted sum of colors along each ray
    c = (weights * colors).sum(dim=1)
    weight_sum = weights.sum(-1).sum(-1)  # Regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1)
