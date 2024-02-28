from dataclasses import dataclass

import torch
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


@dataclass
class Volume: ...
