from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import lightning as L

from utils.geometry import render_rays


@torch.no_grad()
def build_image(
    model: torch.nn.Module | L.LightningModule,
    near_place_distance: float,
    far_plane_distance: float,
    dataset: Dataset,
    chunk_size: int = 10,
    img_index: int = 0,
    nb_bins: int = 192,
    height: int = 400,
    width: int = 400,
) -> torch.Tensor:

    ray_origins = dataset[
        img_index * height * width : (img_index + 1) * height * width, :3
    ]
    ray_directions = dataset[
        img_index * height * width : (img_index + 1) * height * width, 3:6
    ]

    data = []  # list of regenerated pixel values
    for i in range(int(np.ceil(height / chunk_size))):  # iterate over chunks
        # Get chunk of rays
        ray_origins_ = ray_origins[
            i * width * chunk_size : (i + 1) * width * chunk_size
        ].to(model.device)
        ray_directions_ = ray_directions[
            i * width * chunk_size : (i + 1) * width * chunk_size
        ].to(model.device)
        regenerated_px_values = render_rays(
            model,
            ray_origins_,
            ray_directions_,
            near_plane_distance=near_place_distance,
            far_plane_distance=far_plane_distance,
            num_bins=nb_bins,
        )
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(height, width, 3)
    return img
