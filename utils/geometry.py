from dataclasses import dataclass

import torch
import numpy as np


@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray


@dataclass
class Volume: ...
