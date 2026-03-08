from typing import NamedTuple, Optional
import torch
from omegaconf import MISSING
from dataclasses import dataclass


class D4RLSample(NamedTuple):
    states: torch.Tensor  # [(batch_size), T, D] or [(batch_size), T, C, H, W]
    locations: torch.Tensor  # [(batch_size), T, 2]
    actions: torch.Tensor  # [(batch_size), T-1, 2]
    indices: torch.Tensor  # [(batch_size)] needed for prioritized replay
    proprio_vel: torch.Tensor  # [batch_size, T, D]
    proprio_pos: torch.Tensor  # [batch_size, T, D]
    # for hierarchy below
    l2_states: torch.Tensor
    l2_locations: torch.Tensor
    l2_proprio_vel: torch.Tensor
    l2_proprio_pos: torch.tensor
    l2_actions: torch.Tensor


@dataclass
class D4RLDatasetConfig:
    env_name: str = MISSING
    num_workers: int = 10
    batch_size: int = 1024
    seed: int = 0
    quick_debug: bool = False
    val_fraction: float = 0.2
    train: bool = True
    n_steps: int = 17
    l2_n_steps: int = 0
    location_only: bool = False
    path: Optional[str] = None
    images_path: Optional[str] = None
    val_path: Optional[str] = None
    prioritized: bool = False
    alpha: float = 0.6
    beta: float = 0.4
    crop_length: Optional[int] = None
    stack_states: int = 1
    mixture_expert: float = 0.0
    img_size: int = 64
    random_actions: bool = False
    load_top_down_view: bool = False
    image_based: bool = True
    chunked_actions: bool = False
    substitute_action: Optional[str] = None
    l2_step_skip: int = 4
    location_in_proprio_component: bool = False
    train_l1: bool = True

    def __post_init__(self):
        if not self.image_based:
            self.stack_states = 1
