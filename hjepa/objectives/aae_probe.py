from dataclasses import dataclass
from typing import NamedTuple, List

import torch
from torch import nn
from torch.nn import functional as F

from hjepa.configs import ConfigBase
from hjepa.models.jepa import ForwardResult


class AAEProbeInfo(NamedTuple):
    total_loss: torch.Tensor
    init_angle_loss: torch.Tensor
    final_angle_loss: torch.Tensor
    dir_change_loss: torch.Tensor
    dir_change_acc: float
    loss_name: str = "aae_probe"
    name_prefix: str = ""

    def build_log_dict(self):
        return {
            f"{self.name_prefix}/{self.loss_name}_total_loss": self.total_loss.item(),
        }


@dataclass
class AAEProbeConfig(ConfigBase):
    coeff: float = 1.0
    latent_dim: int = 16
    chunk_size: int = 10
    init_angle_coeff: float = 1
    final_angle_coeff: float = 1
    dir_change_coeff: float = 1


class AAEProbeObjective(torch.nn.Module):
    def __init__(self, config: AAEProbeConfig, name_prefix: str = ""):
        """
        Probe for initial direction, final direction, and direction change in AAE encoder
        """
        super().__init__()
        self.config = config
        self.name_prefix = name_prefix
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.angle_prober = nn.Sequential(
            nn.ReLU(), nn.Linear(config.latent_dim, 2)
        ).to(self.device)

        self.dir_change_prober = nn.Sequential(
            nn.ReLU(), nn.Linear(config.latent_dim, config.chunk_size + 1)
        ).to(self.device)

        self.dir_change_criterion = nn.CrossEntropyLoss()

    def __call__(self, batch, results: List[ForwardResult]) -> AAEProbeInfo:
        result = results[-1].aae_output

        # we will only use the first chunk for now TODO: maybe change this later?
        latent_mean = result.latent_mean[0]  # (BS, LD)
        directions = batch.directions[:, : self.config.chunk_size].to(
            self.device
        )  # (BS, chunk_size, 2)

        # convert unit vector to angle
        directions = directions / directions.norm(
            dim=2, keepdim=True
        )  # Ensure unit vectors
        angles = torch.atan2(directions[..., 1], directions[..., 0])

        init_angle = angles[:, 0]  # (BS, 2)
        final_angle = angles[:, -1]  # (BS, 2)

        # get index of direction change. if no change, set it to chunk_size.
        diff = torch.diff(directions, dim=1)
        change_mask = diff.abs().sum(dim=2) > 0
        change_indices = torch.argmax(change_mask.int(), dim=1)
        no_change = ~change_mask.any(dim=1)
        change_indices = torch.where(
            no_change, torch.tensor(directions.shape[1]), change_indices
        )

        # get losses
        angle_preds = self.angle_prober(latent_mean)
        init_angle_loss = F.mse_loss(angle_preds[:, 0], init_angle)
        final_angle_loss = F.mse_loss(angle_preds[:, 1], final_angle)
        dir_change_logits = self.dir_change_prober(latent_mean)
        dir_change_loss = self.dir_change_criterion(dir_change_logits, change_indices)

        # get dir change accuracy
        dir_change_acc = (
            (torch.argmax(dir_change_logits, dim=1) == change_indices).float().mean()
        )

        total_loss = self.config.coeff * (
            self.config.init_angle_coeff * init_angle_loss
            + self.config.final_angle_coeff * final_angle_loss
            + self.config.dir_change_coeff * dir_change_loss
        )

        return AAEProbeInfo(
            total_loss=total_loss,
            init_angle_loss=init_angle_loss,
            final_angle_loss=final_angle_loss,
            dir_change_loss=dir_change_loss,
            dir_change_acc=dir_change_acc,
            name_prefix=self.name_prefix,
        )
