from dataclasses import dataclass
from typing import NamedTuple, List

import torch
from torch import nn
from torch.nn import functional as F

from hjepa.configs import ConfigBase
from hjepa.models.jepa import ForwardResult


class AAEProbeChunkInfo(NamedTuple):
    total_loss: torch.Tensor
    angle_loss: torch.Tensor
    rad_diff: torch.Tensor
    loss_name: str = "aae_probe_chunk"
    name_prefix: str = ""

    def build_log_dict(self):
        return {
            f"{self.name_prefix}/{self.loss_name}_total_loss": self.total_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_rad_diff": self.rad_diff.item(),
        }


@dataclass
class AAEProbeChunkConfig(ConfigBase):
    coeff: float = 1.0
    latent_dim: int = 16


class AAEProbeChunkObjective(torch.nn.Module):
    def __init__(self, config: AAEProbeChunkConfig, name_prefix: str = ""):
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

        nn.init.xavier_uniform_(self.angle_prober[1].weight)

    def _calculate_rad_difference(self, pred, gt):
        """
        Input
            pred: (T, BS, 2) - normalized predicted direction
            gt: (T, BS, 2) - normalized ground truth direction
        output:
            rad_diff: (T, BS)
        """

        # calculate the angle difference between the two vectors
        dot = torch.sum(pred * gt, dim=2)
        det = pred[..., 0] * gt[..., 1] - pred[..., 1] * gt[..., 0]
        rad_diff = torch.atan2(det, dot)
        return torch.abs(rad_diff)

    def __call__(self, batch, results: List[ForwardResult]) -> AAEProbeChunkInfo:
        result = results[-1].aae_output

        # we will only use the first chunk for now TODO: maybe change this later?
        latent_mean = result.latent_mean  # (T, BS, LD)
        directions = batch.directions.transpose(0, 1).to(self.device)  # (T, BS, 2)

        # convert unit vector to angle
        directions = directions / directions.norm(
            dim=2, keepdim=True
        )  # Ensure unit vectors

        # get losses
        angle_preds = self.angle_prober(latent_mean).squeeze(-1)
        angle_preds = angle_preds / (angle_preds.norm(dim=-1, keepdim=True) + 1e-8)

        angle_loss = 1 - F.cosine_similarity(angle_preds, directions, dim=-1).mean()

        total_loss = self.config.coeff * angle_loss

        rad_diff = self._calculate_rad_difference(angle_preds, directions).mean()

        return AAEProbeChunkInfo(
            total_loss=total_loss,
            angle_loss=angle_loss,
            rad_diff=rad_diff,
            name_prefix=self.name_prefix,
        )
