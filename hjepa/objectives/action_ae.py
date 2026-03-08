from dataclasses import dataclass
from typing import NamedTuple, List

import torch
from torch.nn import functional as F

from hjepa.configs import ConfigBase
from hjepa.models.jepa import ForwardResult
from hjepa.objectives.kl import calc_kl_continuous


class AAELossInfo(NamedTuple):
    total_loss: torch.Tensor
    recon_loss: torch.Tensor
    kl_loss: torch.Tensor
    loss_name: str = "aae"
    name_prefix: str = ""

    def build_log_dict(self):
        return {
            f"{self.name_prefix}/{self.loss_name}_total_loss": self.total_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_recon_loss": self.recon_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_kl_loss": self.kl_loss.item(),
        }


@dataclass
class AAEObjectiveConfig(ConfigBase):
    recon_coeff: float = 1.0
    kl_coeff: float = 0.1
    uniform_prior: bool = True


class AAEObjective(torch.nn.Module):
    def __init__(self, config: AAEObjectiveConfig, name_prefix: str = ""):
        super().__init__()
        self.config = config
        self.name_prefix = name_prefix
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, batch, results: List[ForwardResult]) -> AAELossInfo:
        result = results[-1].aae_output

        # convert pred_actions from (T, BS, AD) to (BS, T, AD)
        pred_actions = result.pred_actions.transpose(0, 1)
        device = pred_actions.device

        if len(batch.actions.shape) == 4:
            # dealing with prechunked actions
            # (BS, chunks, chunk_size, 8) --> (BS, chunks * chunk_size, 8)
            actions = batch.actions.view(
                batch.actions.shape[0], -1, batch.actions.shape[-1]
            )
            pred_actions = pred_actions.view(
                pred_actions.shape[0], -1, pred_actions.shape[-1]
            )
        else:
            actions = batch.actions

        recon_loss = F.mse_loss(pred_actions, actions.to(device), reduction="none")

        # mean across action dim, sum across time steps, mean across batch
        recon_loss = recon_loss.mean(dim=-1).sum(dim=-1).mean()

        if self.config.uniform_prior:
            kl_loss = calc_kl_continuous(
                uniform_prior=True,
                posteriors=result.latents,
                posterior_mus=result.latent_mean,
                posterior_vars=result.latent_std,
                reduction=False,
            )
            # sum time, mean batch and LD
            kl_loss = kl_loss.mean(dim=-1).mean(dim=-1).sum()
        else:
            raise NotImplementedError("Only uniform prior is supported")

        total_loss = (
            self.config.recon_coeff * recon_loss + self.config.kl_coeff * kl_loss
        )

        return AAELossInfo(
            total_loss=total_loss,
            recon_loss=recon_loss,
            kl_loss=kl_loss,
            name_prefix=self.name_prefix,
        )
