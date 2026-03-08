from dataclasses import dataclass
from typing import NamedTuple, Optional, List

import torch
from torch.nn import functional as F

from hjepa.models.decoders import MeNet5Decoder
from hjepa.configs import ConfigBase
from hjepa.models.jepa import ForwardResult
from hjepa import models
from hjepa.plotting.reconstruction import plot_reconstructions


class ReconstructionLossInfo(NamedTuple):
    total_loss: torch.Tensor
    per_step_rec_losses: torch.Tensor
    per_step_pred_losses: torch.Tensor
    pred_loss: torch.Tensor
    rec_loss: torch.Tensor
    loss_name: str = "reconstruction"
    name_prefix: str = ""

    def build_log_dict(self):
        return {
            f"{self.name_prefix}/{self.loss_name}_total_loss": self.total_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_rec_loss": self.rec_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_pred_loss": self.pred_loss.item(),
            **{
                f"{self.name_prefix}/{self.loss_name}_rec_loss_step_{i}": x.item()
                for i, x in enumerate(self.per_step_rec_losses)
            },
            **{
                f"{self.name_prefix}/{self.loss_name}_pred_loss_step_{i}": x.item()
                for i, x in enumerate(self.per_step_pred_losses)
            },
        }


@dataclass
class ReconstructionObjectiveConfig(ConfigBase):
    channels: int = 1
    img_size: int = 28
    target_repr_dim: Optional[int] = None
    decoder_arch: str = "menet5"
    rec_coeff: float = 1.0
    pred_coeff: float = 1.0
    width_factor: int = 1
    plot_every: int = 50000


class ReconstructionObjective(torch.nn.Module):
    def __init__(
        self,
        config: ReconstructionObjectiveConfig,
        repr_dim: int,
        name_prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.name_prefix = name_prefix
        self.repr_dim = repr_dim
        if self.config.target_repr_dim is not None:
            # if the target is a vector we make the decoder by just
            # flipping gencoder mlp layers.
            mlp_layers = (
                list(map(int, self.config.decoder_arch.split("-")))
                if self.config.decoder_arch != ""
                else []
            )
            self.decoder = models.misc.build_mlp(
                [repr_dim] + mlp_layers + [self.config.target_repr_dim]
            ).cuda()
        else:
            self.decoder = MeNet5Decoder(
                embedding_size=repr_dim,
                z_dim=0,
                output_channels=self.config.channels,
                width_factor=self.config.width_factor,
            ).cuda()
        self.ctr = 0

    def __call__(self, batch, results: List[ForwardResult]) -> ReconstructionLossInfo:
        # results is the list of forward results from jepa levels,
        # from first to last
        result = results[-1]
        if result.ema_encodings is not None:
            pred_loss = (
                (result.ema_encodings[1:] - result.pred_output.predictions[1:])
                .pow(2)
                .mean(dim=(1, 2))
            )
        else:
            pred_loss = (
                (result.encodings[1:] - result.pred_output.predictions[1:])
                .pow(2)
                .mean(dim=(1, 2))
            )
        # pred_loss dim is [T-1]

        encodings = result.encodings

        if len(results) == 1:
            targets = batch.states  # (B, T, C, H, W)
        else:
            skip = (results[0].encodings.shape[0] - 1) // (
                results[-1].encodings.shape[0] - 1
            )
            targets = results[-2].encodings[::skip].transpose(0, 1)  # (B, T, D)

        reconstructions = (
            self.decoder(encodings.view(-1, self.repr_dim))
            .view(
                encodings.shape[0],
                encodings.shape[1],
                *targets.shape[2:],
            )
            .transpose(0, 1)
        )  # (B, T, C, H, W) or (B, T, D)

        rec_error = F.mse_loss(reconstructions, targets, reduction="none")
        rec_error = rec_error.transpose(0, 1)  # (T, B, C, H, W) or (T, B, D)
        per_step_rec_losses = rec_error.flatten(1).mean(dim=1)  # T
        total_loss = (
            per_step_rec_losses.mean() * self.config.rec_coeff
            + pred_loss.mean() * self.config.pred_coeff
        )

        # plot every self.config.plot_every samples
        new_ctr = self.ctr + encodings.shape[1]
        if (
            self.config.target_repr_dim is None
            and new_ctr // self.config.plot_every > self.ctr // self.config.plot_every
        ):
            plot_reconstructions(
                reconstructions, targets, n_images=4, suffix=f"_{new_ctr}"
            )
        self.ctr = new_ctr

        return ReconstructionLossInfo(
            total_loss=total_loss,
            pred_loss=pred_loss.mean(),
            rec_loss=per_step_rec_losses.mean(),
            per_step_pred_losses=pred_loss,
            per_step_rec_losses=per_step_rec_losses,
            name_prefix=self.name_prefix,
        )
