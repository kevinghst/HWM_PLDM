from dataclasses import dataclass
from typing import NamedTuple, List

import torch
from torch.nn import functional as F

from hjepa.configs import ConfigBase
from hjepa import models
from hjepa.models.jepa import ForwardResult
from hjepa.models.utils import *
from functools import reduce
import operator


class ProbeLossInfo(NamedTuple):
    total_loss: torch.Tensor
    probe_loss: torch.Tensor
    loss_name: str = "probe"
    name_prefix: str = ""

    def build_log_dict(self):
        return {
            f"{self.name_prefix}/{self.loss_name}_total_loss": self.total_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_loss": self.probe_loss.item(),
        }


CONV_LAYERS_CONFIG = {
    "a": [
        (-1, 16, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (16, 8, 1, 1, 0),
        ("max_pool", 2, 2, 0),
        ("fc", -1, 2),
    ],
    "b": [
        (-1, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        ("fc", -1, 2),
    ],
    "c": [
        (-1, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        (32, 32, 3, 1, 1),
        ("fc", -1, 2),
    ],
}


@dataclass
class ProbeObjectiveConfig(ConfigBase):
    coeff: float = 1.0
    arch: str = "conv"
    arch_subclass: str = "a"
    use_pred: bool = True  # whether to probe the predictions or the encoder outputs


class ProbeObjective(torch.nn.Module):
    """
    Probe for certain attributes (proprioceptive state, etc) from the representations
    """

    def __init__(
        self,
        config: ProbeObjectiveConfig,
        repr_dim: int,
        pred_dim: int,
        probe_target: str = "",
        name_prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.name_prefix = name_prefix
        self.probe_target = probe_target

        if config.arch == "conv":
            self.prober = build_conv(
                CONV_LAYERS_CONFIG[config.arch_subclass], input_dim=repr_dim
            ).cuda()
        else:
            repr_dim = reduce(operator.mul, repr_dim)
            self.prober = models.MLP(
                arch=config.arch,
                input_dim=repr_dim,
                output_shape=pred_dim,
            ).cuda()

    def __call__(self, batch, results: List[ForwardResult]) -> ProbeLossInfo:
        result = results[-1]

        if self.config.use_pred:
            embeds = result.pred_output.predictions
        else:
            embeds = result.backbone_output.encodings

        targets = getattr(batch, self.probe_target)
        if self.probe_target == "locations":
            targets = targets[:, :, 0]  # first dot only
        targets = targets.flatten(start_dim=0, end_dim=1).to(embeds.device)

        if self.config.arch != "conv":
            embeds = flatten_conv_output(embeds)

        embeds = embeds.flatten(start_dim=0, end_dim=1)

        preds = self.prober(embeds)

        assert preds.shape == targets.shape

        probe_loss = F.mse_loss(
            preds,
            targets,
            reduction="mean",
        )

        return ProbeLossInfo(
            total_loss=self.config.coeff * probe_loss,
            probe_loss=probe_loss,
            name_prefix=self.name_prefix,
        )
