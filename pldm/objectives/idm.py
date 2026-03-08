from dataclasses import dataclass
from typing import NamedTuple, List

import torch
from torch.nn import functional as F

from pldm.configs import ConfigBase
from pldm.models.jepa import ForwardResult
from pldm.models.utils import *
from pldm.models.misc import MLP
from functools import reduce
import operator


class IDMLossInfo(NamedTuple):
    total_loss: torch.Tensor
    action_loss: torch.Tensor
    loss_name: str = "idm"
    name_prefix: str = ""

    def build_log_dict(self):
        return {
            f"{self.name_prefix}/{self.loss_name}_total_loss": self.total_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_loss": self.action_loss.item(),
        }


CONV_LAYERS_CONFIG = {
    "a": [
        (-1, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        ("fc", -1, 2),
    ],
    "b": [
        (-1, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        ("fc", -1, 2),
    ],
}


@dataclass
class IDMObjectiveConfig(ConfigBase):
    coeff: float = 1.0
    action_dim: int = 2
    arch: str = ""
    arch_subclass: str = "a"
    use_pred: bool = False


class IDMObjective(torch.nn.Module):
    """Inverse Dynamics Model (IDM) objective.
    Trains an action predictor to predict the next action given the current
    state and the next state."""

    def __init__(
        self, config: IDMObjectiveConfig, repr_dim: int, name_prefix: str = ""
    ):
        super().__init__()
        self.config = config
        self.name_prefix = name_prefix

        if config.arch == "conv":
            input_dim = (repr_dim[0] * 2, *repr_dim[1:])
            self.action_predictor = build_conv(
                CONV_LAYERS_CONFIG[config.arch_subclass], input_dim=input_dim
            ).cuda()
        else:
            if isinstance(repr_dim, tuple):
                repr_dim = reduce(operator.mul, repr_dim)
            self.action_predictor = MLP(
                arch=config.arch,
                input_dim=repr_dim * 2,
                output_shape=config.action_dim,
            ).cuda()

    def __call__(self, batch, results: List[ForwardResult]) -> IDMLossInfo:
        result = results[-1]

        encodings = result.backbone_output.encodings

        if self.config.use_pred:
            curr_embeds = result.pred_output.predictions[:-1]
            next_embeds = encodings[1:]
        else:
            curr_embeds = encodings[:-1]
            next_embeds = encodings[1:]

        if self.config.arch == "conv":
            repr_input = torch.cat([curr_embeds, next_embeds], dim=2)
        else:
            curr_embeds = flatten_conv_output(curr_embeds)
            next_embeds = flatten_conv_output(next_embeds)
            repr_input = torch.cat([curr_embeds, next_embeds], dim=-1)

        repr_input = repr_input.flatten(start_dim=0, end_dim=1)

        actions_pred = self.action_predictor(repr_input)
        # need to transpose 0 and 1 to swap time and batch, and only take the first dot's actions

        action_loss = F.mse_loss(
            actions_pred,
            result.actions.flatten(start_dim=0, end_dim=1).to(actions_pred.device),
            reduction="mean",
        )

        return IDMLossInfo(
            total_loss=self.config.coeff * action_loss,
            action_loss=action_loss,
            name_prefix=self.name_prefix,
        )
