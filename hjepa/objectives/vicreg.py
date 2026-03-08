from dataclasses import dataclass
from typing import NamedTuple, Optional, List

import torch
from torch.nn import functional as F

from hjepa.configs import ConfigBase
from hjepa.models.jepa import ForwardResult
from hjepa.models.utils import flatten_conv_output
from functools import reduce
import operator
from hjepa.models.misc import Projector
from hjepa.objectives.prediction import PredictionObjective, PredictionObjectiveConfig


class VICRegLossInfo(NamedTuple):
    total_loss: torch.Tensor
    cov_loss: torch.Tensor
    std_loss: torch.Tensor
    sim_loss: torch.Tensor
    cov_loss_t: torch.Tensor
    std_loss_t: torch.Tensor
    sim_loss_t: torch.Tensor
    loss_name: str = "vicreg"
    name_prefix: str = ""

    def build_log_dict(self):
        return {
            f"{self.name_prefix}/{self.loss_name}_total_loss": self.total_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_cov_loss": self.cov_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_std_loss": self.std_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_sim_loss": self.sim_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_cov_loss_t": self.cov_loss_t.item(),
            f"{self.name_prefix}/{self.loss_name}_std_loss_t": self.std_loss_t.item(),
        }


@dataclass
class VICRegObjectiveConfig(ConfigBase):
    projector: str = "id"
    random_projector: bool = False
    sim_coeff: float = 25.0
    std_coeff: float = 25.0
    cov_coeff: float = 1.0
    std_coeff_t: float = 0.0
    cov_coeff_t: float = 0.0
    cov_per_feature: bool = False
    adjust_cov: bool = True
    cov_chunk_size: Optional[int] = None
    std_margin: float = 1
    std_margin_t: float = 1
    sim_coeff_t: float = 0


class VICRegObjective(torch.nn.Module):
    def __init__(
        self,
        config: VICRegObjectiveConfig,
        repr_dim: int,
        pred_attr: str = "state",
        name_prefix: str = "",
    ):
        super().__init__()
        if isinstance(repr_dim, tuple):
            repr_dim = reduce(operator.mul, repr_dim)
        self.config = config
        self.name_prefix = name_prefix
        self.pred_attr = pred_attr
        self.projector = Projector(
            arch=config.projector,
            embedding=repr_dim,
            random=config.random_projector,
        ).cuda()

        self.pred_objective = PredictionObjective(
            PredictionObjectiveConfig(global_coeff=1.0),
            pred_attr=pred_attr,
            name_prefix=name_prefix,
        )

    def __call__(self, _batch, result: List[ForwardResult]) -> VICRegLossInfo:
        sim_loss = self.pred_objective(_batch, result).pred_loss

        result = result[-1]  # VICReg objective only uses the highest level result

        if self.pred_attr == "state":
            encodings = result.backbone_output.encodings
        elif self.pred_attr == "proprio":
            encodings = result.backbone_output.proprio_component
        elif self.pred_attr == "obs":
            encodings = result.backbone_output.obs_component
        elif self.pred_attr == "locations":
            encodings = result.backbone_output.location_component
        else:
            raise NotImplementedError

        if self.config.sim_coeff_t:
            sim_loss_t = (encodings[1:] - encodings[:-1]).pow(2).mean()
        else:
            sim_loss_t = torch.zeros([1])

        encodings = self.projector(encodings)

        flat_encodings = flatten_conv_output(encodings)

        std_loss = self.std_loss(flat_encodings[:1])

        if self.config.cov_per_feature:
            T, B, ch, h, w = encodings.shape
            # reshape (1, bs, ch, h, w) --> (h*w, bs, ch)
            per_feature_encodings = (
                encodings[:1].reshape(1, B, ch, h * w).permute(0, 3, 1, 2).squeeze(0)
            )
            cov_loss = self.cov_loss(per_feature_encodings)
        else:
            # reshape (1, bs, ch, h, w) --> (w, bs, ch * h * w)
            cov_loss = self.cov_loss(flat_encodings[:1])

        std_loss_t = self.std_loss(
            flat_encodings[1:].permute(1, 0, 2), across_time=True
        )  # (bs, T, repr)
        cov_loss_t = self.cov_loss(
            flat_encodings[1:].permute(1, 0, 2), across_time=True
        )  # (bs, T, repr)

        total_loss = (
            self.config.sim_coeff * sim_loss
            + self.config.cov_coeff * cov_loss.mean()
            + self.config.std_coeff * std_loss.mean()
            + self.config.cov_coeff_t * cov_loss_t.mean()
            + self.config.std_coeff_t * std_loss_t.mean()
            + self.config.sim_coeff_t * sim_loss_t.mean()
        )

        return VICRegLossInfo(
            total_loss=total_loss,
            sim_loss=sim_loss,
            cov_loss=cov_loss.mean(),
            std_loss=std_loss.mean(),
            cov_loss_t=cov_loss_t.mean(),
            std_loss_t=std_loss_t.mean(),
            sim_loss_t=sim_loss_t.mean(),
            loss_name=f"vicreg_{self.pred_attr}",
            name_prefix=self.name_prefix,
        )

    def std_loss(self, x: torch.Tensor, across_time=False):
        x = x - x.mean(dim=1, keepdim=True)  # mean for each dim across batch samples

        if (
            not across_time
            and self.config.std_coeff
            or across_time
            and self.config.std_coeff_t
        ):
            std = torch.sqrt(x.var(dim=1) + 0.0001)

            std_margin = (
                self.config.std_margin_t if across_time else self.config.std_margin
            )
            std_loss = torch.mean(F.relu(std_margin - std), dim=-1)
        else:
            std_loss = torch.zeros([1])

        return std_loss

    def cov_loss(self, x: torch.Tensor, across_time=False):
        batch_size = x.shape[1]
        num_features = x.shape[-1]

        x = x - x.mean(dim=1, keepdim=True)

        if (
            not across_time
            and self.config.cov_coeff
            or across_time
            and self.config.cov_coeff_t
        ):
            cov = torch.einsum("bki,bkj->bij", x, x) / (batch_size - 1)
            diagonals = torch.einsum("bii->bi", cov).pow(2).sum(dim=-1)
            # cov shape is TxDxD

            cov_loss = (cov.pow(2).sum(dim=[-1, -2]) - diagonals).div(num_features)
            if self.config.adjust_cov:
                cov_loss = cov_loss / (
                    num_features - 1
                )  # divide by num of elements on off-diagonal.
                # in orig paper they divide by num_features
                # but the correct version is (num_features - 1)*num_features
        else:
            cov_loss = torch.zeros([1])

        return cov_loss
