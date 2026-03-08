from dataclasses import dataclass
from typing import NamedTuple, List

import torch

from hjepa.configs import ConfigBase
from hjepa.models.jepa import ForwardResult


class PredictionLossInfo(NamedTuple):
    total_loss: torch.Tensor
    pred_loss: torch.Tensor
    loss_name: str = "prediction"
    name_prefix: str = ""

    def build_log_dict(self):
        return {
            f"{self.name_prefix}/{self.loss_name}": self.pred_loss.item(),
        }


@dataclass
class PredictionObjectiveConfig(ConfigBase):
    global_coeff: float = 1.0


class PredictionObjective(torch.nn.Module):
    def __init__(
        self,
        config: PredictionObjectiveConfig,
        pred_attr: str = "state",
        name_prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.pred_attr = pred_attr
        self.name_prefix = name_prefix

    def __call__(self, _batch, result: List[ForwardResult]) -> PredictionLossInfo:
        result = result[-1]  # Prediction objective only uses the highest level result

        ensemble_predictions = result.pred_output.ensemble_predictions is not None

        if self.pred_attr == "state":
            encodings = result.backbone_output.encodings[1:]
            if ensemble_predictions:
                predictions = result.pred_output.ensemble_predictions[1:]
            else:
                predictions = result.pred_output.predictions[1:]
        elif self.pred_attr == "obs":
            encodings = result.backbone_output.obs_component[1:]
            if ensemble_predictions:
                predictions = result.pred_output.ensemble_obs_component[1:]
            else:
                predictions = result.pred_output.obs_component[1:]
        elif self.pred_attr == "proprio":
            encodings = result.backbone_output.proprio_component[1:]
            if ensemble_predictions:
                predictions = result.pred_output.ensemble_proprio_component[1:]
            else:
                predictions = result.pred_output.proprio_component[1:]
        elif self.pred_attr == "locations":
            encodings = result.backbone_output.location_component[1:]
            if ensemble_predictions:
                predictions = result.pred_output.ensemble_location_component[1:]
            else:
                predictions = result.pred_output.location_component[1:]
        elif self.pred_attr == "raw_locations":
            encodings = result.backbone_output.raw_locations[1:]
            if ensemble_predictions:
                predictions = result.pred_output.ensemble_raw_locations[1:]
            else:
                predictions = result.pred_output.raw_locations[1:]
        else:
            raise NotImplementedError

        if result.ema_backbone_output is not None:
            if self.pred_attr == "state":
                encodings = result.ema_backbone_output.encodings[1:]
            elif self.pred_attr == "obs":
                encodings = result.ema_backbone_output.obs_component[1:]
            elif self.pred_attr == "proprio":
                encodings = result.ema_backbone_output.proprio_component[1:]
            else:
                raise NotImplementedError

        if ensemble_predictions:
            ensemble_size = predictions.shape[1]
            encodings = encodings.unsqueeze(1).repeat(
                1, ensemble_size, *([1] * (encodings.dim() - 1))
            )

        pred_loss = (encodings - predictions).pow(2).mean()

        return PredictionLossInfo(
            total_loss=pred_loss * self.config.global_coeff,
            pred_loss=pred_loss,
            loss_name=f"prediction_{self.pred_attr}",
        )
