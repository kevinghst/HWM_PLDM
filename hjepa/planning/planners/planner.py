from abc import ABC, abstractmethod
from typing import NamedTuple
from hjepa.planning import objectives_v2 as objectives
import torch
from typing import Optional


class PlanningResult(NamedTuple):
    ensemble_predictions: torch.Tensor  # the ensemble of predicted encodings
    ensemble_obs_component: (
        torch.Tensor
    )  # the observation component of the ensemble of predicted encodings
    ensemble_proprio_component: Optional[
        torch.Tensor
    ]  # the proprioception component of the ensemble of predicted encodings
    ensemble_raw_locations: Optional[torch.Tensor]
    pred_obs: torch.Tensor  # the observation component of the predicted encoding
    pred_proprio: torch.Tensor  # the proprioception component of the predicted encoding
    pred_location: torch.Tensor  # the location component of the predicted encoding
    raw_locations: torch.Tensor
    actions: torch.Tensor
    locations: torch.Tensor
    # locations that the model has planned to achieve
    losses: torch.Tensor


class Planner(ABC):
    def __init__(self):
        self.objective = None

    def set_objective(self, objective: objectives.BaseMPCObjective):
        self.objective = objective

    @abstractmethod
    def plan(self, obs: torch.Tensor, steps_left: int):
        pass

    def reset_targets(self, targets: torch.Tensor, repr_input: bool = False):
        self.objective.set_target(targets, repr_input=repr_input)
