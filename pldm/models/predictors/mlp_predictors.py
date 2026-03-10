from typing import Optional

import torch
from torch import nn

from pldm.models.misc import build_mlp
from pldm.models.utils import *
from pldm.models.predictors.enums import *
from pldm.models.predictors.sequence_predictor import SequencePredictor


class MLPPredictor(SequencePredictor):
    def __init__(
        self,
        # parent inputs
        config,
        repr_dim,
        action_dim=2,  # action + z_dim
        pred_proprio_dim=0,
        pred_obs_dim=0,
        pred_loc_dim=0,
        backbone_ln: Optional[torch.nn.Module] = None,
    ):
        super().__init__(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
        )

        self.fc = build_mlp(
            layers_dims=config.predictor_subclass,
            input_dim=repr_dim + action_dim,
            output_shape=repr_dim,
            norm="layer_norm" if config.predictor_ln else None,
            activation="mish",
            dropout=config.dropout,
        )

    def forward(self, current_state, curr_action, **kwargs):
        out = torch.cat([current_state, curr_action], dim=-1)
        out = self.fc(out)
        out = self.final_ln(out)
        return out


class MLPPredictorV2(SequencePredictor):
    def __init__(
        self,
        # parent inputs
        config,
        repr_dim,
        action_dim=2,  # action + z_dim
        pred_proprio_dim=0,
        pred_obs_dim=0,
        pred_loc_dim=0,
        backbone_ln: Optional[torch.nn.Module] = None,
    ):
        super().__init__(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_loc_dim=pred_loc_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
        )
        repr_dim = pred_obs_dim + pred_proprio_dim + pred_loc_dim

        self.fc = build_mlp(
            layers_dims=config.predictor_subclass,
            input_dim=repr_dim + action_dim,
            output_shape=repr_dim,
            norm="layer_norm" if config.predictor_ln else None,
            activation="mish",
            dropout=config.dropout,
        )

        self.location_predictor = build_mlp(
            layers_dims="64",
            input_dim=pred_obs_dim,
            output_shape=2,
            norm=None,
        )

        # kinda hacky - if proprio ln is identity, replace it with real ln
        if self.config.predictor_ln:
            if isinstance(self.final_ln["proprio"], nn.Identity):
                self.final_ln_proprio = nn.LayerNorm(pred_proprio_dim)
            else:
                self.final_ln_proprio = self.final_ln["proprio"]

    def forward_and_format(self, current_state, curr_action, curr_proprio, **kwargs):
        """
        It is responsible for:
            distangle the observation and proprioception components from the state encodings
            formatting and output of the predictor.
        """
        curr_location = kwargs.get("curr_location", None)

        obs, proprio, loc, raw_loc = self.forward(
            current_state,
            curr_action=curr_action,
            curr_proprio=curr_proprio,
            curr_location=curr_location,
        )

        out = SingleStepPredictorOutput(
            prediction=obs,
            obs_component=obs,
            proprio_component=proprio,
            location_component=loc,
            raw_location=raw_loc,
        )

        return out

    def forward(self, current_state, curr_action, **kwargs):

        curr_proprio = kwargs.get("curr_proprio", None)
        curr_location = kwargs.get("curr_location", None)

        out = torch.cat(
            [current_state, curr_proprio, curr_location, curr_action], dim=-1
        )

        out = self.fc(out)

        # separate into pred_obs_dim, pred_proprio_dim, pred_loc_dim

        pred_obs = out[:, : self.pred_obs_dim]
        pred_proprio = out[
            :, self.pred_obs_dim : self.pred_obs_dim + self.pred_proprio_dim
        ]
        pred_loc = out[:, self.pred_obs_dim + self.pred_proprio_dim :]

        raw_locs = self.location_predictor(pred_obs)

        if self.final_ln is not None:
            pred_obs = self.final_ln["obs"](pred_obs)
            pred_loc = self.final_ln["location"](pred_loc)
            pred_proprio = self.final_ln_proprio(pred_proprio)

        return pred_obs, pred_proprio, pred_loc, raw_locs
