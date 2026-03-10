from typing import Optional, Union, Callable

import torch
from torch import nn

from pldm.models.utils import *
from pldm.models.predictors.enums import *
import dataclasses
from functorch import vmap
from torch.func import stack_module_state, functional_call
import copy
from pldm.models.predictors.sequence_predictor import SequencePredictor


class EnsemblePredictor(SequencePredictor):
    def __init__(
        self,
        predictor_creator: Callable[[], nn.Module],
        config: PredictorConfig,
        repr_dim: int,
        action_dim: int,
        pred_proprio_dim: Union[int, tuple],
        pred_obs_dim: Union[int, tuple],
        backbone_ln: Optional[torch.nn.Module] = None,
    ):
        # We get rid of the final layernorm from the ensemble object.
        # since its children have it
        ensemble_config = dataclasses.replace(
            config,
            tie_backbone_ln=False,
            predictor_ln=False,
        )
        self.num_layers = config.rnn_layers

        super().__init__(
            config=ensemble_config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
            ensemble_size=config.ensemble_size,
        )

        self.predictors = nn.ModuleList(
            [predictor_creator() for _ in range(self.ensemble_size)]
        ).to("cuda")

        self.initialize_vmap_model()

    def initialize_vmap_model(self):
        self.ensemble_params, self.buffers = stack_module_state(self.predictors)
        base_model = copy.deepcopy(self.predictors[0])
        base_model = base_model.to("meta")

        def fmodel(params, buffers, inputs):
            x, actions = inputs
            return functional_call(base_model, (params, buffers), (x, actions))

        self.vmap_model = vmap(fmodel)

    def forward(self, current_state, curr_action, **kwargs):
        """
        Input:
            current_state: (ensemble_size, BS, repr_dim)
            curr_action: (BS, action_dim)
        Output:
            output: (ensemble_size, BS, repr_dim)
        """

        if self.config.use_vmap:
            ensemble_size = current_state.shape[0]
            curr_action = curr_action.unsqueeze(0).expand(ensemble_size, -1, -1)
            output = self.vmap_model(
                self.ensemble_params, self.buffers, (current_state, curr_action)
            )
        else:
            output = torch.stack(
                [
                    p(current_state[i], curr_action)
                    for i, p in enumerate(self.predictors)
                ],
                dim=0,
            )

        return output


class EnsembleRNNPredictor(EnsemblePredictor):
    def __init__(
        self,
        predictor_creator: Callable[[], nn.Module],
        config: PredictorConfig,
        repr_dim: int,
        action_dim: int,
        pred_proprio_dim: Union[int, tuple],
        pred_obs_dim: Union[int, tuple],
        backbone_ln: Optional[torch.nn.Module] = None,
    ):
        super().__init__(
            predictor_creator=predictor_creator,
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
        )

    def forward(self, current_state, curr_action, **kwargs):
        """
        Input:
            current_state: (ensemble_size, num_layers, BS, repr_dim)
            curr_action: (BS, action_dim)
        Output:
            output: (ensemble_size, BS, repr_dim)
        """

        if self.config.use_vmap:
            raise NotImplementedError("vmap not implemented for RNNs")
        else:
            output = [
                p(current_state[i], curr_action) for i, p in enumerate(self.predictors)
            ]

            next_state = torch.stack(
                [o[0] for o in output], dim=0
            )  # [ensemble, bs, repr_dim]
            next_hidden_state = torch.stack(
                [o[1] for o in output], dim=0
            )  # [ensemble, num_layers, bs, repr_dim]

        return next_state, next_hidden_state


class EnsembleDistangledPredictor(EnsemblePredictor):
    def __init__(
        self,
        predictor_creator: Callable[[], nn.Module],
        config: PredictorConfig,
        repr_dim: int,
        action_dim: int,
        pred_proprio_dim: Union[int, tuple],
        pred_obs_dim: Union[int, tuple],
        backbone_ln: Optional[torch.nn.Module] = None,
    ):
        super().__init__(
            predictor_creator=predictor_creator,
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
        )

    def initialize_vmap_model(self):
        self.ensemble_params, self.buffers = stack_module_state(self.predictors)
        base_model = copy.deepcopy(self.predictors[0])
        base_model = base_model.to("meta")

        def fmodel(params, buffers, inputs):
            x, actions, proprio = inputs
            return functional_call(base_model, (params, buffers), (x, actions, proprio))

        self.vmap_model = vmap(fmodel)

    def forward(self, current_state, curr_action, curr_proprio, **kwargs):
        """
        Input:
            current_state: (ensemble_size, BS, CH, H, W)
            curr_action: (BS, action_dim)
            curr_proprio: (ensemble_size, BS, pred_proprio_dim)
        Output:
            output: (ensemble_size, BS, repr_dim)
        """

        if self.config.use_vmap:
            ensemble_size = current_state.shape[0]
            curr_action = curr_action.unsqueeze(0).expand(ensemble_size, -1, -1)

            obs, proprio = self.vmap_model(
                self.ensemble_params,
                self.buffers,
                (current_state, curr_action, curr_proprio),
            )
        else:
            output = [
                p(current_state[i], curr_action, curr_proprio[i])
                for i, p in enumerate(self.predictors)
            ]

            obs = torch.stack([o[0] for o in output], dim=0)  # [ensemble, bs, repr_dim]
            proprio = torch.stack(
                [o[1] for o in output], dim=0
            )  # [ensemble, num_layers, bs, repr_dim]

        return obs, proprio

    def forward_and_format(self, current_state, curr_action, curr_proprio, **kwargs):
        obs, proprio = self.forward(
            current_state,
            curr_action=curr_action,
            curr_proprio=curr_proprio,
        )

        out = SingleStepPredictorOutput(
            prediction=obs,
            obs_component=obs,
            proprio_component=proprio,
        )

        return out
