from typing import Optional

import torch

from pldm.models.utils import *
from pldm.models.predictors.enums import *
from pldm.models.predictors.sequence_predictor import SequencePredictor


class RNNPredictorV2(SequencePredictor):
    def __init__(
        self,
        # parent inputs
        config,
        repr_dim: int = 512,
        action_dim: Optional[int] = None,
        pred_proprio_dim=0,
        pred_obs_dim=0,
        # child inputs
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

        self.num_layers = config.rnn_layers
        # self.input_size = input_size

        self.rnn = torch.nn.GRU(
            input_size=action_dim,
            hidden_size=self.repr_dim,
            num_layers=self.num_layers,
        )

    def forward_and_format(self, rnn_state, rnn_input, **kwargs):
        """
        It is responsible for:
            distangle the observation and proprioception components from the state encodings
            formatting and output of the predictor.
        """

        next_state, next_hidden_state = self.forward(rnn_state, rnn_input, **kwargs)

        out = SingleStepPredictorOutput(
            prediction=next_state[0],
            obs_component=next_state[0],
            proprio_component=None,
            next_hidden_state=next_hidden_state,
        )

        return out

    def forward(self, rnn_state, rnn_input, **kwargs):
        """
        Propagate one step forward
        Parameters:
            rnn_state: (num_layers, bs, dim)
            rnn_input: (bs, a_dim)
        Output:
            output: next_state (bs, dim), next_hidden_state (num_layers, bs, dim)
        """
        # This only does one step

        next_state, next_hidden_state = self.rnn(rnn_input.unsqueeze(0), rnn_state)

        next_state = self.final_ln(next_state)
        next_hidden_state = self.final_ln(next_hidden_state)

        return next_state, next_hidden_state
