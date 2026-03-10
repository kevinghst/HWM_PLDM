from typing import Optional, Union, Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal

from pldm.models.utils import *
from pldm.models.predictors.enums import *
from pldm.models.predictors.sequence_predictor import SequencePredictor
from pldm.models.predictors.conv_predictors import (
    ConvPredictor,
    ConvDistangledPredictor,
    ConvLocalPredictor,
    MLPLocalPredictor,
)
from pldm.models.predictors.mlp_predictors import MLPPredictor, MLPPredictorV2
from pldm.models.predictors.rnn_predictors import RNNPredictorV2
from pldm.models.predictors.ensemble_predictors import *


class RSSMPredictor(torch.nn.Module):
    def __init__(
        self,
        rnn_state_dim: int,
        z_dim: int,
        action_dim: int = 2,
        min_var: float = 0,
        use_action_only: bool = True,
        z_stochastic: bool = True,
    ):
        super().__init__()
        self.rnn_state_dim = rnn_state_dim
        self.z_dim = z_dim
        self.input_dim = z_dim + action_dim
        self.action_dim = action_dim
        self.prior_mu_net = nn.Linear(self.rnn_state_dim, self.z_dim)
        self.prior_var_net = nn.Linear(self.rnn_state_dim, self.z_dim)
        self.rnn = torch.nn.GRUCell(self.input_dim, self.rnn_state_dim)
        self.min_var = min_var
        self.use_action_only = use_action_only
        self.z_stochastic = z_stochastic

    def forward(self, sampled_prior, action, rnn_state, **kwargs):
        if action is not None and self.use_action_only:
            rnn_input = action  # torch.cat([sampled_prior, action], dim=-1)
        elif action is not None and not self.use_action_only:
            rnn_input = torch.cat([sampled_prior, action], dim=-1)
        else:
            rnn_input = sampled_prior

        rnn_state_new = self.rnn(rnn_input, rnn_state)
        prior_mu = self.prior_mu_net(rnn_state_new)
        prior_var = self.prior_var_net(rnn_state_new)
        return rnn_state_new, prior_mu, prior_var

    def burn_in(self, *args, **kwargs):
        return None

    def predict_sequence(
        self,
        enc: torch.Tensor,
        actions: torch.Tensor,
        h: torch.Tensor,
        latents: Optional[torch.Tensor] = None,
    ):
        initial_belief = enc
        batch_size = enc.shape[0]
        sampled_prior_state = torch.zeros(batch_size, self.z_dim).to(enc.device)
        sampled_prior_states = []
        beliefs = []
        rnn_belief = initial_belief
        for i in range(len(actions)):
            rnn_belief, prior_mu, prior_var = self(
                sampled_prior_state, actions[i], rnn_belief
            )
            prior_var = F.softplus(prior_var) + self.min_var
            z = Normal(prior_mu, (prior_var))
            if latents is not None:
                sampled_prior_state = latents[i]
            else:
                if self.z_stochastic:
                    sampled_prior_state = z.sample()
                else:
                    sampled_prior_state = prior_mu  # Deterministic: use mean
            sampled_prior_states.append(sampled_prior_state)
            beliefs.append(rnn_belief)
        beliefs = torch.stack(beliefs, dim=0)
        return beliefs

    def predict_sequence_posterior(
        self,
        encs: torch.Tensor,
        h: torch.Tensor,
        hjepa: torch.nn.Module,
    ):
        result = []
        T = encs.shape[0] + 1
        batch_size = encs.shape[1]
        rnn_state = encs[0]
        sampled_posterior_state = torch.zeros(batch_size, self.z_dim).to(encs.device)
        for i in range(T - 1):
            rnn_state = hjepa.predictor_l2(
                sampled_prior=sampled_posterior_state, action=None, rnn_state=rnn_state
            )
            posterior_mu, posterior_var = hjepa.posterior_l2(encs[i + 1], rnn_state)
            posterior_var = F.softplus(posterior_var) + self.min_var
            if (
                hasattr(hjepa.level2.predictor, "z_stochastic")
                and not hjepa.level2.predictor.z_stochastic
            ):
                sampled_posterior_state = posterior_mu  # Deterministic: use mean
            else:
                sampled_posterior_state = Normal(posterior_mu, (posterior_var)).sample()
            prediction = hjepa.decoder(rnn_state, sampled_posterior_state)
            result.append(prediction)
        return result

    def predict_decode_sequence(
        self,
        enc: torch.Tensor,
        h: torch.Tensor,
        latents: torch.Tensor,
        decoder: torch.nn.Module,
    ):
        beliefs = self.predict_sequence(enc, [None] * latents.shape[0], h, latents)
        return decoder(beliefs, latents)


class TransformerPredictor(SequencePredictor):
    def __init__(
        self,
        config,
        repr_dim,
        action_dim=2,
        pred_proprio_dim=0,
        pred_obs_dim=0,
        backbone_ln: Optional[torch.nn.Module] = None,
    ):
        super(TransformerPredictor, self).__init__(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
        )

        # Transformer encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=repr_dim,
            nhead=config.transformer_nhead,
            dim_feedforward=config.transformer_dim_feedforward,
            dropout=config.dropout,
            activation=config.transformer_activation,
            batch_first=True,
        )

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=config.transformer_num_layers,
            norm=nn.LayerNorm(repr_dim) if config.predictor_ln else None,
        )

        # Positional encoding
        self.pos_encoder = nn.Parameter(
            torch.randn(1, config.transformer_max_seq_len, repr_dim)
        )

        # Input projection for action
        self.action_proj = nn.Linear(action_dim, repr_dim)

        # Output projection
        self.output_proj = nn.Linear(repr_dim, repr_dim)

    def forward(self, current_state, curr_action=None, **kwargs):
        """
        Forward pass with teacher forcing support
        Args:
            current_state: Current state embedding (B, D) or (B, 1, D)
            curr_action: Current action (B, A)
            **kwargs: Additional arguments
        Returns:
            next_state: Predicted next state (B, D)
        """
        if len(current_state.shape) > 2:
            current_state = current_state.squeeze(1)

        batch_size = current_state.shape[0]

        # Add positional encoding
        pos_encoding = self.pos_encoder[:, :1, :]  # Use just one position token
        current_state_with_pos = current_state + pos_encoding.squeeze(1)

        # Prepare for transformer input: (B, D) -> (B, 1, D)
        current_state_seq = current_state_with_pos.unsqueeze(1)

        # Prepare action input
        if curr_action is not None:
            # Project action to same dimension as state
            curr_action_proj = self.action_proj(curr_action)  # (B, D)
            curr_action_seq = curr_action_proj.unsqueeze(1)
            input_seq = torch.cat(
                [current_state_seq, curr_action_seq], dim=1
            )  # (B, 2, D) Combine state and action as a sequence
        else:
            input_seq = current_state_seq  # (B, 1, D)

        # Apply transformer with dropout
        forced_dropout = False
        if forced_dropout:
            self.transformer.train()
        else:
            self.transformer.eval()

        if self.config.use_checkpointing and self.training:

            def create_custom_forward():
                def custom_forward(*inputs):
                    return self.transformer(*inputs)

                return custom_forward

            transformer_out = torch.utils.checkpoint.checkpoint(
                create_custom_forward(), input_seq, use_reentrant=False
            )
        else:
            transformer_out = self.transformer(input_seq)

        # Get the first token's output (state token) and project
        next_state = self.output_proj(transformer_out[:, 0])

        return next_state

    def _is_transformer(self):
        return True


def build_single_predictor(
    config: PredictorConfig,
    repr_dim: int,
    action_dim: int,
    pred_proprio_dim: Union[int, tuple],
    pred_loc_dim: Union[int, tuple],
    pred_obs_dim: Union[int, tuple],
    backbone_ln: Optional[torch.nn.Module] = None,
    normalizer=None,
):
    arch = config.predictor_arch

    if arch == "mlp":
        predictor = MLPPredictor(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
            pred_loc_dim=pred_loc_dim,
            backbone_ln=backbone_ln,
        )
    elif arch == "mlp_v2":
        predictor = MLPPredictorV2(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
            pred_loc_dim=pred_loc_dim,
            backbone_ln=backbone_ln,
        )
    elif arch == "conv2":
        predictor = ConvPredictor(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
        )
    elif arch == "conv_distangled":
        predictor = ConvDistangledPredictor(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
        )
    elif arch == "conv_local":
        predictor = ConvLocalPredictor(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_loc_dim=pred_loc_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
            normalizer=normalizer,
        )
    elif arch == "mlp_local":
        predictor = MLPLocalPredictor(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_loc_dim=pred_loc_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
            normalizer=normalizer,
        )
    elif arch == "rnnV2":
        predictor = RNNPredictorV2(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
        )
    elif arch == "transformer":
        predictor = TransformerPredictor(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
        )
    elif arch == "conv":
        predictor = PixelPredictorConv(action_dim=action_dim)
    elif arch == "id":
        predictor = IDPredictor()
    else:
        predictor = Predictor(arch, repr_dim, action_dim=action_dim, z_dim=z_dim)

    return predictor


def build_ensemble_predictor(
    predictor_creator: Callable[[], nn.Module],
    config: PredictorConfig,
    repr_dim: int,
    action_dim: int,
    pred_proprio_dim: Union[int, tuple],
    pred_loc_dim: Union[int, tuple],
    pred_obs_dim: Union[int, tuple],
    backbone_ln: Optional[torch.nn.Module] = None,
    normalizer=None,
):
    arch = config.predictor_arch
    if arch == "mlp":
        predictor = EnsemblePredictor(
            predictor_creator=predictor_creator,
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
        )
    elif arch == "rnnV2":
        predictor = EnsembleRNNPredictor(
            predictor_creator=predictor_creator,
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
        )
    elif arch == "conv_distangled":
        predictor = EnsembleDistangledPredictor(
            predictor_creator=predictor_creator,
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
        )
    else:
        raise NotImplementedError(
            f"Ensemble predictor for {arch} is not implemented yet."
        )

    return predictor


def build_predictor(
    config: PredictorConfig,
    repr_dim: int,
    action_dim: int,
    pred_proprio_dim: Union[int, tuple],
    pred_loc_dim: Union[int, tuple],
    pred_obs_dim: Union[int, tuple],
    backbone_ln: Optional[torch.nn.Module] = None,
    normalizer=None,
):

    predictor_creator = lambda: build_single_predictor(
        config=config,
        repr_dim=repr_dim,
        action_dim=action_dim,
        pred_proprio_dim=pred_proprio_dim,
        pred_loc_dim=pred_loc_dim,
        pred_obs_dim=pred_obs_dim,
        backbone_ln=backbone_ln,
        normalizer=normalizer,
    )

    if config.ensemble_size > 1:
        return build_ensemble_predictor(
            predictor_creator=predictor_creator,
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_loc_dim=pred_loc_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
            normalizer=normalizer,
        )
    else:
        return predictor_creator()
