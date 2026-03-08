from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
import numpy as np
import random

from pldm.models.misc import (
    build_mlp,
    PriorContinuous,
    PosteriorContinuous,
    DiscreteNet,
)
from pldm import models
from .utils import *
from pldm.models.enums import PredictorConfig, PredictorOutput
import dataclasses
from functorch import combine_state_for_ensemble, vmap
from torch.func import stack_module_state, functional_call
import copy


class SequencePredictor(torch.nn.Module):
    def __init__(
        self,
        config,
        repr_dim,
        action_dim: Optional[int] = None,
        pred_proprio_dim: Optional[Union[int, tuple]] = 0,
        pred_obs_dim: Optional[Union[int, tuple]] = 0,
        backbone_ln: Optional[torch.nn.Module] = None,
        ensemble_size: int = 1,
    ):
        super().__init__()
        self.config = config
        self.repr_dim = repr_dim  # may need to flatten for prior and posterior...
        self.posterior_drop_p = config.posterior_drop_p
        self.posterior_input_type = config.posterior_input_type
        self.z_discrete = config.z_discrete
        self.action_dim = action_dim
        self.pred_proprio_dim = pred_proprio_dim
        self.pred_obs_dim = pred_obs_dim
        self.ensemble = config.ensemble_size > 1
        self.ensemble_size = ensemble_size
        self.ensemble_params = None

        if config.tie_backbone_ln:
            self.final_ln = backbone_ln
        elif config.predictor_ln:
            self.final_ln = nn.LayerNorm(repr_dim)
        else:
            self.final_ln = nn.Identity()

        if config.z_dim is not None and config.z_dim > 0:
            if self.z_discrete:
                self.prior_model = DiscreteNet(
                    input_dim=repr_dim,
                    arch=config.prior_arch,
                    z_discrete_dim=config.z_discrete_dim,
                    z_discrete_dists=config.z_discrete_dists,
                    min_std=config.z_min_std,
                )
                self.posterior_model = DiscreteNet(
                    input_dim=config.posterior_input_dim,
                    arch=config.posterior_arch,
                    z_discrete_dim=config.z_discrete_dim,
                    z_discrete_dists=config.z_discrete_dists,
                    min_std=config.z_min_std,
                )
                self.latent_merger = nn.Linear(
                    config.z_discrete_dim * config.z_discrete_dists, config.z_dim
                )

            else:
                self.prior_model = PriorContinuous(
                    input_dim=repr_dim,
                    arch=config.prior_arch,
                    z_dim=config.z_dim,
                    min_std=config.z_min_std,
                )
                self.posterior_model = PosteriorContinuous(
                    input_dim=repr_dim,
                    arch=config.posterior_arch,
                    z_dim=config.z_dim,
                    min_std=config.z_min_std,
                    posterior_input_type=config.posterior_input_type,
                )
        else:
            self.prior_model = None
            self.posterior_model = None

    def _is_rnn(self):
        if self.ensemble:
            return self.predictors[0].__class__ == RNNPredictorV2
        else:
            return self.__class__ == RNNPredictorV2

    def _is_transformer(self):
        # Base implementation, overridden in TransformerPredictor
        return False

    def forward_multiple(
        self,
        state_encs,
        actions,
        T,
        latents=None,
        flatten_output=False,
        compute_posterior=False,
        forced_dropout=False,
        ensemble_input=False,
    ):
        """
        This does multiple steps
        Parameters:
            state_encs: (t, BS, input_dim) OR (t, ensemble_size, BS, input_dim) if ensemble_input
            actions: (t-1, BS, action_dim)
            T: timesteps to propagate forward
        Output:
            state_predictions: (T, BS, hidden_dim)
            rnn_states: (T, BS, hidden_dim)
        """

        bs = state_encs.shape[1]
        current_state = state_encs[0]

        if self.ensemble and not ensemble_input:
            current_state = current_state.unsqueeze(0).expand(
                self.config.ensemble_size, *current_state.shape
            )

        state_predictions = [current_state]

        if self._is_rnn():
            # repeat num layers
            if self.ensemble:
                # ensemble_size, num_layers, bs, repr_dim
                current_state = (
                    current_state.unsqueeze(1)
                    .repeat(1, self.num_layers, 1, 1)
                    .contiguous()
                )
            else:
                # num_layers, bs, repr_dim
                current_state = (
                    current_state.unsqueeze(0)
                    .repeat(self.num_layers, 1, 1)
                    .contiguous()
                )

        prior_mus = []
        prior_vars = []
        prior_logits = []
        priors = []
        posterior_mus = []
        posterior_vars = []
        posterior_logits = []
        posteriors = []

        # Determine if we should use teacher forcing for this sequence
        use_teacher_forcing = (
            self._is_transformer()  # Check if it's a transformer
            and self.config.use_teacher_forcing
            and self.training
            and random.random() < self.config.transformer_teacher_forcing_ratio
        )

        for i in range(T):
            predictor_input = []
            if self.prior_model is not None:
                if self.ensemble:
                    raise NotImplementedError(
                        "EnsemblePredictor with KL reg not implemented"
                    )

                prior_stats = self.prior_model(flatten_conv_output(current_state))
                # z is of shape BxD

                if latents is not None:
                    prior = latents[i]
                else:
                    prior = self.prior_model.sample(prior_stats)

                if self.z_discrete:
                    prior = self.latent_merger(prior)
                    prior_logits.append(prior_stats)
                else:
                    mu, var = prior_stats
                    prior_mus.append(mu)
                    prior_vars.append(var)

                priors.append(prior)

                if compute_posterior:
                    # compute posterior
                    # if self.posterior_input_type == "term_states":
                    #     posterior_input = torch.cat(
                    #         [
                    #             flatten_conv_output(current_state),
                    #             flatten_conv_output(state_encs[i + 1]),
                    #         ],
                    #         dim=-1,
                    #     )
                    # elif self.posterior_input_type == "actions":
                    #     posterior_input = actions[i]

                    posterior_stats = self.posterior_model(
                        x_prev=current_state,
                        x_next=state_encs[i + 1],
                        action=actions[i] if actions is not None else None,
                    )

                    # posterior_stats = self.posterior_model(posterior_input)
                    posterior = self.posterior_model.sample(posterior_stats)

                    if self.z_discrete:
                        posterior = self.latent_merger(posterior)
                        posterior_logits.append(posterior_stats)
                    else:
                        posterior_mu, posterior_var = posterior_stats
                        posterior_mus.append(posterior_mu)
                        posterior_vars.append(posterior_var)

                    posteriors.append(posterior)

                    z_input = posterior

                    if (
                        self.posterior_drop_p
                        and np.random.random() < self.posterior_drop_p
                    ):
                        z_input = prior
                        # TODO check this. seems like a bug. not supposed to replace all posterior with prior
                    predictor_input.append(z_input)
                else:
                    predictor_input.append(prior)
            else:
                prior = None
                predictor_input.append(actions[i])

            assert len(predictor_input) > 0
            curr_action = (
                torch.cat(predictor_input, dim=-1) if predictor_input else None
            )

            if self._is_transformer():
                if self.ensemble:
                    raise NotImplementedError(
                        "EnsemblePredictor with Transformer not implemented"
                    )
                next_state = self.forward(
                    current_state,
                    curr_action=curr_action,
                    forced_dropout=forced_dropout,
                )
                current_state = next_state

            elif self._is_rnn():
                next_state, next_hidden_state = self.forward_rnn(
                    current_state,
                    torch.cat(predictor_input, dim=-1),
                    forced_dropout=forced_dropout,
                )
                current_state = next_hidden_state

            else:
                next_state = self.forward(
                    current_state,
                    torch.cat(predictor_input, dim=-1),
                    forced_dropout=forced_dropout,
                )
                current_state = next_state

            state_predictions.append(next_state)

            # Update current state based on teacher forcing (only for Transformer)
            if use_teacher_forcing and i < T - 1:
                # Use the ground truth next state if teacher forcing is active
                current_state = state_encs[i + 1]

        t = len(state_predictions)
        state_predictions = torch.stack(state_predictions)

        if self.ensemble:
            first_part_shape = (t, self.config.ensemble_size, bs)
        else:
            first_part_shape = (t, bs)

        if flatten_output:
            state_predictions = state_predictions.view(*first_part_shape, -1)

        prior_mus = torch.stack(prior_mus) if prior_mus else None
        prior_vars = torch.stack(prior_vars) if prior_vars else None
        prior_logits = torch.stack(prior_logits) if prior_logits else None
        priors = torch.stack(priors) if priors else None
        posterior_mus = torch.stack(posterior_mus) if posterior_mus else None
        posterior_vars = torch.stack(posterior_vars) if posterior_vars else None
        posterior_logits = torch.stack(posterior_logits) if posterior_logits else None
        posteriors = torch.stack(posteriors) if posteriors else None

        if self.pred_proprio_dim:
            if isinstance(self.pred_proprio_dim, int):
                pred_proprio_dim = self.pred_proprio_dim
            else:
                pred_proprio_dim = self.pred_proprio_dim[0]

            # index using first_part_shape
            obs_slicing = (
                *[slice(None)] * len(first_part_shape),
                slice(None, -pred_proprio_dim),
            )
            proprio_slicing = (
                *[slice(None)] * len(first_part_shape),
                slice(-pred_proprio_dim, None),
            )
            obs_component = state_predictions[obs_slicing]
            proprio_component = state_predictions[proprio_slicing]
            # else:
            #     pred_proprio_channels = self.pred_proprio_dim[0]
            #     obs_component = state_predictions[:, :, :-pred_proprio_channels]
            #     proprio_component = state_predictions[:, :, -pred_proprio_channels:]
        else:
            obs_component = state_predictions
            proprio_component = None

        if self.ensemble:
            # put the first network's prediction as the main prediction
            ensemble_state_predictions = state_predictions
            ensemble_obs_component = obs_component
            ensemble_proprio_component = proprio_component

            state_predictions = state_predictions[:, 0]
            obs_component = obs_component[:, 0]
            proprio_component = (
                proprio_component[:, 0] if proprio_component is not None else None
            )
        else:
            # we treat it as an ensemble of size 1
            ensemble_state_predictions = state_predictions.unsqueeze(1)
            ensemble_obs_component = obs_component.unsqueeze(1)
            if proprio_component is not None:
                ensemble_proprio_component = proprio_component.unsqueeze(1)
            else:
                ensemble_proprio_component = None

        output = PredictorOutput(
            predictions=state_predictions,
            obs_component=obs_component,
            proprio_component=proprio_component,
            ensemble_predictions=ensemble_state_predictions,
            ensemble_obs_component=ensemble_obs_component,
            ensemble_proprio_component=ensemble_proprio_component,
            prior_mus=prior_mus,
            prior_vars=prior_vars,
            prior_logits=prior_logits,
            priors=priors,
            posterior_mus=posterior_mus,
            posterior_vars=posterior_vars,
            posterior_logits=posterior_logits,
            posteriors=posteriors,
        )

        return output


class EnsemblePredictor(SequencePredictor):
    def __init__(
        self,
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
            [
                build_single_predictor(
                    config=config,
                    repr_dim=repr_dim,
                    action_dim=action_dim,
                    pred_proprio_dim=pred_proprio_dim,
                    pred_obs_dim=pred_obs_dim,
                    backbone_ln=backbone_ln,
                )
                for _ in range(self.ensemble_size)
            ]
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

    def forward_rnn(self, current_state, curr_action, **kwargs):
        """
        Input:
            current_state: (ensemble_size, num_layers, BS, repr_dim)
            curr_action: (BS, action_dim)
        Output:
            output: (ensemble_size, BS, repr_dim)
        """

        if self.config.use_vmap:
            raise NotImplementedError("vmap not implemented for RNNs")
            ensemble_size = current_state.shape[0]
            curr_action = curr_action.unsqueeze(0).expand(ensemble_size, -1, -1)
            next_state, next_hidden_state = self.vmap_model(
                self.ensemble_params, self.buffers, (current_state, curr_action)
            )
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


class MLPPredictor(SequencePredictor):
    def __init__(
        self,
        # parent inputs
        config,
        repr_dim,
        action_dim=2,  # action + z_dim
        pred_proprio_dim=0,
        pred_obs_dim=0,
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

    def forward(self, current_state, curr_action, forced_dropout=False, **kwargs):
        out = torch.cat([current_state, curr_action], dim=-1)

        for layer in self.fc:
            if isinstance(layer, nn.Dropout) and forced_dropout:
                out = F.dropout(
                    out, p=layer.p, training=True
                )  # Force dropout even in eval mode
            else:
                out = layer(out)

        out = self.final_ln(out)
        return out


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

        return next_state[0], next_hidden_state

    forward_rnn = forward


ConvPredictorConfig = {
    "a": [(18, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 16, 3, 1, 1)],
    "b": [(18, 32, 5, 1, 2), (32, 32, 5, 1, 2), (32, 16, 5, 1, 2)],
    "c": [(18, 32, 7, 1, 3), (32, 32, 7, 1, 3), (32, 16, 7, 1, 3)],
    "a_proprio": [(20, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 18, 3, 1, 1)],
    "d4rl_b_p": [(20, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 18, 3, 1, 1)],
    "d4rl_c_p": [(20, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 24, 3, 1, 1)],
}


class ConvPredictor(SequencePredictor):
    def __init__(
        self,
        # parent inputs
        config,
        repr_dim,
        action_dim=2,  # action + z_dim
        pred_proprio_dim=0,
        pred_obs_dim=0,
        # child inputs
        num_groups=4,
    ):
        super(ConvPredictor, self).__init__(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
        )
        self.num_groups = num_groups

        # Define convolutional layers
        layers = []
        layers_config = ConvPredictorConfig[config.predictor_subclass]
        for i in range(len(layers_config) - 1):
            in_channels, out_channels, kernel_size, stride, padding = layers_config[i]

            if i == 0:
                in_channels = repr_dim[0] + action_dim

            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            )
            layers.append(nn.GroupNorm(4, out_channels))
            layers.append(nn.ReLU())

        # last layer
        in_channels, out_channels, kernel_size, stride, padding = layers_config[-1]
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        )

        self.layers = nn.Sequential(*layers)

        # Action encoder
        if self.config.action_encoder_arch and self.config.action_encoder_arch != "id":
            layer_dims = [int(x) for x in self.config.action_encoder_arch.split("-")]
            layers = []
            for i in range(len(layer_dims) - 1):
                layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
                layers.append(nn.ReLU())
            # remove last ReLU
            layers.pop()

            self.action_encoder = nn.Sequential(
                *layers,
                Expander2D(w=repr_dim[-2], h=repr_dim[-1]),
            )
            # self.action_dim = layer_dims[-1]
        else:
            self.action_encoder = Expander2D(w=repr_dim[-2], h=repr_dim[-1])

    def forward(self, current_state, curr_action, **kwargs):
        bs, _, h, w = current_state.shape
        curr_action = self.action_encoder(curr_action)
        x = torch.cat([current_state, curr_action], dim=1)
        x = self.layers(x)
        if self.config.residual:
            x = x + current_state

        return x


class RSSMPredictor(torch.nn.Module):
    def __init__(
        self,
        rnn_state_dim: int,
        z_dim: int,
        action_dim: int = 2,
        min_var: float = 0,
        use_action_only: bool = True,
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
                sampled_prior_state = z.sample()
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

    def forward(self, current_state, curr_action=None, forced_dropout=False, **kwargs):
        """
        Forward pass with teacher forcing support
        Args:
            current_state: Current state embedding (B, D) or (B, 1, D)
            curr_action: Current action (B, A)
            forced_dropout: Whether to force dropout for teacher forcing
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
    pred_obs_dim: Union[int, tuple],
    backbone_ln: Optional[torch.nn.Module] = None,
):
    arch = config.predictor_arch

    if arch == "mlp":
        predictor = MLPPredictor(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
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


def build_predictor(
    config: PredictorConfig,
    repr_dim: int,
    action_dim: int,
    pred_proprio_dim: Union[int, tuple],
    pred_obs_dim: Union[int, tuple],
    backbone_ln: Optional[torch.nn.Module] = None,
):
    if config.ensemble_size > 1:
        return EnsemblePredictor(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
        )
    else:
        return build_single_predictor(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
        )
