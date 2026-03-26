from typing import Optional, Union

import torch
from torch import nn

from pldm.models.misc import (
    PriorContinuous,
    PosteriorContinuous,
    AnalyticalPosterior,
    IdLn,
    DiscreteNet,
)
from pldm.models.utils import *
from pldm.models.predictors.enums import *


class SequencePredictor(torch.nn.Module):
    def __init__(
        self,
        config,
        repr_dim,
        action_dim: Optional[int] = None,
        pred_proprio_dim: Optional[Union[int, tuple]] = 0,
        pred_loc_dim: Optional[Union[int, tuple]] = 0,
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
        self.z_stochastic = config.z_stochastic
        self.action_dim = action_dim
        self.pred_proprio_dim = pred_proprio_dim
        self.pred_loc_dim = pred_loc_dim
        self.pred_obs_dim = pred_obs_dim
        self.ensemble = config.ensemble_size > 1
        self.ensemble_size = ensemble_size
        self.ensemble_params = None

        if config.z_dim is not None and config.z_dim > 0:
            self.prior_model = PriorContinuous(
                input_dim=repr_dim,
                arch=config.prior_arch,
                z_dim=config.z_dim,
                min_std=config.z_min_std,
            )
            if config.posterior_arch == 'id_ln':
                self.posterior_model = IdLn(
                    input_dim=(
                        repr_dim * 2
                        if self.posterior_input_type == "term_states"
                        else config.posterior_input_dim
                    ),
                    z_dim=config.z_dim,
                    min_std=config.z_min_std,
                )
            elif config.posterior_arch == 'analytical':
                self.posterior_model = AnalyticalPosterior(
                    min_std=config.z_min_std,
                )
            else:
                self.posterior_model = PosteriorContinuous(
                    input_dim=(
                        repr_dim * 2
                        if self.posterior_input_type == "term_states"
                        else config.posterior_input_dim
                    ),
                    arch=config.posterior_arch,
                    z_dim=config.z_dim,
                    min_std=config.z_min_std,
                    posterior_input_type=config.posterior_input_type,
                )
        else:
            self.prior_model = None
            self.posterior_model = None

        if self.config.tie_backbone_ln:
            self.final_ln = backbone_ln
        elif self.config.predictor_ln:
            self.final_ln = self._init_finial_ln()
        else:
            self.final_ln = nn.Identity()

    def _init_finial_ln(self):
        return nn.LayerNorm(self.repr_dim)

    def _is_rnn(self):
        if self.ensemble:
            return "RNN" in str(self.predictors[0].__class__)
        else:
            return "RNN" in str(self.__class__)

    def _is_transformer(self):
        # Base implementation, overridden in TransformerPredictor
        return False

    def _separate_obs_proprio_from_fused_repr(self, state_encs, ensemble_input=False):
        """
        Separate the observation and proprioception components from the state encodings.
        This is useful for models that predict both observation and proprioception.

        Parameters:
            state_encs:
                (BS, input_dim)
                OR (ensemble_size, BS, input_dim)
                OR (BS, channels, height, width)
                OR (ensemble_size, BS, channels, height, width)
        """

        if ensemble_input:
            first_part_shape = (state_encs.shape[0], state_encs.shape[1])
        else:
            first_part_shape = (state_encs.shape[0],)

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
            obs_component = state_encs[obs_slicing]
            proprio_component = state_encs[proprio_slicing]
        else:
            obs_component = state_encs
            proprio_component = None

        return obs_component, proprio_component

    def _prepare_ensemble_input(self, tensor):
        """
        Prepare the input tensor for ensemble processing.
        This is useful for models that use ensemble predictions.
        """
        if tensor is None:
            return None

        # Expand the tensor to include the ensemble dimension
        tensor = tensor.unsqueeze(0).expand(self.config.ensemble_size, *tensor.shape)
        return tensor

    def _prepare_ensemble_output(self, tensor):
        """
        Prepare the output tensor for ensemble processing.
        This is useful for models that use ensemble predictions.
        """
        if self.ensemble:
            ensemble_tensor = tensor

            tensor = tensor[:, 0] if tensor is not None else None
        else:
            # we treat it as an ensemble of size 1
            ensemble_tensor = tensor.unsqueeze(1) if tensor is not None else None

        return ensemble_tensor, tensor

    def forward_multiple(
        self,
        state_encs,
        actions,
        T,
        proprio=None,
        locations=None,
        raw_locations=None,
        latents=None,
        flatten_output=False,
        compute_posterior=False,
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

        # if proprio is already somehow included in the state, we extract it
        if self.config.prefused_repr:
            current_obs, current_proprio = self._separate_obs_proprio_from_fused_repr(
                current_state, ensemble_input=ensemble_input
            )
        else:
            current_obs = current_state
            current_proprio = proprio[0] if proprio is not None else None

        current_location = locations[0] if locations is not None else None
        current_raw_location = raw_locations[0] if raw_locations is not None else None

        if self.ensemble and not ensemble_input:
            current_state = self._prepare_ensemble_input(current_state)
            current_obs = self._prepare_ensemble_input(current_obs)
            current_proprio = self._prepare_ensemble_input(current_proprio)
            current_location = self._prepare_ensemble_input(current_location)
            current_raw_location = self._prepare_ensemble_input(current_raw_location)

        state_predictions = [current_state]
        obs_component = [current_obs]
        proprio_component = [current_proprio] if current_proprio is not None else []
        location_component = [current_location] if current_location is not None else []
        raw_locations = (
            [current_raw_location] if current_raw_location is not None else []
        )

        prior_mus = []
        prior_vars = []
        prior_logits = []
        priors = []
        posterior_mus = []
        posterior_vars = []
        posterior_logits = []
        posteriors = []

        for i in range(T):
            predictor_input = []
            if self.prior_model is not None:
                if self.ensemble:
                    raise NotImplementedError(
                        "EnsemblePredictor with KL reg not implemented"
                    )

                prior_stats = self.prior_model(
                    flatten_conv_output(current_state),
                    batch_dim=1 if self._is_rnn() else 0,
                )
                # z is of shape BxD

                if latents is not None:
                    prior = latents[i]
                else:
                    # Deterministic: use only the mean/mode
                    mu, var = prior_stats
                    prior = mu

                # Extract mu, var for logging (reuse if already extracted in deterministic case)
                if not (latents is None and not self.z_stochastic):
                    mu, var = prior_stats
                prior_mus.append(mu)
                prior_vars.append(var)

                priors.append(prior)

                if compute_posterior:
                    if self.posterior_input_type == "term_states":
                        posterior_input = torch.cat(
                            [
                                flatten_conv_output(current_state),
                                flatten_conv_output(state_encs[i + 1]),
                            ],
                            dim=-1,
                        )
                    elif self.posterior_input_type == "actions":
                        posterior_input = actions[i]  # (bs, chunk_size, action_dim)
                        if self.config.posterior_arch != "analytical":
                            # flatten to (bs, chunk_size * action_dim)
                            posterior_input = posterior_input.view(
                                posterior_input.shape[0], -1
                            )

                    posterior_stats = self.posterior_model(posterior_input)
                    # Deterministic: use only the mean/mode
                    posterior_mu, posterior_var = posterior_stats
                    posterior = posterior_mu

                    posterior_mus.append(posterior_mu)
                    posterior_vars.append(posterior_var)

                    posteriors.append(posterior)

                    z_input = posterior

                    predictor_input.append(z_input)
                else:
                    predictor_input.append(prior)
            elif self.posterior_model is not None and actions is not None:
                posterior_input = actions[i]  # (bs, chunk_size, action_dim)
                if self.config.posterior_arch != "analytical":
                    posterior_input = posterior_input.view(posterior_input.shape[0], -1)

                posterior_stats = self.posterior_model(posterior_input)
                posterior_mu, posterior_var = posterior_stats
                posterior = posterior_mu

                predictor_input.append(posterior)

            else:
                prior = None
                predictor_input.append(actions[i])

            assert len(predictor_input) > 0
            curr_action = (
                torch.cat(predictor_input, dim=-1) if predictor_input else None
            )

            pred_output = self.forward_and_format(
                current_state,
                curr_action=torch.cat(predictor_input, dim=-1),
                curr_obs=current_obs,
                curr_proprio=current_proprio,
                curr_location=current_location,
                curr_raw_location=current_raw_location,
                timestep=i,
            )
            current_state = pred_output.prediction
            current_obs = pred_output.obs_component
            current_proprio = pred_output.proprio_component
            current_location = pred_output.location_component
            current_raw_location = pred_output.raw_location

            state_predictions.append(pred_output.prediction)
            obs_component.append(pred_output.obs_component)
            if pred_output.proprio_component is not None:
                proprio_component.append(pred_output.proprio_component)
            if pred_output.location_component is not None:
                location_component.append(pred_output.location_component)
            if pred_output.raw_location is not None:
                raw_locations.append(pred_output.raw_location)

        t = len(state_predictions)
        state_predictions = torch.stack(state_predictions)
        obs_component = torch.stack(obs_component)
        proprio_component = (
            torch.stack(proprio_component) if proprio_component else None
        )
        location_component = (
            torch.stack(location_component) if location_component else None
        )
        raw_locations = torch.stack(raw_locations) if raw_locations else None

        prior_mus = torch.stack(prior_mus) if prior_mus else None
        prior_vars = torch.stack(prior_vars) if prior_vars else None
        prior_logits = torch.stack(prior_logits) if prior_logits else None
        priors = torch.stack(priors) if priors else None
        posterior_mus = torch.stack(posterior_mus) if posterior_mus else None
        posterior_vars = torch.stack(posterior_vars) if posterior_vars else None
        posterior_logits = torch.stack(posterior_logits) if posterior_logits else None
        posteriors = torch.stack(posteriors) if posteriors else None

        ensemble_state_predictions, state_predictions = self._prepare_ensemble_output(
            state_predictions
        )
        ensemble_obs_component, obs_component = self._prepare_ensemble_output(
            obs_component
        )
        ensemble_proprio_component, proprio_component = self._prepare_ensemble_output(
            proprio_component
        )
        ensemble_location_component, location_component = self._prepare_ensemble_output(
            location_component
        )
        ensemble_raw_locations, raw_locations = self._prepare_ensemble_output(
            raw_locations
        )

        if flatten_output:
            ensemble_first_part_shape = (t, self.config.ensemble_size, bs)
            ensemble_state_predictions = ensemble_state_predictions.view(
                *ensemble_first_part_shape, -1
            )
            ensemble_obs_component = ensemble_obs_component.view(
                *ensemble_first_part_shape, -1
            )
            ensemble_proprio_component = (
                ensemble_proprio_component.view(*ensemble_first_part_shape, -1)
                if ensemble_proprio_component is not None
                else None
            )

            first_part_shape = (t, bs)
            state_predictions = state_predictions.view(*first_part_shape, -1)
            obs_component = obs_component.view(*first_part_shape, -1)
            proprio_component = (
                proprio_component.view(*first_part_shape, -1)
                if proprio_component is not None
                else None
            )

            # we dont need to flatten the location and raw locations for now

        output = PredictorOutput(
            predictions=state_predictions,
            obs_component=obs_component,
            proprio_component=proprio_component,
            location_component=location_component,
            raw_locations=raw_locations,
            ensemble_predictions=ensemble_state_predictions,
            ensemble_obs_component=ensemble_obs_component,
            ensemble_proprio_component=ensemble_proprio_component,
            ensemble_location_component=ensemble_location_component,
            ensemble_raw_locations=ensemble_raw_locations,
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

    def forward_and_format(
        self, current_state, curr_action, curr_obs, curr_proprio, **kwargs
    ):
        """
        It is responsible for:
            distangle the observation and proprioception components from the state encodings
            formatting and output of the predictor.
        """

        pred_output = self.forward(
            current_state,
            curr_action=curr_action,
            curr_obs=curr_obs,
            curr_proprio=curr_proprio,
        )

        assert isinstance(pred_output, torch.Tensor), "pred_output should be a tensor"

        if self.config.ensemble_size > 1:
            obs, proprio = self._separate_obs_proprio_from_fused_repr(
                pred_output, ensemble_input=True
            )
        else:
            obs, proprio = self._separate_obs_proprio_from_fused_repr(pred_output)

        out = SingleStepPredictorOutput(
            prediction=pred_output,
            obs_component=obs,
            proprio_component=proprio,
        )

        return out
