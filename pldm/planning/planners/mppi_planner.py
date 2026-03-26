from typing import Optional, Callable, NamedTuple, Union

import torch
from torch import nn

from .mppi_torch import MPPI
from pldm.models.utils import flatten_conv_output
from .planner import PlanningResult
from pldm.models.encoders.enums import BackboneOutput


class DynamicsResult(NamedTuple):
    predictions: torch.Tensor  # the predicted state from the first model from ensemble
    obs_component: (
        torch.Tensor
    )  # the observation component from the first model from ensemble
    proprio_component: (
        torch.Tensor
    )  # the proprioception component from the first model from ensemble
    location_component: (
        torch.Tensor
    )  # the location component from the first model from ensemble
    raw_locations: (
        torch.Tensor
    )  # the raw location component from the first model from ensemble
    ensemble_predictions: torch.Tensor  # the predicted states from the ensemble models
    ensemble_obs_component: (
        torch.Tensor
    )  # the observation components from the ensemble models
    ensemble_proprio_component: (
        torch.Tensor
    )  # the proprioception components from the ensemble models
    ensemble_location_component: (
        torch.Tensor
    )  # the location components from the ensemble models
    ensemble_raw_locations: (
        torch.Tensor
    )  # the raw location components from the ensemble models


class LearnedDynamics:
    def __init__(self, model, state_dim=None):
        self.model = model
        self.dump_dict = None
        self.state_dim = state_dim
        self.max_batch_size = 500

    def __call__(
        self,
        state,
        proprio,
        location,
        raw_location,
        action,
        only_return_last=True,
        flatten_output=True,
    ):
        """
        Params:
            state: [ensemble_size x K x nx]
            proprio: [ensemble_size x K x n_proprio] OR None
            action: [K x nx]
        Output:
            preds: [ensemble_size x K x nx]
            pred_obs: [ensemble_size x K x nx]
        """
        # make sure state is in correct format
        og_shape = state.shape
        ensemble_size = og_shape[0]
        n_samples = og_shape[1]

        if isinstance(self.state_dim, int):
            self.state_dim = (self.state_dim,)

        state = state.unsqueeze(0)  # (t=1, BS, ...)
        proprio = proprio.unsqueeze(0) if proprio is not None else None
        location = location.unsqueeze(0) if location is not None else None
        raw_location = raw_location.unsqueeze(0) if raw_location is not None else None

        # introduce time dimension to action if needed
        if len(action.shape) < 3:
            action = action.unsqueeze(0)

        T = action.shape[0]

        if self.model.predictor.ensemble_size > 1:
            assert state.shape[1] > 1
            if proprio is not None:
                assert proprio.shape[1] > 1
            if location is not None:
                assert location.shape[1] > 1
            if raw_location is not None:
                assert raw_location.shape[1] > 1
            ensemble_input = True
        else:
            assert state.shape[1] == 1
            state = state.squeeze(1)  # remove the ensemble dimension

            if proprio is not None:
                assert proprio.shape[1] == 1
                proprio = proprio.squeeze(1)

            if location is not None:
                assert location.shape[1] == 1
                location = location.squeeze(1)

            if raw_location is not None:
                assert raw_location.shape[1] == 1
                raw_location = raw_location.squeeze(1)

            ensemble_input = False

        if self.model.config.action_dim:
            pred_output = self.model.predictor.forward_multiple(
                state,
                action.float(),
                T,
                proprio=proprio,
                locations=location,
                raw_locations=raw_location,
                ensemble_input=ensemble_input,
            )
        else:
            pred_output = self.model.predictor.forward_multiple(
                state,
                actions=None,
                T=T,
                proprio=proprio,
                latents=action.float(),
                ensemble_input=ensemble_input,
            )

        preds = pred_output.predictions
        pred_obs = pred_output.obs_component
        pred_proprio = pred_output.proprio_component
        pred_location = pred_output.location_component
        pred_raw_locations = pred_output.raw_locations

        ensemble_preds = pred_output.ensemble_predictions
        ensemble_pred_obs = pred_output.ensemble_obs_component
        ensemble_pred_proprio = pred_output.ensemble_proprio_component
        ensemble_pred_location = pred_output.ensemble_location_component
        ensemble_pred_raw_locations = pred_output.ensemble_raw_locations

        if only_return_last:
            preds = preds[-1]
            pred_obs = pred_obs[-1]
            pred_proprio = pred_proprio[-1] if pred_proprio is not None else None
            pred_location = pred_location[-1] if pred_location is not None else None
            pred_raw_locations = (
                pred_raw_locations[-1] if pred_raw_locations is not None else None
            )

            ensemble_preds = ensemble_preds[-1]
            ensemble_pred_obs = ensemble_pred_obs[-1]
            ensemble_pred_proprio = (
                ensemble_pred_proprio[-1] if ensemble_pred_proprio is not None else None
            )
            ensemble_pred_location = (
                ensemble_pred_location[-1]
                if ensemble_pred_location is not None
                else None
            )
            ensemble_pred_raw_locations = (
                ensemble_pred_raw_locations[-1]
                if ensemble_pred_raw_locations is not None
                else None
            )

        # if flatten_output:
        #     preds = flatten_conv_output(preds)  # required for 3rd party MPPI code...
        #     pred_obs = flatten_conv_output(pred_obs)
        #     pred_proprio = (
        #         flatten_conv_output(pred_proprio) if pred_proprio is not None else None
        #     )
        #     ensemble_preds = flatten_conv_output(ensemble_preds)
        #     ensemble_pred_obs = flatten_conv_output(ensemble_pred_obs)
        #     ensemble_pred_proprio = (
        #         flatten_conv_output(ensemble_pred_proprio)
        #         if ensemble_pred_proprio is not None
        #         else None
        #     )

        output = DynamicsResult(
            predictions=preds,
            obs_component=pred_obs,
            proprio_component=pred_proprio,
            location_component=pred_location,
            raw_locations=pred_raw_locations,
            ensemble_predictions=ensemble_preds,
            ensemble_obs_component=ensemble_pred_obs,
            ensemble_proprio_component=ensemble_pred_proprio,
            ensemble_location_component=ensemble_pred_location,
            ensemble_raw_locations=ensemble_pred_raw_locations,
        )

        # we need to return both. preds is used to propagate the state forward. pred_obs is used to take cost
        return output

    def before_planning_callback(self):
        self.orig_training_state = self.model.training
        self.model.train(False)

    def after_planning_callback(self):
        self.model.train(self.orig_training_state)


class RunningCost:
    def __init__(
        self, objective, idx=None, projector=None, cost_dim_range="0:99999999"
    ):
        self.objective = objective
        self.idx = idx
        self.projector = nn.Identity() if projector is None else projector
        self.cost_dim_range = [int(x) for x in cost_dim_range.split(":")]
        self.sum_all_diffs = getattr(objective, "sum_all_diffs", None)
        self.sum_last_n = getattr(objective, "sum_last_n", None)
        if self.sum_last_n is None and self.sum_all_diffs is not None:
            self.sum_last_n = 3 if not self.sum_all_diffs else None

    def __call__(self, state, action):
        """encoding shape is B X D
        Note that B are samples for the same environment
        You want to diff against target_enc of shape (D) retrieved from objective
        """
        objective = self.objective

        if hasattr(objective, "pred_encoder") and objective.pred_encoder is not None:
            if len(state.shape) == 5:
                state = state.reshape(state.shape[0] * state.shape[1], *state.shape[2:])

            pred_encoder = objective.pred_encoder
            if hasattr(pred_encoder, "using_proprio") and pred_encoder.using_proprio:
                obs_dim_info = objective.model.backbone.output_obs_dim
                proprio_dim_info = objective.model.backbone.output_proprio_dim

                if isinstance(obs_dim_info, tuple):
                    obs_channels = obs_dim_info[0]
                else:
                    obs_channels = obs_dim_info

                if isinstance(proprio_dim_info, tuple):
                    proprio_channels = proprio_dim_info[0]
                else:
                    proprio_channels = proprio_dim_info

                obs_state = state[:, :obs_channels, :, :]
                proprio_state = state[
                    :, obs_channels : obs_channels + proprio_channels, :, :
                ]

                state = pred_encoder(obs_state, proprio=proprio_state).encodings
            else:
                state = pred_encoder(state).encodings

            state = flatten_conv_output(state)
        else:
            state = flatten_conv_output(state)

        target = objective.target_enc[self.idx]

        state = self.projector(state)
        target = self.projector(target)

        state = state[..., self.cost_dim_range[0] : self.cost_dim_range[1]]
        target = target[..., self.cost_dim_range[0] : self.cost_dim_range[1]]

        diff = (state - target).pow(2)

        return diff.mean(dim=-1)


class MPPIPlanner:
    def __init__(
        self,
        config,
        model,
        normalizer,
        objective,
        prober: Optional[torch.nn.Module] = None,
        action_normalizer: Optional[Callable] = None,
        num_refinement_steps: int = 1,
        n_envs: int = None,
        l2: bool = False,
        projected_cost: bool = False,
        cost_entity: str = "obs_component",
        cost_dim_range: str = "0:99999999",
        use_latent_mean_std: bool = False,
    ):
        device = next(model.parameters()).device

        latent_actions = l2 and model.config.predictor.z_dim > 0
        self.model = model
        self.config = config
        self.dynamics = LearnedDynamics(
            model,
            state_dim=model.spatial_repr_dim,
        )
        self.normalizer = normalizer
        self.action_normalizer = action_normalizer
        self.prober = prober
        self.latent_actions = latent_actions
        self.use_latent_mean_std = use_latent_mean_std

        action_dim = model.predictor.action_dim
        noise_mu = None
        u_init = None

        if (
            self.use_latent_mean_std
            and
            self.latent_actions
            and hasattr(normalizer, "l2_latent_mean")
            and hasattr(normalizer, "l2_latent_std")
            and normalizer.l2_latent_mean is not None
            and normalizer.l2_latent_std is not None
            and normalizer.l2_latent_mean.numel() == action_dim
            and normalizer.l2_latent_std.numel() == action_dim
        ):
            noise_mu = normalizer.l2_latent_mean.to(device=device, dtype=torch.float32)
            # Use empirical std as per-dim exploration, scaled by config.noise_sigma.
            per_dim_std = (
                normalizer.l2_latent_std.to(device=device, dtype=torch.float32)
                * float(config.noise_sigma)
            ).clamp_min(1e-4)
            noise_sigma = torch.diag(per_dim_std.pow(2))
            u_init = noise_mu
        else:
            noise_sigma = torch.diag(
                torch.tensor(
                    [config.noise_sigma] * action_dim,
                    dtype=torch.float32,
                    device=device,
                )
            )

        self.objective = objective

        self.mppi_costs = [
            RunningCost(
                objective,
                idx=i,
                projector=prober if projected_cost else None,
                cost_dim_range=cost_dim_range,
            )
            for i in range(n_envs)
        ]

        if isinstance(model.spatial_repr_dim, int):
            nx = torch.Size((model.spatial_repr_dim,))
        else:
            nx = torch.Size(model.spatial_repr_dim)

        self.ctrls = [
            MPPI(
                self.dynamics,
                running_cost=self.mppi_costs[i],
                nx=nx,
                noise_sigma=noise_sigma,
                noise_mu=noise_mu,
                u_init=u_init,
                proprio_dim=model.backbone.output_proprio_dim,
                num_samples=config.num_samples,
                lambda_=config.lambda_,
                device=device,
                action_normalizer=action_normalizer,
                u_per_command=-1,
                latent_actions=latent_actions,
                z_reg_coeff=config.z_reg_coeff,
                var_samples=config.var_samples,
                rollout_var_cost=config.rollout_var_cost,
                rollout_obs_var_cost=config.rollout_obs_var_cost,
                rollout_proprio_var_cost=config.rollout_proprio_var_cost,
                rollout_var_discount=config.rollout_var_discount,
                rollout_samples=self.dynamics.model.predictor.ensemble_size,
                cost_entity=cost_entity,
            )
            for i in range(n_envs)
        ]
        self.last_plan_size = None
        self.num_refinement_steps = num_refinement_steps
        self.l2 = l2

    @torch.no_grad()
    def plan(
        self,
        current_state: Union[torch.Tensor, BackboneOutput],
        plan_size: int,
        repr_input: bool = True,
        curr_proprio_pos: Optional[torch.Tensor] = None,
        curr_proprio_vel: Optional[torch.Tensor] = None,
        curr_locations: Optional[torch.Tensor] = None,
        diff_loss_idx: Optional[torch.tensor] = None,
    ):
        """_summary_
        Args:
            current_state (bs, ch, w, h): representation of current obs
            plan_size (int): how many predictions to make into the future

        Returns:
            predictions (plan_size + 1, bs, n)
            actions (bs, plan_size, 2)
            locations (plan_size + 1, bs, 2) - probed locations from the predictions
            losses - set to None for now
        """
        self.dynamics.before_planning_callback()

        # Compute the representation of inital timestep
        if not repr_input:
            if self.model.backbone.using_proprio:
                if curr_proprio_vel is not None and curr_proprio_pos is not None:
                    curr_proprio_states = torch.cat(
                        [curr_proprio_pos, curr_proprio_vel], dim=-1
                    )
                elif curr_proprio_vel is not None:
                    curr_proprio_states = curr_proprio_vel
                elif curr_proprio_pos is not None:
                    curr_proprio_states = curr_proprio_pos
                else:
                    raise ValueError("Need proprio states to plan")

                curr_proprio_states = curr_proprio_states.cuda()
            else:
                curr_proprio_states = None

            backbone_output = self.model.backbone(
                current_state.cuda(),
                proprio=curr_proprio_states,
                locations=curr_locations.cuda() if curr_locations is not None else None,
            )
        else:
            backbone_output = current_state

        current_state = backbone_output.encodings.detach()
        batch_size = current_state.shape[0]

        proprio = backbone_output.proprio_component
        location = backbone_output.location_component
        raw_location = backbone_output.raw_locations

        # forward propagation
        actions = []
        for i in range(batch_size):
            if self.last_plan_size is not None and plan_size < self.last_plan_size:
                for _ in range(self.last_plan_size - plan_size):
                    self.ctrls[i].shift_nominal_trajectory()

            self.ctrls[i].change_horizon(plan_size)

            # add refinement steps?
            actions.append(
                self.ctrls[i].command(
                    state=current_state[i],  # 256
                    proprio=proprio[i] if proprio is not None else None,
                    location=location[i] if location is not None else None,
                    raw_location=raw_location[i] if raw_location is not None else None,
                    shift_nominal_trajectory=False,
                )
            )

        actions = torch.stack(actions)

        ensemble_state_input = self.dynamics.model.predictor._prepare_ensemble_input(
            current_state
        )
        ensemble_proprio_input = self.dynamics.model.predictor._prepare_ensemble_input(
            proprio
        )
        ensemble_location_input = self.dynamics.model.predictor._prepare_ensemble_input(
            location
        )
        ensemble_raw_location = self.dynamics.model.predictor._prepare_ensemble_input(
            raw_location
        )

        dynamics_output = self.dynamics(
            state=ensemble_state_input,
            proprio=ensemble_proprio_input,
            location=ensemble_location_input,
            raw_location=ensemble_raw_location,
            action=actions.permute(1, 0, 2),
            only_return_last=False,
            flatten_output=False,
        )

        pred_obs = dynamics_output.obs_component
        pred_proprio = dynamics_output.proprio_component
        pred_location = dynamics_output.location_component
        pred_raw_locations = dynamics_output.raw_locations

        if self.action_normalizer is not None:
            actions = self.action_normalizer(actions)

        if self.l2:
            if not self.latent_actions:
                actions = self.normalizer.unnormalize_l2_action(actions)
        else:
            actions = self.normalizer.unnormalize_action(actions)

        self.dynamics.after_planning_callback()
        self.last_plan_size = plan_size

        losses = [0]

        if pred_raw_locations is not None:
            unnormed_locations = self.normalizer.unnormalize_location(
                pred_raw_locations
            ).detach()
        elif self.prober is not None:
            pred_locs = torch.stack([self.prober(x) for x in pred_obs])

            unnormed_locations = self.normalizer.unnormalize_location(
                pred_locs
            ).detach()
        else:
            unnormed_locations = None

        return PlanningResult(
            ensemble_predictions=dynamics_output.ensemble_predictions,
            ensemble_obs_component=dynamics_output.ensemble_obs_component,
            ensemble_proprio_component=dynamics_output.ensemble_proprio_component,
            ensemble_raw_locations=dynamics_output.ensemble_raw_locations,
            pred_obs=pred_obs,
            pred_proprio=pred_proprio,
            pred_location=pred_location,
            raw_locations=pred_raw_locations,
            actions=actions,
            locations=unnormed_locations,
            losses=losses,
        )

    def reset_targets(self, targets: torch.Tensor, repr_input: bool = True):
        self.objective.set_target(targets, repr_input=repr_input)
