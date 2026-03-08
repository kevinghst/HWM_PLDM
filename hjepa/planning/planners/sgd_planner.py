from typing import Optional, Callable

import torch

from environments.utils.normalizer import Normalizer
from hjepa.planning import objectives_v2 as objectives
from .enums import SGDConfig
from .planner import Planner, PlanningResult


class SGDPlanner(Planner):
    def __init__(
        self,
        config: SGDConfig,
        model: torch.nn.Module,
        normalizer: Normalizer,
        objective: objectives.BaseMPCObjective,
        l1: bool = True,
        prober: Optional[torch.nn.Module] = None,
        action_normalizer: Optional[Callable] = None,
    ):
        """
        Args:
            next_level_enc: if present, we need to encode the predictions of model using this prior to
                taking loss against objective
        """
        super().__init__()
        self.config = config
        self.model = model
        self.normalizer = normalizer
        self.action_normalizer = action_normalizer
        self.objective = objective
        self.prober = prober
        self.l1 = l1
        self.z_input = self.model.config.action_dim == 0
        self.action_dim = (
            self.model.config.z_dim if self.z_input else self.model.config.action_dim
        )

    def plan(
        self,
        current_state: torch.Tensor,
        plan_size: int,
        repr_input: bool = False,
        curr_proprio_pos: Optional[torch.Tensor] = None,
        curr_proprio_vel: Optional[torch.Tensor] = None,
        diff_loss_idx: Optional[torch.tensor] = None,
    ):
        """
        Args:
            current_state
                if repr_input: (bs, ch, w, h) OR (bs, dim)
                if not repr_input: (bs, ch, w, h)
            repr_input: whether current_state is encoded representation or raw observation
            diff_loss_idx: tensor of (bs) indicating what time steps to compute loss from for each sample
        """
        batch_size = current_state.shape[0]

        if not repr_input:
            current_state = self.model.backbone(current_state.cuda()).encodings

        actions = torch.zeros(
            (batch_size, plan_size, self.action_dim),
            requires_grad=True,
            device=torch.device("cuda"),
        )

        if curr_proprio_pos is not None:
            curr_proprio_pos = curr_proprio_pos.to(current_state.device)

        if curr_proprio_vel is not None:
            curr_proprio_vel = curr_proprio_vel.to(current_state.device)

        opt = torch.optim.Adam((actions,), lr=self.config.lr)
        orig_training_state = self.model.predictor.training
        self.model.predictor.train(True)

        # if isinstance(self.model.predictor, models.RNNPredictorV2):
        #     self.model.predictor.train(True)

        losses = []
        for i in range(self.config.n_iters + 1):
            # normalize without backprop
            if self.action_normalizer is not None:
                actions.data = self.action_normalizer(actions)

            actions_n = actions

            if self.z_input:
                forward_result = self.model.forward_prior(
                    input_states=current_state.detach(),
                    proprio_pos=curr_proprio_pos,
                    proprio_vel=curr_proprio_vel,
                    latents=actions_n.permute(1, 0, 2),
                    repr_input=True,
                )
            else:
                forward_result = self.model.forward_prior(
                    input_states=current_state.detach(),
                    proprio_pos=curr_proprio_pos,
                    proprio_vel=curr_proprio_vel,
                    actions=actions_n.permute(1, 0, 2),
                    repr_input=True,
                )

            pred_result = forward_result.pred_output
            pred_encs = pred_result.predictions
            pred_obs = pred_result.obs_component

            obj_val = self.objective(pred_obs, diff_loss_idx=diff_loss_idx)

            action_regularization = torch.tensor(0.0).to(actions.device)

            if self.config.l2_reg > 0:
                action_regularization += (
                    self.config.l2_reg * actions.pow(2).sum(dim=0).mean()
                )

            if self.config.action_change_reg > 0 and actions.shape[1] > 1:
                action_regularization += (
                    self.config.action_change_reg
                    * (actions[:, 1:] - actions[:, :-1]).pow(2).sum(dim=0).mean()
                )

            obj_val += action_regularization

            if pred_result.priors is not None:
                prior_d = torch.distributions.Normal(
                    loc=pred_result.prior_mus, scale=pred_result.prior_vars
                )
                # mean over dimension and time, sum over batch
                z_reg = (
                    -prior_d.log_prob(actions.transpose(0, 1))
                    .mean(dim=-1)
                    .sum(dim=1)
                    .mean()
                    * self.config.z_reg_coeff
                )
                obj_val += z_reg

            opt.zero_grad()
            if i != self.config.n_iters:  # last iteration is for evaluation only.
                obj_val.backward()
                opt.step()
                losses.append(obj_val.item())

        self.model.predictor.train(orig_training_state)

        if self.action_normalizer is not None:
            actions.data = self.action_normalizer(actions)

        if self.prober is not None:
            pred_locs = torch.stack([self.prober(x) for x in pred_obs])
            unnormed_locations = self.normalizer.unnormalize_location(
                pred_locs
            ).detach()
        else:
            unnormed_locations = None

        if self.l1:
            actions = self.normalizer.unnormalize_action(actions)
        elif not self.z_input:
            actions = self.normalizer.unnormalize_l2_action(actions)

        return PlanningResult(
            ensemble_predictions=pred_result.ensemble_predictions,
            pred_obs=pred_obs.detach(),
            actions=actions.detach(),
            locations=unnormed_locations,
            losses=torch.tensor(losses).detach(),
        )
