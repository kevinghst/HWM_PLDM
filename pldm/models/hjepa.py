from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
import random

from pldm.configs import ConfigBase
from pldm.models.jepa import JEPA, JEPAConfig, ForwardResult as JEPAForwardResult


@dataclass
class HJEPAConfig(ConfigBase):
    level1: JEPAConfig = JEPAConfig()
    level2: JEPAConfig = JEPAConfig()
    step_skip: int = 4
    disable_l2: bool = False
    freeze_l1: bool = False
    train_l1: bool = False
    l1_n_steps: int = 17
    correct_l2_merge: bool = False


class ForwardResult(NamedTuple):
    level1: JEPAForwardResult
    level2: Optional[JEPAForwardResult]


class HJEPA(torch.nn.Module):
    def __init__(
        self,
        config: JEPAConfig,
        input_dim,
        normalizer=None,
        ppos_dim=0,
        pvel_dim=0,
        loc_dim=0,
    ):
        super().__init__()
        self.config = config
        self.level1 = JEPA(
            config.level1,
            input_dim=input_dim,
            ppos_dim=ppos_dim,
            pvel_dim=pvel_dim,
            loc_dim=loc_dim,
            normalizer=normalizer,
        )
        if not self.config.disable_l2:
            # For L2, use obs_component dimensions, not full encoding (which includes proprio)
            l2_input_dim = (
                self.level1.backbone.output_obs_dim
                if hasattr(self.level1.backbone, "output_obs_dim")
                else self.level1.backbone.output_dim
            )

            # L1's proprio component (, 2, 35, 35), L2 will downsample to (, 2, 16, 16)
            l1_output_proprio_dim = self.level1.backbone.output_proprio_dim
            if isinstance(l1_output_proprio_dim, tuple):
                l2_input_proprio_dim = l1_output_proprio_dim
            else:
                l2_input_proprio_dim = (
                    l1_output_proprio_dim if l1_output_proprio_dim else 0
                )

            self.level2 = JEPA(
                config.level2,
                input_dim=l2_input_dim,
                input_obs_dim=self.level1.backbone.output_obs_dim,
                input_proprio_dim=l2_input_proprio_dim,
                l1_action_dim=config.level1.action_dim,
                step_skip=config.step_skip,
                l2=True,
                normalizer=normalizer,
            )
        else:
            self.level2 = None
        self.normalizer = normalizer

    def encode_actions(self, actions: torch.Tensor):
        """
        encode primitive actions into higher level actions by summing them up.
        ONLY works for Wall dataset.
        Parameters:
            actions: Tensor shape T x bs x action_dim
        Return:
            l2_actions: Tensor shape T // step_skip, bs, action_dim
        """
        T, bs, action_dim = actions.shape
        assert T % self.config.step_skip == 0

        if self.config.correct_l2_merge:
            # unnormalize l1 actions before summing them up
            actions = self.normalizer.unnormalize_action(actions).view(
                T // self.config.step_skip, self.config.step_skip, bs, action_dim
            )
        else:
            actions = actions.view(
                T // self.config.step_skip, self.config.step_skip, bs, action_dim
            )

        l2_actions = actions.sum(dim=1)

        l2_actions = self.normalizer.normalize_l2_action(l2_actions)  # norm mean = 1.86
        return l2_actions

    def forward_prior(
        self,
        input_states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        T: Optional[int] = None,
    ) -> ForwardResult:
        if not self.config.disable_l2:
            raise NotImplementedError(
                "forward_prior should be called for each level individually."
            )
        else:
            return ForwardResult(
                level1=self.level1.forward_prior(input_states, actions, T),
                level2=None,
            )

    def forward_posterior(
        self,
        input_states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        proprio_pos: Optional[torch.Tensor] = None,
        proprio_vel: Optional[torch.Tensor] = None,
        locations: Optional[torch.Tensor] = None,
        chunked_locations: Optional[torch.Tensor] = None,
        chunked_proprio_pos: Optional[torch.Tensor] = None,
        chunked_proprio_vel: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        l2_states: Optional[torch.Tensor] = None,
        l2_actions: Optional[torch.Tensor] = None,
        l2_proprio_vel: Optional[torch.Tensor] = None,
        l2_proprio_pos: Optional[torch.Tensor] = None,
    ) -> ForwardResult:
        forward_result_l1 = None

        if self.config.train_l1:
            # sample a subsequence of length l1_n_steps
            sub_idx = random.randint(0, input_states.shape[0] - self.config.l1_n_steps)
            l1_input_states = input_states[sub_idx : sub_idx + self.config.l1_n_steps]
            l1_actions = actions[sub_idx : sub_idx + self.config.l1_n_steps - 1]

            forward_result_l1 = self.level1.forward_posterior(
                l1_input_states,
                l1_actions,
                proprio_pos=proprio_pos,
                proprio_vel=proprio_vel,
                locations=locations,
                chunked_locations=chunked_locations,
                chunked_proprio_pos=chunked_proprio_pos,
                chunked_proprio_vel=chunked_proprio_vel,
                encode_only=False,
            )

        if not self.config.disable_l2:
            # skip some steps for the hierarchy
            # input_states = input_states[:: self.config.step_skip]
            # actions = actions[: self.config.step_skip * (input_states.shape[0] - 1)]
            forward_result_l1_for_l2 = self.level1.forward_posterior(
                l2_states,
                actions=None,
                proprio_pos=l2_proprio_pos,
                proprio_vel=l2_proprio_vel,
                chunked_locations=chunked_locations,
                chunked_proprio_pos=chunked_proprio_pos,
                chunked_proprio_vel=chunked_proprio_vel,
                encode_only=True,
            )

            # For L2, use obs_component from L1 if L2 uses proprio (to separate obs from proprio),
            # otherwise use encodings (for wall)
            if self.level2.backbone.using_proprio:
                encodings = forward_result_l1_for_l2.backbone_output.obs_component
            else:
                encodings = forward_result_l1_for_l2.backbone_output.encodings

            l2_kwargs = {"actions": l2_actions, "goal": goal}
            if self.level2.backbone.using_proprio:
                # L1's proprio component (, 2, 35, 35), L2 will downsample to (, 2, 16, 16)
                l1_proprio = forward_result_l1_for_l2.backbone_output.proprio_component
                l1_obs = forward_result_l1_for_l2.backbone_output.obs_component
                forward_result_l2 = self.level2.forward_posterior(l1_obs, proprio=l1_proprio, **l2_kwargs)
            else:
                forward_result_l2 = self.level2.forward_posterior(encodings, **l2_kwargs)

        else:
            forward_result_l2 = None

        return ForwardResult(level1=forward_result_l1, level2=forward_result_l2)

    def subsampling_ratio(self):
        return self.level2.subsampling_ratio()

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict() to add a post-call hook.
        """
        res = super().load_state_dict(state_dict, strict)  # Call original method
        self.post_load_hook()  # Run custom post-load logic
        return res

    def post_load_hook(self):
        # vmap the predictor if its an ensemble
        for model in [self.level1, self.level2]:
            if model is None:
                continue

            model.post_load_hook()
