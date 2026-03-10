from dataclasses import dataclass
from typing import Optional, NamedTuple

import torch

from pldm.configs import ConfigBase
from pldm.models.encoders.enums import BackboneConfig, BackboneOutput
from pldm.models.predictors.enums import PredictorOutput, PredictorConfig
from functools import reduce
import operator
from pldm.models.encoders.encoders import build_backbone
from pldm.models.predictors.predictors import build_predictor


@dataclass
class JEPAConfig(ConfigBase):
    backbone: BackboneConfig = BackboneConfig()
    predictor: PredictorConfig = PredictorConfig()

    action_dim: int = 2

    # whether to use the trajectory goal as the latent variable.
    use_z_goal: bool = False
    encode_only: bool = False

    def __post_init__(self):
        self.predictor.prefused_repr = (
            not self.backbone.late_proprio_cfg.ignore
            and self.backbone.late_proprio_cfg.fuse
        )


class ForwardResult(NamedTuple):
    backbone_output: BackboneOutput
    pred_output: PredictorOutput
    actions: torch.Tensor


class JEPA(torch.nn.Module):
    """Joint-Embedding Predictive Architecture
    Includes an image encoder and a predictor.
    """

    def __init__(
        self,
        config: JEPAConfig,
        input_dim,
        input_obs_dim=None,
        input_proprio_dim=None,
        l1_action_dim: Optional[int] = None,
        step_skip: Optional[int] = None,
        l2: bool = False,
        ppos_dim=0,
        pvel_dim=0,
        loc_dim=0,
        normalizer=None,
    ):
        super().__init__()
        self.config = config
        self.step_skip = step_skip
        # self.use_proprio_pos = use_proprio_pos
        # self.use_proprio_vel = use_proprio_vel

        self.ppos_dim = ppos_dim
        self.pvel_dim = pvel_dim
        self.loc_dim = loc_dim

        if input_proprio_dim is None:
            input_proprio_dim = ppos_dim + pvel_dim

        self.backbone = build_backbone(
            config.backbone,
            input_dim=input_dim,
            input_obs_dim=input_obs_dim,
            input_proprio_dim=input_proprio_dim,
            input_loc_dim=loc_dim,
            l2=l2,
            normalizer=normalizer,
        )

        self.l2 = l2

        self.spatial_repr_dim = self.backbone.output_dim

        if isinstance(self.spatial_repr_dim, tuple):
            self.repr_dim = reduce(operator.mul, self.spatial_repr_dim)
        else:
            self.repr_dim = self.spatial_repr_dim

        if l2:
            if config.predictor.posterior_input_type == "term_states":
                config.predictor.posterior_input_dim = self.repr_dim * 2
            elif config.predictor.posterior_input_type == "actions":
                config.predictor.posterior_input_dim = l1_action_dim * step_skip
            else:
                raise NotImplementedError

        # right now we don't support hybrid true action + latent action
        assert (self.config.action_dim or self.config.predictor.z_dim) and not (
            self.config.action_dim and self.config.predictor.z_dim
        )

        predictor_input_dim = self.config.action_dim + self.config.predictor.z_dim

        self.config.predictor.rnn_state_dim = self.repr_dim

        # predictor.tie_backbone_ln ==> backbone.final_ln
        assert not config.predictor.tie_backbone_ln or config.backbone.final_ln

        self.predictor = build_predictor(
            config.predictor,
            repr_dim=self.backbone.output_dim,
            action_dim=predictor_input_dim,
            pred_proprio_dim=self.backbone.output_proprio_dim,
            pred_loc_dim=self.backbone.output_loc_dim,
            pred_obs_dim=self.backbone.output_obs_dim,
            backbone_ln=self.backbone.final_ln if config.backbone.final_ln else None,
            normalizer=normalizer,
        )

        self.using_proprio_pos = self.backbone.using_proprio and self.ppos_dim
        self.using_proprio_vel = self.backbone.using_proprio and self.pvel_dim
        self.using_proprio = self.backbone.using_proprio
        self.using_location = self.backbone.using_location

    def subsampling_ratio(self):
        if self.l2 and self.step_skip:
            return self.step_skip
        else:
            return 1

    def forward_prior(
        self,
        input_states: torch.Tensor,
        repr_input: bool = False,
        actions: Optional[torch.Tensor] = None,
        proprio_pos: Optional[torch.Tensor] = None,
        proprio_vel: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        T: Optional[int] = None,
    ):
        # If latents is not None, we use it instead of sampling from prior.

        # Input states are either images or encoded representations.
        # input_states is of shape BxD or BxCxHxW
        # actions is of shape TxBxA

        assert (
            actions is not None
            or T is not None
            or latents is not None
            or goal is not None
        )

        if repr_input:
            current_state = input_states
        else:
            current_state = self.backbone.forward_multiple(input_states).encodings

        if T is None:
            if actions is not None:
                T = actions.shape[0]
            elif latents is not None:
                T = latents.shape[0]
            else:
                raise ValueError("T is None but actions and latents are None")

        pred_output = self.predictor.forward_multiple(
            current_state.unsqueeze(0),
            actions,
            T,
            latents=latents,
        )

        return ForwardResult(
            backbone_output=None,
            pred_output=pred_output,
            actions=actions,
        )

    def forward_posterior(
        self,
        input_states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        proprio_pos: Optional[torch.Tensor] = None,
        proprio_vel: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        locations: Optional[torch.Tensor] = None,
        chunked_locations: Optional[torch.Tensor] = None,
        chunked_proprio_pos: Optional[torch.Tensor] = None,
        chunked_proprio_vel: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
        encode_only: bool = False,
    ):
        """
        input_states:
            TxBxD or TxBxCxHxW
        actions:
            (T-1)xBxA
        """

        if input_states.shape[-1] != self.repr_dim:
            if self.backbone.using_proprio:
                if proprio is not None:
                    # L2 case: proprio is already spatial from L1, use directly
                    proprio_states = proprio
                else:
                    # L1 case: concatenate raw proprio_pos and proprio_vel
                    if proprio_pos is None or proprio_pos.numel() == 0:
                        if proprio_vel is None or proprio_vel.numel() == 0:
                            raise ValueError(
                                "backbone.using_proprio is True but both proprio_pos and proprio_vel are None/empty"
                            )
                        proprio_states = proprio_vel
                    elif proprio_vel is None or proprio_vel.numel() == 0:
                        proprio_states = proprio_pos
                    else:
                        proprio_states = torch.cat([proprio_pos, proprio_vel], dim=-1)

                backbone_output = self.backbone.forward_multiple(
                    input_states,
                    proprio=proprio_states,
                    locations=locations,
                )
            else:
                backbone_output = self.backbone.forward_multiple(input_states)

            state_encs = backbone_output.encodings
        else:
            state_encs = input_states  # might be problematic for l2

        if self.config.encode_only or encode_only:
            return ForwardResult(
                backbone_output=backbone_output,
                pred_output=None,
                actions=actions,
            )

        T = input_states.shape[0] - 1

        pred_output = self.predictor.forward_multiple(
            state_encs=state_encs,
            actions=actions,
            T=T,
            proprio=backbone_output.proprio_component,
            locations=backbone_output.location_component,
            raw_locations=backbone_output.raw_locations,
            compute_posterior=True,
        )

        return ForwardResult(
            backbone_output=backbone_output,
            pred_output=pred_output,
            actions=actions,
        )

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict() to add a post-call hook.
        """
        res = super().load_state_dict(state_dict, strict)  # Call original method
        self.post_load_hook()  # Run custom post-load logic
        return res

    def post_load_hook(self):
        if self.predictor.ensemble:
            print("Initializing vmap model")
            self.predictor.initialize_vmap_model()
