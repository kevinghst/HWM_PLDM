from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from pldm.models.utils import *
from pldm.models.predictors.enums import *
from pldm.models.predictors.sequence_predictor import SequencePredictor
from pldm.models.misc import build_mlp, PROBER_CONV_LAYERS_CONFIG

ConvPredictorConfig = {
    "a": [(18, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 16, 3, 1, 1)],
    "b": [(18, 32, 5, 1, 2), (32, 32, 5, 1, 2), (32, 16, 5, 1, 2)],
    "c": [(18, 32, 7, 1, 3), (32, 32, 7, 1, 3), (32, 16, 7, 1, 3)],
    "a_proprio": [(20, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 18, 3, 1, 1)],
    "d4rl_b_p": [(20, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 18, 3, 1, 1)],
    "d4rl_c_p": [(20, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 24, 3, 1, 1)],
    # "d4rl_d_p": [(20, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 51, 3, 1, 1)],
    "d4rl_d_p": [(20, 32, 3, 1, 1), (32, 32, 3, 1, 1)],
    "d4rl_e_p": [(20, 64, 3, 1, 1), (64, 64, 3, 1, 1)],
    "l2_d4rl_b_p": [(42, 32, 3, 1, 1), (32, 32, 3, 1, 1), (32, 34, 3, 1, 1)],
    "l2_d4rl_c_p": [(42, 32, 5, 1, 2), (32, 32, 5, 1, 2), (32, 18, 5, 1, 2)],
    "l2_d4rl_e_p": [
        (42, 32, 5, 1, 2),
        (32, 32, 5, 1, 2),
        (32, 32, 5, 1, 2),
        (32, 32, 5, 1, 2),
        (32, 18, 5, 1, 2),
    ],
    "d4rl_b_p_l": [
        (20, 32, 3, 1, 1),
        (32, 32, 3, 1, 1),
        (32, 32, 3, 1, 1),
        (32, 32, 3, 1, 1),
        (32, 18, 3, 1, 1),
    ]
}


class LocalPredictor(SequencePredictor):
    def __init__(
        self,
        # parent inputs
        config,
        repr_dim,
        action_dim=2,  # action + z_dim
        pred_proprio_dim=0,
        pred_loc_dim=0,
        pred_obs_dim=0,
        # child inputs
        backbone_ln: Optional[torch.nn.Module] = None,
        normalizer=None,
    ):
        super(LocalPredictor, self).__init__(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_loc_dim=pred_loc_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
        )
        self.normalizer = normalizer
        if self.config.local_patch:
            self.patch_input_dim = repr_dim[0] * 3 * 3
        else:
            self.patch_input_dim = 0

    def _get_normed_xy(self, curr_raw_location):
        ant_xy_obs = self.normalizer.unnormalize_location(curr_raw_location)
        ant_xy_pixels = self.normalizer.pixel_mapper.obs_coord_to_pixel_coord(
            ant_xy_obs
        )
        ant_xy_normalized = ant_xy_pixels / self.normalizer.pixel_mapper.img_width
        return ant_xy_normalized

    def _extract_local_patch_fixed(self, current_state, curr_proprio):
        """
        Args:
            current_state: Tensor of shape (B, C, H, W) - conv feature map
            ant_xy_normalized: Tensor of shape (B, 2) - ant position in normalized image coordinates [0, 1]

        Returns:
            patch: Tensor of shape (B, C, 3, 3)
        """
        B, C, H, W = current_state.shape
        device = current_state.device

        ant_xy_normalized = self.init_xy

        # Convert normalized coordinates to pixel indices in feature map
        x_idx = (ant_xy_normalized[:, 0] * (W - 1)).round().long().clamp(1, W - 2)
        y_idx = (ant_xy_normalized[:, 1] * (H - 1)).round().long().clamp(1, H - 2)

        # Prepare output tensor
        patch = torch.zeros((B, C, 3, 3), device=device, dtype=current_state.dtype)

        for i in range(3):
            for j in range(3):
                # Offsets: -1, 0, 1
                dx, dy = j - 1, i - 1
                xi = (x_idx + dx).clamp(0, W - 1)
                yi = (y_idx + dy).clamp(0, H - 1)

                # Gather current_state: for batch b, at (yi[b], xi[b])
                batch_indices = torch.arange(B, device=device)
                patch[:, :, i, j] = current_state[batch_indices, :, yi, xi]

        return patch  # shape (B, C, 3, 3)

    def _extract_local_patch_dynamic(self, current_state, curr_raw_location):
        """
        Copied from ChatGPT. No QA yet.
        """
        B, C, H, W = current_state.shape

        ant_xy_normalized = self._get_normed_xy(curr_raw_location)

        # Normalize ant_xy to [-1, 1] range for grid_sample
        ant_xy_grid = ant_xy_normalized * 2 - 1  # (B, 2)

        # Generate 3x3 relative offsets (in [-1, 1] space of grid_sample)
        offsets = torch.tensor(
            [
                [-1, -1],
                [0, -1],
                [1, -1],
                [-1, 0],
                [0, 0],
                [1, 0],
                [-1, 1],
                [0, 1],
                [1, 1],
            ],
            dtype=current_state.dtype,
            device=current_state.device,
        )  # (9, 2)

        # Convert offsets from pixels to normalized feature space
        dx = 2.0 / (W - 1)
        dy = 2.0 / (H - 1)
        norm_offsets = offsets * torch.tensor([dx, dy], device=current_state.device)

        # Repeat offsets per batch and add to base positions
        grid = ant_xy_grid[:, None, :] + norm_offsets[None, :, :]  # (B, 9, 2)
        grid = grid.view(B, 3, 3, 2)  # (B, 3, 3, 2)

        # grid_sample expects (B, H_out, W_out, 2)
        patch = F.grid_sample(
            current_state,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (B, C, 3, 3)

        return patch


class MLPLocalPredictor(LocalPredictor):
    def __init__(
        self,
        # parent inputs
        config,
        repr_dim,
        action_dim=2,  # action + z_dim
        pred_proprio_dim=0,
        pred_loc_dim=0,
        pred_obs_dim=0,
        # child inputs
        num_groups=4,
        backbone_ln: Optional[torch.nn.Module] = None,
        normalizer=None,
    ):
        super(MLPLocalPredictor, self).__init__(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_loc_dim=pred_loc_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
            normalizer=normalizer,
        )

        input_dim = self.patch_input_dim + action_dim + pred_proprio_dim

        self.proprio_head = build_mlp(
            layers_dims="256-256",
            input_dim=proprio_input_dim,
            output_shape=pred_proprio_dim,
            norm="layer_norm" if config.predictor_ln else None,
            activation="mish",
            dropout=config.dropout,
        )


class ConvLocalPredictor(LocalPredictor):
    """ """

    def __init__(
        self,
        # parent inputs
        config,
        repr_dim,
        action_dim=2,  # action + z_dim
        pred_proprio_dim=0,
        pred_loc_dim=0,
        pred_obs_dim=0,
        # child inputs
        num_groups=4,
        backbone_ln: Optional[torch.nn.Module] = None,
        normalizer=None,
    ):
        super(ConvLocalPredictor, self).__init__(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_loc_dim=pred_loc_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
            normalizer=normalizer,
        )
        self.num_groups = num_groups

        input_dim = (repr_dim[0] + pred_proprio_dim + action_dim, *repr_dim[1:])

        self.shared_conv = build_conv(
            ConvPredictorConfig[config.predictor_subclass],
            input_dim=input_dim,
            group_factor=8,
            output_dim=repr_dim,
        )

        proprio_input_dim = self.patch_input_dim + action_dim + pred_proprio_dim

        self.proprio_head = build_mlp(
            layers_dims="256-256",
            input_dim=proprio_input_dim,
            output_shape=pred_proprio_dim,
            norm="layer_norm" if config.predictor_ln else None,
            activation="mish",
            dropout=config.dropout,
        )

        self.action_encoder = Expander2D(w=repr_dim[-2], h=repr_dim[-1])
        self.proprio_encoder = Expander2D(w=repr_dim[-2], h=repr_dim[-1])

        self.location_predictor = build_conv(
            PROBER_CONV_LAYERS_CONFIG["b"],
            input_dim=repr_dim,
            output_dim=2,
        )

    def _init_finial_ln(self):
        return nn.LayerNorm(self.pred_proprio_dim)

    def forward_and_format(self, current_state, curr_action, curr_proprio, **kwargs):
        """
        It is responsible for:
            distangle the observation and proprioception components from the state encodings
            formatting and output of the predictor.
        """

        timestep = kwargs["timestep"]
        location = kwargs["curr_location"]
        raw_location = kwargs["curr_raw_location"]

        obs, proprio, raw_location = self.forward(
            current_state,
            curr_action=curr_action,
            curr_proprio=curr_proprio,
            curr_location=location,
            curr_raw_location=raw_location,
            timestep=timestep,
        )

        out = SingleStepPredictorOutput(
            prediction=obs,
            obs_component=obs,
            proprio_component=proprio,
            raw_location=raw_location,
        )

        return out

    def forward(self, current_state, curr_action, curr_proprio, timestep, **kwargs):
        """
        Args:
            current_state: (bs, ch, h, w)
            curr_action: (bs, action_dim)
            curr_proprio: (bs, proprio_dim)
        """

        curr_raw_location = kwargs.get("curr_raw_location", None)

        if timestep == 0:
            self.init_xy = self._get_normed_xy(curr_raw_location)

        bs, _, h, w = current_state.shape
        curr_action_2d = self.action_encoder(curr_action)
        curr_proprio_2d = self.proprio_encoder(curr_proprio)
        x = torch.cat([current_state, curr_proprio_2d, curr_action_2d], dim=1)
        obs = self.shared_conv(x)

        if self.config.local_patch:
            if self.config.local_patch_type == "dynamic":
                local_patch = self._extract_local_patch_dynamic(
                    current_state, curr_raw_location
                )
            elif self.config.local_patch_type == "fixed":
                local_patch = self._extract_local_patch_fixed(
                    current_state, curr_proprio
                )
            else:
                raise ValueError(
                    "Invalid local patch type. Choose 'dynamic' or 'fixed'."
                )
            proprio_input = torch.cat(
                [curr_proprio, curr_action, local_patch.view(bs, -1)], dim=1
            )
        else:
            proprio_input = torch.cat([curr_proprio, curr_action], dim=1)

        proprio = self.final_ln(self.proprio_head(proprio_input))

        if self.config.residual:
            # only apply residual to obs
            obs = obs + current_state

        raw_location = self.location_predictor(obs)

        return obs, proprio, raw_location


class ConvDistangledPredictor(SequencePredictor):
    """
    Unlike ConvPredictor, ConvDistangledPredictor
    excepts NON-fused obs and proprio parts in its input
    and outputs NON-fused obs and proprio parts in its output
    """

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
        backbone_ln: Optional[torch.nn.Module] = None,
    ):
        super(ConvDistangledPredictor, self).__init__(
            config=config,
            repr_dim=repr_dim,
            action_dim=action_dim,
            pred_proprio_dim=pred_proprio_dim,
            pred_obs_dim=pred_obs_dim,
            backbone_ln=backbone_ln,
        )
        self.num_groups = num_groups

        input_dim = (repr_dim[0] + pred_proprio_dim + action_dim, *repr_dim[1:])

        self.shared_conv = build_conv(
            ConvPredictorConfig[config.predictor_subclass],
            input_dim=input_dim,
            group_factor=8,
            last_layer_act_norm=True,
        )

        hidden_channels = get_output_channels(self.shared_conv)

        self.visual_head = nn.Conv2d(
            hidden_channels,
            repr_dim[0],
            kernel_size=3,
            padding=1,
        )

        self.proprio_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # → (B, hidden_channels, 1, 1)
            nn.Flatten(),  # → (B, hidden_channels)
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, pred_proprio_dim),  # → (B, 27)
        )

        self.action_encoder = Expander2D(w=repr_dim[-2], h=repr_dim[-1])
        self.proprio_encoder = Expander2D(w=repr_dim[-2], h=repr_dim[-1])

    def _init_finial_ln(self):
        return nn.LayerNorm(self.pred_proprio_dim)

    def forward_and_format(self, current_state, curr_action, curr_proprio, **kwargs):
        """
        It is responsible for:
            distangle the observation and proprioception components from the state encodings
            formatting and output of the predictor.
        """

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

    def forward(self, current_state, curr_action, curr_proprio, **kwargs):
        """
        Args:
            current_state: (bs, ch, h, w)
            curr_action: (bs, action_dim)
            curr_proprio: (bs, proprio_dim)
        """
        bs, _, h, w = current_state.shape
        curr_action = self.action_encoder(curr_action)
        curr_proprio_2d = self.proprio_encoder(curr_proprio)
        x = torch.cat([current_state, curr_proprio_2d, curr_action], dim=1)
        x = self.shared_conv(x)

        obs = self.visual_head(x)
        proprio = self.final_ln(self.proprio_head(x))

        if self.config.residual:
            obs = obs + current_state
            proprio = proprio + curr_proprio

        return obs, proprio


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
        action_channels = action_dim
        if self.config.action_encoder_arch and self.config.action_encoder_arch != "id":
            action_channels = int(self.config.action_encoder_arch.split("-")[-1])
        input_dim = (
            pred_obs_dim[0]
            + (
                pred_proprio_dim
                if isinstance(pred_proprio_dim, int)
                else pred_proprio_dim[0]
            )
            + action_channels,
            *pred_obs_dim[1:],
        )

        self.layers = build_conv(
            ConvPredictorConfig[config.predictor_subclass],
            input_dim=input_dim,
            group_factor=8,
        )

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
                Expander2D(w=pred_obs_dim[-2], h=pred_obs_dim[-1]),
            )
            # self.action_dim = layer_dims[-1]
        else:
            self.action_encoder = Expander2D(w=pred_obs_dim[-2], h=pred_obs_dim[-1])

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

    def forward(self, current_state, curr_action, curr_obs, curr_proprio, **kwargs):
        """
        Args:
            current_state: (bs, ch, h, w)
            curr_action: (bs, action_dim)
            curr_proprio: (bs, proprio_dim)
        """
        bs, _, h, w = current_state.shape
        curr_action = self.action_encoder(curr_action)
        if curr_action.dim() == 5:
            curr_action = curr_action.flatten(1, 2)
        x = torch.cat([current_state, curr_action], dim=1)
        x = self.layers(x)
        if self.config.residual:
            x = x + current_state

        return x
