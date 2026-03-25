import torch
from torch import nn
from torch.nn import functional as F

from pldm.models.encoders.resnet import resnet18, resnet18ID
from pldm.models.encoders import resnet
from pldm.models.misc import (
    build_mlp,
    Projector,
    MLP,
    build_norm1d,
)
from pldm.models.utils import build_conv, Expander2D
from pldm.models.encoders.enums import BackboneConfig, BackboneOutput
from pldm.models.encoders.base_class import SequenceBackbone
from pldm.models.encoders.impala import ImpalaEncoder

ResNet18 = resnet18
ResNet18ID = resnet18ID

ENCODER_LAYERS_CONFIG = {
    # L1
    "a": [(2, 32, 5, 1, 0), (32, 32, 4, 2, 0), (32, 32, 3, 1, 0), (32, 16, 1, 1, 0)],
    "b": [(2, 16, 5, 1, 0), (16, 32, 4, 2, 0), (32, 32, 3, 1, 0), (32, 16, 1, 1, 0)],
    "c": [(2, 16, 5, 1, 0), (16, 16, 4, 2, 0), (16, 16, 3, 1, 0)],
    "f": [(2, 16, 5, 1, 0), (16, 16, 5, 2, 0), (16, 16, 5, 1, 2)],
    "g": [(2, 32, 5, 1, 0), (32, 32, 5, 2, 0), (32, 32, 5, 1, 2), (32, 16, 1, 1, 0)],
    "h": [(2, 16, 5, 1, 0), (16, 16, 5, 2, 0), (16, 16, 3, 1, 0)],
    "i": [(2, 16, 5, 1, 0), (16, 16, 5, 2, 0), (16, 16, 3, 1, 1)],
    "i_fc": [
        (2, 16, 5, 1, 0),
        (16, 16, 5, 2, 0),
        (16, 16, 3, 1, 1),
        ("fc", 13456, 512),
    ],
    "i_b": [(6, 16, 5, 1, 0), (16, 16, 5, 2, 0), (16, 16, 3, 1, 0)],
    "d4rl_a": [
        (6, 16, 5, 1, 0),
        (16, 32, 5, 2, 0),
        (32, 32, 3, 1, 0),
        (32, 32, 3, 1, 1),
        (32, 16, 1, 1, 0),
    ],
    "d4rl_a_l": [
        (6, 24, 5, 1, 0),
        (24, 48, 5, 2, 0),
        (48, 48, 3, 1, 0),
        (48, 48, 3, 1, 1),
        (48, 16, 1, 1, 0),
    ],
    "d4rl_a_xl": [
        (6, 32, 5, 1, 0),
        (32, 64, 5, 2, 0),
        (64, 64, 3, 1, 0),
        (64, 64, 3, 1, 1),
        (64, 16, 1, 1, 0),
    ],
    "d4rl_a_xxl": [
        (6, 32, 5, 1, 0),
        (32, 64, 5, 2, 0),
        (64, 96, 3, 1, 0),
        (96, 96, 3, 1, 1),
        (96, 64, 3, 1, 1),   # extra layer
        (64, 32, 1, 1, 0),
    ],
    "d4rl_a2": [
        (6, 16, 5, 1, 2),
        (16, 32, 5, 2, 2),
        (32, 32, 3, 1, 1),
        (32, 32, 3, 1, 1),
        (32, 16, 1, 1, 0),
    ],  # try # (16, 32, 32)
    "d4rl_b": [
        (6, 16, 5, 1, 2),
        (16, 32, 5, 2, 2),
        (32, 32, 3, 1, 1),
        (32, 24, 3, 2, 1),
    ],
    "d4rl_c": [
        (6, 16, 5, 1, 2),
        (16, 32, 5, 2, 2),
        (32, 32, 3, 1, 1),
        (32, 32, 3, 2, 1),
        (32, 24, 1, 1, 0),  # (24, 16, 16)
    ],
    "d4rl_c_l1": [
        (6, 16, 5, 1, 2),
        (16, 32, 5, 2, 2),
        (32, 32, 3, 1, 0),
        (32, 16, 3, 1, 0),  # (16, 28, 28)
    ],
    "l2_d4rl_b": [
        (32, 16, 1, 1, 0),
    ],
    "l2_d4rl_c": ["id"],
    "j": [(2, 32, 5, 1, 0), (32, 32, 5, 2, 0), (32, 32, 3, 1, 1), (32, 16, 1, 1, 0)],
    "k": [(2, 16, 5, 1, 0), (16, 32, 5, 2, 0), (32, 32, 3, 1, 1), (32, 16, 1, 1, 0)],
    # L2
    "d": [(16, 16, 3, 1, 0), (16, 16, 3, 1, 0)],
    "e": [
        ("pad", (0, 1, 0, 1)),
        (16, 16, 3, 1, 0),
        ("avg_pool", 2, 2, 0),
        (16, 16, 3, 1, 0),
    ],
    "l2b": [(16, 16, 3, 1, 1), (16, 16, 3, 2, 1), (16, 16, 3, 1, 1)],  # (8, 16, 15, 15)
}


class PassThrough(nn.Module):
    def forward(self, x):
        return x


class MLPNet(nn.Module):
    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, output_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        out = x.flatten(1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = BackboneOutput(encodings=out)
        return out


class MeNet6(SequenceBackbone):
    def __init__(
        self,
        config,
        input_dim: int,
        input_proprio_dim: int = 0,
        input_loc_dim: int = 0,
        normalizer=None,
    ):
        super().__init__()

        self.config = config
        self.normalizer = normalizer
        subclass = config.backbone_subclass
        layers_config = ENCODER_LAYERS_CONFIG[subclass]

        if "l2" in subclass and layers_config[0] != "id":
            # add prenormalization and relu layers?
            pre_conv = nn.Sequential(nn.GroupNorm(4, layers_config[0][0]), nn.ReLU())
        else:
            pre_conv = nn.Identity()

        input_dim = list(input_dim)

        # initiate proprio encoder
        self.early_proprio_encoder = None
        self.late_proprio_encoder = None
        self.late_loc_encoder = None

        early_proprio_cfg = config.early_proprio_cfg
        late_proprio_cfg = config.late_proprio_cfg
        early_location_cfg = config.early_location_cfg
        late_location_cfg = config.late_location_cfg

        self.using_proprio = not early_proprio_cfg.ignore or not late_proprio_cfg.ignore
        self.using_location = (
            not early_location_cfg.ignore or not late_location_cfg.ignore
        )

        if not early_proprio_cfg.ignore:
            self.early_proprio_encoder, _ = self._build_proprio_encoder(
                early_proprio_cfg.encoder_arch,
                w=input_dim[1],
                h=input_dim[2],
                proprio_dim=input_proprio_dim,
                fuse=True,  # we always use early proprio
            )

            # we always fuse the early proprio features with the obs input
            if early_proprio_cfg.encoder_arch == "id_expand":
                input_dim[0] += input_proprio_dim
            elif early_proprio_cfg.encoder_arch == "id":
                raise "must expand the proprio features prior to early fusion"
            else:
                input_dim[0] += early_proprio_cfg.encoder_arch.split("-")[-1]

        conv_layers = (
            build_conv(layers_config, (input_dim[0],))
            if not layers_config[0] == "id"
            else nn.Identity()
        )
        self.layers = nn.Sequential(pre_conv, conv_layers)

        proprio_out_dim = None

        self.final_ln = nn.ModuleDict(
            {
                "proprio": nn.Identity(),
                "location": nn.Identity(),
                "obs": nn.Identity(),
            }
        )

        if not late_proprio_cfg.ignore:
            self.late_proprio_encoder, proprio_out_dim = self._build_proprio_encoder(
                late_proprio_cfg.encoder_arch,
                input_dim=input_dim,
                proprio_dim=input_proprio_dim,
                fuse=late_proprio_cfg.fuse,
            )
            if late_proprio_cfg.final_ln:
                self.final_ln["proprio"] = torch.nn.LayerNorm(proprio_out_dim)

        if not late_location_cfg.ignore:
            self.late_loc_encoder, loc_out_dim = self._build_proprio_encoder(
                late_location_cfg.encoder_arch,
                input_dim=input_dim,
                proprio_dim=input_loc_dim,
                fuse=late_location_cfg.fuse,
            )
            if late_location_cfg.final_ln:
                self.final_ln["location"] = torch.nn.LayerNorm(loc_out_dim)

        if not config.local_patch_cfg.ignore:
            self.local_patch_encoder, patch_out_dim = self._build_proprio_encoder(
                config.local_patch_cfg.encoder_arch,
                input_dim=input_dim,
                proprio_dim=layers_config[-1][1] * 3 * 3,  # 3x3 patch
                fuse=False,
            )
            if config.final_ln:
                self.final_ln["obs"] = torch.nn.LayerNorm(patch_out_dim)

    def _build_proprio_encoder(self, arch: str, input_dim, proprio_dim, fuse: bool):
        # infer output dim of encoder
        sample_input = torch.randn(input_dim).unsqueeze(0)
        sample_output = self.layers(sample_input)
        encoder_output_dim = tuple(sample_output.shape[1:])

        w = encoder_output_dim[-2]
        h = encoder_output_dim[-1]

        # Check if proprio_dim is spatial (tuple) - for L2 using L1's processed proprio
        is_spatial_input = isinstance(proprio_dim, tuple)

        if is_spatial_input:
            # L2 case: proprio_dim is (channels, H, W) from L1, e.g., (2, 35, 35)
            # Need to downsample to match encoder output spatial dims (e.g., 16, 16)
            input_channels, input_h, input_w = proprio_dim
            output_h, output_w = h, w

            if arch == "id_expand" or (not arch or arch == "id"):
                if input_h == output_h and input_w == output_w:
                    proprio_encoder = nn.Identity()
                else:
                    scale_h = input_h / output_h
                    scale_w = input_w / output_w

                    if scale_h == scale_w and scale_h == int(scale_h):
                        stride = int(scale_h)
                        kernel_size = stride + 1 if stride > 1 else 3
                        padding = kernel_size // 2
                        proprio_encoder = nn.Sequential(
                            nn.Conv2d(
                                input_channels,
                                input_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                groups=input_channels,
                            ),
                            nn.GroupNorm(1, input_channels),
                            nn.ReLU(),
                        )
                    else:
                        proprio_encoder = nn.AdaptiveAvgPool2d((output_h, output_w))

                proprio_dim = (input_channels, output_h, output_w)
            else:
                raise NotImplementedError(
                    f"Spatial proprio input not supported with arch '{arch}'"
                )
        else:
            # L1 case: proprio_dim is scalar, expand to spatial
            if not arch or arch == "id":
                if fuse:
                    proprio_encoder = Expander2D(w=w, h=h)
                    proprio_dim = (proprio_dim, w, h)
                else:
                    proprio_encoder = nn.Identity()
            elif arch == "id_expand":
                proprio_encoder = Expander2D(w=w, h=h)
                proprio_dim = (proprio_dim, w, h)
            else:
                proprio_encoder = build_mlp(
                    layers_dims=arch,
                    input_dim=proprio_dim,
                    norm=self.config.backbone_norm,
                    activation="mish",
                )
                proprio_dim = int(arch.split("-")[-1])

                if fuse:
                    proprio_encoder = nn.Sequential(
                        *list(proprio_encoder),
                        Expander2D(w=w, h=h),
                    )
                    proprio_dim = (proprio_dim, w, h)

        return proprio_encoder, proprio_dim

    def forward(self, x, proprio=None, **kwargs):
        """
        torch.Size([bs, 2, 64, 64])
        torch.Size([bs, 32, 60, 60])
        torch.Size([bs, 32, 29, 29])
        torch.Size([bs, 32, 27, 27])
        torch.Size([bs, 16, 27, 27])
        """

        if self.early_proprio_encoder is not None:
            assert proprio is not None
            early_proprio = self.early_proprio_encoder(proprio)
            x = torch.cat([x, early_proprio], dim=1)

        locations = kwargs.get("locations", None)

        obs = self.layers(x)

        if not self.config.local_patch_cfg.ignore:
            obs = self._extract_local_patch(obs, locations)

        encodings = obs
        encodings = self.final_ln["obs"](encodings)

        late_proprio = None
        late_loc = None

        if self.late_proprio_encoder is not None:
            assert proprio is not None
            late_proprio = self.late_proprio_encoder(proprio)
            late_proprio = self.final_ln["proprio"](late_proprio)

            if self.config.late_proprio_cfg.fuse:
                encodings = torch.cat([encodings, late_proprio], dim=1)
                # late_proprio = None

        if self.late_loc_encoder is not None:
            assert locations is not None
            late_loc = self.late_loc_encoder(locations)
            late_loc = self.final_ln["location"](late_loc)

            if self.config.late_location_cfg.fuse:
                encodings = torch.cat([encodings, late_loc], dim=1)
                # late_loc = None

        output = BackboneOutput(
            encodings=encodings,
            obs_component=obs,
            proprio_component=late_proprio,
            location_component=late_loc,
            raw_locations=locations,
        )

        return output


class ResizeConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        scale_factor,
        mode="nearest",
        groups=1,
        bias=False,
        padding=1,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        x = BackboneOutput(encodings=x)
        return x


class MLPEncoder(SequenceBackbone):
    def __init__(self, config, input_dim, final_ln=False):
        super().__init__()
        self.encoder = MLP(
            arch=config.backbone_subclass,
            input_dim=input_dim,
            norm=config.backbone_norm,
        )
        out_dim = int(config.backbone_subclass.split("-")[-1])
        if config.final_ln:
            self.final_ln = build_norm1d(config.backbone_norm, out_dim)
        else:
            self.final_ln = nn.Identity()

    def forward(self, x, proprio=None, locations=None):
        x = self.encoder(x)
        x = self.final_ln(x)
        x = BackboneOutput(
            encodings=x,
            obs_component=x,
        )
        return x


class IdentityEncoder(SequenceBackbone):
    def __init__(self, config, input_dim, input_obs_dim, input_proprio_dim):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.input_obs_dim = input_obs_dim
        self.input_proprio_dim = input_proprio_dim

        early_proprio_cfg = config.early_proprio_cfg
        late_proprio_cfg = config.late_proprio_cfg
        self.using_proprio = not early_proprio_cfg.ignore or not late_proprio_cfg.ignore

    def forward(self, x, proprio=None, **kwargs):
        """
        x: torch.Size (bs, channels, w, h)
        """
        if proprio is None:
            raise ValueError("Proprio input is required for IdentityEncoder")

        obs_component = x
        proprio_component = proprio
        encodings = torch.cat([obs_component, proprio_component], dim=1) if proprio_component is not None else obs_component
        locations = kwargs.get("locations", None)

        x = BackboneOutput(
            encodings=encodings,
            obs_component=obs_component,
            proprio_component=proprio_component,
            raw_locations=locations,
        )
        
        return x


class FuseXYEncoder(SequenceBackbone):
    """
    encoder for features and proprio state
    (features, xy, pprio, pvel) --> (features, xy) --> features
    """

    def __init__(self, config, input_dim, input_obs_dim, input_proprio_dim):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.input_obs_dim = input_obs_dim
        self.input_proprio_dim = input_proprio_dim

        subclass = config.backbone_subclass
        layers_config = ENCODER_LAYERS_CONFIG[subclass]
        input_channels = input_obs_dim[0] + 2

        if "l2" in subclass:
            # add prenormalization and relu layers?
            self.pre_conv = nn.Sequential(
                nn.GroupNorm(4, layers_config[0][0]), nn.ReLU()
            )
        else:
            self.pre_conv = nn.Identity()

        # we know apriori that the first 2 proprio channels are x and y
        self.conv_layers = build_conv(layers_config, (input_channels,))

    def forward(self, x):
        """
        x: torch.Size (bs, channels, w, h)
        """

        obs_component = x[:, : self.input_obs_dim[0]]
        xy_component = x[:, self.input_obs_dim[0] : self.input_obs_dim[0] + 2]

        obs_component = self.pre_conv(obs_component)

        x = torch.cat([obs_component, xy_component], dim=1)

        x = self.conv_layers(x)

        x = BackboneOutput(encodings=x)

        return x


def build_backbone(
    config: BackboneConfig,
    input_dim,
    input_obs_dim,
    input_proprio_dim,
    input_loc_dim,
    l2: bool = False,
    normalizer=None,
):
    backbone, embedding = None, None
    arch = config.arch

    if arch == "resnet18" or "resnet18s" in arch:
        backbone, embedding = resnet.__dict__[arch](
            zero_init_residual=True,
            num_channels=config.channels,
            final_ln=config.final_ln,
        )
    elif arch == "resnet18ID":
        backbone, embedding = resnet.__dict__[arch](
            zero_init_residual=False, num_channels=config.channels
        )
    elif arch == "impala":
        backbone = ImpalaEncoder(
            input_channels=config.channels, final_ln=config.final_ln
        )
    elif arch == "id":
        backbone = PassThrough()
    elif arch == "menet6":
        backbone = MeNet6(
            config=config,
            input_dim=input_dim,
            input_proprio_dim=input_proprio_dim,
            input_loc_dim=input_loc_dim,
            normalizer=normalizer,
        )
    elif arch == "mlp":
        backbone = MLPEncoder(
            config=config,
            input_dim=input_dim,
            final_ln=config.final_ln,
        )
    elif l2:
        # Used for the second-level HJEPA.
        if arch == "identity":
            backbone = nn.Identity()
        elif arch == "identity_encoder":
            backbone = IdentityEncoder(
                config=config,
                input_dim=input_dim,
                input_obs_dim=input_obs_dim,
                input_proprio_dim=input_proprio_dim,
            )
        elif arch == "fuse_xy":
            backbone = FuseXYEncoder(
                config=config,
                input_dim=input_dim,
                input_obs_dim=input_obs_dim,
                input_proprio_dim=input_proprio_dim,
            )
        else:
            # We assume it's mlp with input that's not image.
            mlp_params = list(map(int, arch.split("-"))) if arch != "" else []
            backbone = build_mlp(
                [config.input_dim] + mlp_params + [config.fc_output_dim]
            )
    else:
        raise NotImplementedError(f"backbone arch {arch} is unknown")

    if config.backbone_mlp is not None:
        backbone_mlp = Projector(config.backbone_mlp, embedding)
        backbone = nn.Sequential(backbone, backbone_mlp)

    backbone.input_dim = input_dim
    sample_input = torch.randn(input_dim).unsqueeze(0)

    kwargs = {}
    if input_loc_dim:
        kwargs["locations"] = torch.randn(input_loc_dim).unsqueeze(0)

    if input_proprio_dim:
        sample_proprio_input = torch.randn(input_proprio_dim).unsqueeze(0)
        sample_output = backbone(sample_input, proprio=sample_proprio_input, **kwargs)
    else:
        sample_output = backbone(sample_input, **kwargs)

    output_dim = tuple(sample_output.encodings.shape[1:])
    output_dim = output_dim[0] if len(output_dim) == 1 else output_dim
    backbone.output_dim = output_dim

    if sample_output.proprio_component is not None:
        output_obs_dim = tuple(sample_output.obs_component.shape[1:])
        output_obs_dim = (
            output_obs_dim[0] if len(output_obs_dim) == 1 else output_obs_dim
        )
        output_proprio_dim = tuple(sample_output.proprio_component.shape[1:])
        output_proprio_dim = (
            output_proprio_dim[0]
            if len(output_proprio_dim) == 1
            else output_proprio_dim
        )
    else:
        output_obs_dim = output_dim
        output_proprio_dim = 0

    if sample_output.location_component is not None:
        output_loc_dim = tuple(sample_output.location_component.shape[1:])
        output_loc_dim = (
            output_loc_dim[0] if len(output_loc_dim) == 1 else output_loc_dim
        )
    else:
        output_loc_dim = 0

    backbone.output_obs_dim = output_obs_dim
    backbone.output_proprio_dim = output_proprio_dim
    backbone.output_loc_dim = output_loc_dim

    backbone.config = config

    return backbone
