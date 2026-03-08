from pldm.configs import ConfigBase
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class ProprioConfig(ConfigBase):
    ignore: bool = True
    encoder_arch: str = "id"
    fuse: bool = False
    final_ln: bool = False


@dataclass
class LocalPatchConfig(ConfigBase):
    ignore: bool = True
    patch_type: str = "dynamic"
    encoder_arch: str = "id"


@dataclass
class BackboneConfig(ConfigBase):
    arch: str = "menet6"
    backbone_subclass: str = "a"
    backbone_width_factor: int = 1
    backbone_mlp: Optional[str] = None  # mlp to slap on top of backbone
    backbone_norm: str = "batch_norm"
    backbone_pool: str = "avg_pool"
    backbone_final_fc: bool = True
    channels: int = 1
    input_dim: Optional[int] = None  # if it's none, we assume it's image.
    # proprio_dim: Optional[int] = 0
    # early_proprio_dim: int = 0
    # early_proprio_encoder_arch: Optional[str] = None
    # late_proprio_encoder_arch: Optional[str] = None
    # late_proprio_fuse: bool = True
    # late_loc_encoder_arch: Optional[str] = None
    # late_loc_fuse: bool = False

    early_proprio_cfg: ProprioConfig = ProprioConfig()
    late_proprio_cfg: ProprioConfig = ProprioConfig()
    early_location_cfg: ProprioConfig = ProprioConfig()
    late_location_cfg: ProprioConfig = ProprioConfig()

    local_patch_cfg: LocalPatchConfig = LocalPatchConfig()

    # local_patch: bool = False
    # local_patch_type: str = "dynamic"
    # local_patch_arch:str = "id"

    # late_fuse_proprio_encoder_arch: Optional[str] = None
    fc_output_dim: Optional[int] = None  # if it's none, it will be a spatial output
    final_ln: bool = False
    # early_fuse_proprio_encoder_arch: Optional[str] = None


class BackboneOutput:
    def __init__(
        self,
        encodings: torch.Tensor,
        obs_component: Optional[torch.Tensor] = None,
        proprio_component: Optional[torch.Tensor] = None,
        location_component: Optional[
            torch.Tensor
        ] = None,  # the representation of location
        raw_locations: Optional[torch.Tensor] = None,  # the raw location
    ):
        self.encodings = encodings
        self._obs_component = obs_component
        self.proprio_component = proprio_component
        self.location_component = location_component
        self.raw_locations = raw_locations

    @property
    def obs_component(self):
        return (
            self._obs_component if self._obs_component is not None else self.encodings
        )

    @obs_component.setter
    def obs_component(self, value: Optional[torch.Tensor]):
        self._obs_component = value
