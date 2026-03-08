from typing import Optional, NamedTuple
from enum import Enum, auto
from dataclasses import dataclass
from pldm.configs import ConfigBase

from pldm_envs.wall.data.offline_wall import OfflineWallDatasetConfig
from pldm_envs.wall.data.wall import WallDatasetConfig
from pldm_envs.wall.data.single import DotDatasetConfig
from pldm_envs.wall.data.wall_expert import WallExpertDatasetConfig

from pldm_envs.diverse_maze.enums import D4RLDatasetConfig

from pldm_envs.ogbench_manispace.dataset import ManispaceDatasetConfig


class DatasetType(Enum):
    Single = auto()
    Multiple = auto()
    Wall = auto()
    WallExpert = auto()
    WallEigenfunc = auto()
    D4RL = auto()
    D4RLEigf = auto()
    LocoMaze = auto()
    OgbenchManispace = auto()


class ProbingDatasets(NamedTuple):
    ds: DatasetType
    val_ds: DatasetType
    extra_datasets: dict = {}


class Datasets(NamedTuple):
    ds: DatasetType
    val_ds: DatasetType
    probing_datasets: Optional[ProbingDatasets] = None
    l2_probing_datasets: Optional[ProbingDatasets] = None
    aae_dataset: Optional[DatasetType] = None


@dataclass
class DataConfig(ConfigBase):
    dataset_type: DatasetType = DatasetType.Single
    dot_config: DotDatasetConfig = DotDatasetConfig()
    wall_config: WallDatasetConfig = WallDatasetConfig()
    offline_wall_config: OfflineWallDatasetConfig = OfflineWallDatasetConfig()
    wall_expert_config: WallExpertDatasetConfig = WallExpertDatasetConfig()

    # if "AMD" not in torch.cuda.get_device_name(0):
    d4rl_config: D4RLDatasetConfig = D4RLDatasetConfig()
    ogbench_manispace_config: ManispaceDatasetConfig = ManispaceDatasetConfig()

    normalize: bool = False
    min_max_normalize_state: bool = False
    normalizer_hardset: bool = False
    quick_debug: bool = False
    num_workers: int = 0
    prefetch_factor: int = 3