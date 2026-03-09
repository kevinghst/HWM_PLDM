from typing import Optional, NamedTuple
from enum import Enum, auto
from dataclasses import dataclass
from pldm.configs import ConfigBase

from pldm_envs.diverse_maze.enums import D4RLDatasetConfig


class DatasetType(Enum):
    Single = auto()
    Multiple = auto()
    D4RL = auto()
    

class ProbingDatasets(NamedTuple):
    ds: DatasetType
    val_ds: DatasetType
    extra_datasets: dict = {}


class Datasets(NamedTuple):
    ds: DatasetType
    val_ds: DatasetType
    probing_datasets: Optional[ProbingDatasets] = None
    l2_probing_datasets: Optional[ProbingDatasets] = None


@dataclass
class DataConfig(ConfigBase):
    dataset_type: DatasetType = DatasetType.Single
    d4rl_config: D4RLDatasetConfig = D4RLDatasetConfig()

    normalize: bool = False
    min_max_normalize_state: bool = False
    normalizer_hardset: bool = False
    quick_debug: bool = False
    num_workers: int = 0
    prefetch_factor: int = 3