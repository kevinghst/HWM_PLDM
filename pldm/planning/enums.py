from dataclasses import dataclass, field
from pldm.configs import ConfigBase
import sys
from pldm.planning.planners.enums import PlannerConfig
from omegaconf import MISSING
from typing import Optional, NamedTuple, List
import torch


@dataclass
class LevelConfig(ConfigBase):
    n_steps: int = 30
    n_envs: int = 4
    max_plan_length: int = 10
    max_plan_length_l2: int = 10
    offline_T: int = 10  # timesteps between start and goal
    plot_every: int = 2
    override_config: bool = False
    set_start_target_path: Optional[str] = None
    # Maze specific below
    min_block_radius: int = -1
    max_block_radius: int = sys.maxsize
    level1: PlannerConfig = PlannerConfig()
    level2: PlannerConfig = PlannerConfig()


@dataclass
class MPCConfig(ConfigBase):
    seed: Optional[int] = 42
    env_name: str = MISSING
    n_envs: int = 4
    n_envs_batch_size: int = 4
    n_steps: int = 200  # Number of steps to run the env for
    visualize_planning: bool = True
    level1: PlannerConfig = PlannerConfig()
    plot_failure_only: bool = False
    log_pred_dist_every: int = sys.maxsize
    plot_every: int = 16
    data_path: str = ""  # predefined starts and goals
    action_repeat: int = 1
    action_repeat_mode: str = "null"
    img_size: int = 64
    replan_every: int = 1
    image_obs: bool = False
    stack_states: int = 1
    random_actions: bool = False
    # for choosing start, goal pairs from offline data
    offline_T: int = 10  # timesteps between start and goal
    start_target_from_data: bool = False
    set_start_target_path: Optional[str] = None
    levels: str = ""  # total levels to run
    level: str = "medium"  # level for this particular mpc eval run
    easy: LevelConfig = LevelConfig()
    medium: LevelConfig = LevelConfig()
    hard: LevelConfig = LevelConfig()

    # below used for hierarchy only:
    mock_l1: bool = False
    level2: PlannerConfig = PlannerConfig()


class MPCResult(NamedTuple):
    observations: List[torch.Tensor]
    locations: List[torch.Tensor]
    action_history: List[torch.Tensor]
    reward_history: List[torch.Tensor]
    pred_locations: List[torch.Tensor]
    final_preds_dist: List[torch.Tensor]
    targets: torch.Tensor
    loss_history: List[torch.Tensor]
    ensemble_var_history: List[torch.Tensor]
    ensemble_obs_var_history: List[torch.Tensor]
    ensemble_proprio_var_history: Optional[List[torch.Tensor]] = None
    qpos_history: Optional[List[torch.Tensor]] = None
    proprio_history: Optional[List[torch.Tensor]] = None
    # for hierarchy
    pred_locations_l2: Optional[List[torch.Tensor]] = None
    loss_history_l2: Optional[List[torch.Tensor]] = None
    success_history: Optional[List[torch.Tensor]] = None
    visual_observations: Optional[List[torch.Tensor]] = None
    visual_targets: Optional[torch.Tensor] = None


@dataclass
class PooledMPCResult:
    observations: list = field(default_factory=list)
    locations: list = field(default_factory=list)
    action_history: list = field(default_factory=list)
    reward_history: list = field(default_factory=list)
    pred_locations: list = field(default_factory=list)
    final_preds_dist: list = field(default_factory=list)
    targets: list = field(default_factory=list)
    loss_history: list = field(default_factory=list)
    extra: list = field(default_factory=list)
    qpos_history: list = field(default_factory=list)
    proprio_history: list = field(default_factory=list)
    ensemble_var_history: list = field(default_factory=list)
    ensemble_obs_var_history: list = field(default_factory=list)
    ensemble_proprio_var_history: list = field(default_factory=list)

    # for hierarchy
    pred_locations_l2: list = field(default_factory=list)
    loss_history_l2: list = field(default_factory=list)
    success_history: list = field(default_factory=list)
    visual_observations: list = field(default_factory=list)
    visual_targets: list = field(default_factory=list)

    def concatenate_chunks(self):
        # combine different chunks together in batch dimension
        self.targets = torch.cat(self.targets)

        self.observations = [torch.cat(t, dim=0) for t in zip(*self.observations)]
        self.locations = [torch.cat(t, dim=0) for t in zip(*self.locations)]
        self.action_history = [torch.cat(t, dim=0) for t in zip(*self.action_history)]
        self.reward_history = [torch.cat(t, dim=0) for t in zip(*self.reward_history)]

        self.pred_locations = [torch.cat(t, dim=1) for t in zip(*self.pred_locations)]
        self.final_preds_dist = [
            torch.cat(t, dim=1) for t in zip(*self.final_preds_dist)
        ]

        self.ensemble_var_history = [
            torch.cat(t, dim=1) for t in zip(*self.ensemble_var_history)
        ]

        self.ensemble_obs_var_history = [
            torch.cat(t, dim=1) for t in zip(*self.ensemble_obs_var_history)
        ]

        self.ensemble_proprio_var_history = [
            torch.cat(t, dim=1) for t in zip(*self.ensemble_proprio_var_history)
        ]

        # to prevent including residual chunk during averaging
        if len(self.loss_history) > 1:
            self.loss_history = self.loss_history[:-1]

        self.loss_history = [
            [sum(elements) / len(elements) for elements in zip(*sublists)]
            for sublists in zip(*self.loss_history)
        ]

        self.qpos_history = [torch.cat(t, dim=0) for t in zip(*self.qpos_history)]
        self.proprio_history = [torch.cat(t, dim=0) for t in zip(*self.proprio_history)]

        # for hierarchy
        self.pred_locations_l2 = [
            torch.cat(t, dim=1) for t in zip(*self.pred_locations_l2)
        ]
        self.loss_history_l2 = [
            [sum(elements) / len(elements) for elements in zip(*sublists)]
            for sublists in zip(*self.loss_history_l2)
        ]
        self.success_history = [torch.cat(t, dim=0) for t in zip(*self.success_history)]
        self.visual_observations = [
            torch.cat(t, dim=0) for t in zip(*self.visual_observations)
        ]
        self.visual_targets = (
            torch.cat(self.visual_targets) if self.visual_observations else None
        )
