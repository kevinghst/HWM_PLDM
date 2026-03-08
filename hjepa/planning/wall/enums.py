from dataclasses import dataclass
from typing import Optional, NamedTuple
import torch
from hjepa.planning.enums import MPCConfig
from hjepa.planning.planners.enums import PlannerConfig


@dataclass
class WallMPCConfig(MPCConfig):
    error_threshold: float = 1


@dataclass
class HierarchicalWallMPCConfig(WallMPCConfig):
    final_trans_norm_cutoff: float = 4
    final_trans_steps: int = 8


class MPCReport(NamedTuple):
    error_mean: torch.Tensor
    errors: torch.Tensor
    terminations: list
    planning_time: int
    cross_wall_rate: float
    init_plan_cross_wall_rate: float

    def build_log_dict(self, prefix: str = ""):
        return {
            f"{prefix}planning_error_mean": self.error_mean.item(),
            f"{prefix}planning_error_mean_rmse": self.error_mean.pow(0.5).item(),
            f"{prefix}success_rate": (self.errors < 1).float().mean().item(),
            f"{prefix}cross_wall_rate": self.cross_wall_rate,
            f"{prefix}init_plan_cross_wall_rate": self.init_plan_cross_wall_rate,
            f"{prefix}avg_termination_step": sum(self.terminations)
            / len(self.terminations),
        }


class HierarchicalMPCReport(NamedTuple):
    error_mean: torch.Tensor
    errors: torch.Tensor
    terminations: list
    planning_time: int
    cross_wall_rate: float
    init_plan_cross_wall_rate: float
    reach_target_plans_rate: float
    succ_plans_rate: float
    succ_illegal_plans_rate: float
    unsucc_trials_with_succ_plans: torch.Tensor
    unsucc_trials_with_succ_plans_rate: float
    succ_legal_plans_avg_dist: float
    succ_illegal_plans_avg_dist: float
    unsucc_plans_avg_dist: float
    succ_plans_avg_norm_diff: float
    succ_plans_avg_angle_diff: float
    unsucc_plans_avg_norm_diff: float
    unsucc_plans_avg_angle_diff: float
    norm_diff_p: float
    angle_diff_p: float

    def build_log_dict(self, prefix=""):
        return {
            f"{prefix}h_planning_error_mean": self.error_mean.item(),
            f"{prefix}h_planning_error_mean_rmse": self.error_mean.pow(0.5).item(),
            f"{prefix}h_success_rate": (self.errors < 1).float().mean().item(),
            f"{prefix}h_cross_wall_rate": self.cross_wall_rate,
            f"{prefix}h_init_plan_cross_wall_rate": self.init_plan_cross_wall_rate,
            f"{prefix}avg_termination_step": sum(self.terminations)
            / len(self.terminations),
            f"{prefix}reach_target_plans_rate": self.reach_target_plans_rate,
            f"{prefix}succ_plans_rate": self.succ_plans_rate,
            f"{prefix}succ_illegal_plans_rate": self.succ_illegal_plans_rate,
            f"{prefix}unsucc_trials_with_succ_plans_rate": self.unsucc_trials_with_succ_plans_rate,
            f"{prefix}succ_legal_plans_avg_dist": self.succ_legal_plans_avg_dist,
            f"{prefix}succ_illegal_plans_avg_dist": self.succ_illegal_plans_avg_dist,
            f"{prefix}unsucc_plans_avg_dist": self.unsucc_plans_avg_dist,
            f"{prefix}succ_plans_avg_norm_diff": self.succ_plans_avg_norm_diff,
            f"{prefix}succ_plans_avg_angle_diff": self.succ_plans_avg_angle_diff,
            f"{prefix}unsucc_plans_avg_norm_diff": self.unsucc_plans_avg_norm_diff,
            f"{prefix}unsucc_plans_avg_angle_diff": self.unsucc_plans_avg_angle_diff,
            f"{prefix}norm_diff_p": self.norm_diff_p,
            f"{prefix}angle_diff_p": self.angle_diff_p,
        }
