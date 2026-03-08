from typing import NamedTuple
import torch


class MPCReport(NamedTuple):
    success_rate: float
    block_success_rate: float
    success: torch.Tensor
    avg_steps_to_goal: float
    median_steps_to_goal: float
    terminations: list
    avg_proportion_traveled: float

    def build_log_dict(self, prefix=""):
        return {
            f"{prefix}_planning_success_rate": self.success_rate,
            f"{prefix}_planning_block_success_rate": self.block_success_rate,
            f"{prefix}_avg_steps_to_goal": self.avg_steps_to_goal,
            f"{prefix}_median_steps_to_goal": self.median_steps_to_goal,
            f"{prefix}_avg_proportion_traveled": self.avg_proportion_traveled,
        }
