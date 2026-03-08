from dataclasses import dataclass
from typing import NamedTuple
import torch
from pldm.planning.enums import MPCConfig
from pldm.planning.planners.enums import PlannerConfig


@dataclass
class D4RLMPCConfig(MPCConfig):
    position_only: bool = True
    subgoal_planning: bool = False
    set_full_states: bool = False
    unique_shortest_path: bool = (
        False  # gen trials where there's a unique shortest path
    )


@dataclass
class HierarchicalD4RLMPCConfig(D4RLMPCConfig):
    final_trans_norm_cutoff: float = 4
    final_trans_steps: int = 15
    error_threshold: float = 1.0
    mock_l1: bool = False
    level2: PlannerConfig = PlannerConfig()


class MPCReport(NamedTuple):
    success_rate: float
    success: torch.Tensor
    avg_steps_to_goal: float
    median_steps_to_goal: float
    terminations: list
    one_turn_success_rate: float
    two_turn_success_rate: float
    three_turn_success_rate: float
    num_one_turns: int
    num_two_turns: int
    num_three_turns: int
    num_turns: list
    block_dists: list
    ood_report: dict

    def build_log_dict(self, prefix=""):
        log_dict = {
            f"{prefix}_planning_success_rate": self.success_rate,
            f"{prefix}_avg_steps_to_goal": self.avg_steps_to_goal,
            f"{prefix}_median_steps_to_goal": self.median_steps_to_goal,
            f"{prefix}_one_turn_success_rate": self.one_turn_success_rate,
            f"{prefix}_two_turn_success_rate": self.two_turn_success_rate,
            f"{prefix}_three_turn_success_rate": self.three_turn_success_rate,
            f"{prefix}_num_one_turns": self.num_one_turns,
            f"{prefix}_num_two_turns": self.num_two_turns,
            f"{prefix}_num_three_turns": self.num_three_turns,
            # f"{prefix}_block_dist_p": self.block_dist_p,
            # f"{prefix}_num_turns_p": self.num_turns_p,
        }

        log_dict.update(self.ood_report)
        return log_dict
