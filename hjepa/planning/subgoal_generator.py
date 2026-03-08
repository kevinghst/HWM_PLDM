from abc import ABC, abstractmethod

import numpy as np
import torch

from planning.planners.expert_planner_maze import ExpertPlanner


class SubgoalGenerator(ABC):
    @abstractmethod
    def generate_subgoals(self, pos, goal):
        pass


class ExpertMazeSubgoalGenerator(SubgoalGenerator):
    def __init__(self, env_name: str):
        self.planner = ExpertPlanner(env_name=env_name)

    def generate_subgoals(self, pos_v, goal_v):
        subgoals = []
        for pos, goal in zip(pos_v[:, :2], goal_v[:, :2]):
            subgoals.append(self.planner.plan_next_subgoal(pos, goal))
        subgoals = torch.from_numpy(np.stack(subgoals)).float()
        # subgoals are Nx2. We need to pad with 0 to make it Nx4
        subgoals = torch.cat([subgoals, torch.zeros_like(subgoals)], dim=1)
        return subgoals
