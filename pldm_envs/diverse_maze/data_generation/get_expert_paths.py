"""
Used to run expert policy on a set of predefined trials.
"""

import torch
import numpy as np

import pldm_envs.diverse_maze.ant_draw as ant_draw
from pldm_envs.diverse_maze.wrappers import *


trials_path = "/vast/wz1232/maze2d_small_diverse/probe_train/start_target_planning.pt"

trials = torch.load(trials_path)


def make_env(start, target, map_idx=None, map_key=None, block_dist=None, turns=None):
    env = ant_draw.load_environment(
        name="maze2d_small_diverse", map_key=map_key, block_dist=block_dist, turns=turns
    )

    env = ActionRepeatWrapper(
        env,
        action_repeat=4,
        action_repeat_mode="id",
    )

    env = RenderImageWrapper(env, None)

    env = NavigationWrapper(env)

    env.reset()
    env.set_state(qpos=start[:2], qvel=np.array([0, 0]))
    env.set_target(target[:2])

    return env


n_trials = len(trials["map_layouts"])

# for i in range(n_trials):
# create env

i = 0
env = make_env(
    start=trials["starts"][i],
    target=trials["targets"][i],
    map_idx=i,
    map_key=trials["map_layouts"][i],
)

reward = 0
steps = 0
while not reward:
    print(f"Step {steps}")
    sub_goal = env.get_oracle_subgoal()
    curr_xy = env.unwrapped._get_obs()[:2]
    action = sub_goal - curr_xy
    action = action / (np.linalg.norm(action) + 1e-6)

    obs, reward, done, info = env.step(action)
    print(reward)
    steps += 1
