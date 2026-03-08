"""
This script dumps the trials of the OGBench environment into a path file.
"""

import torch
import ogbench.locomaze
import gymnasium
from tqdm import tqdm

env_name = "antmaze-u-v0"
n_envs = 30

save_path = "/vast/wz1232/ogbench/ant_maze_u_explore_1M_noise1"

easy_trials = {
    "init_xy": [],
    "goal_xy": [],
}

medium_trials = {
    "init_xy": [],
    "goal_xy": [],
}

hard_trials = {
    "init_xy": [],
    "goal_xy": [],
}


for i in tqdm(range(n_envs)):
    env = gymnasium.make(
        env_name,
        terminate_at_goal=False,
        max_episode_steps=9999,
    )

    num_tasks = len(env.unwrapped.task_infos)
    task_id = i % num_tasks + 1
    env_options = {"task_id": task_id}
    _, info = env.reset(options=env_options)

    difficulty = env.unwrapped.task_infos[task_id - 1]["difficulty"]

    if difficulty == "easy":
        easy_trials["init_xy"].append(torch.tensor(env.unwrapped.init_xy))
        easy_trials["goal_xy"].append(torch.tensor(env.unwrapped.cur_goal_xy))
    elif difficulty == "medium":
        medium_trials["init_xy"].append(torch.tensor(env.unwrapped.init_xy))
        medium_trials["goal_xy"].append(torch.tensor(env.unwrapped.cur_goal_xy))
    elif difficulty == "hard":
        hard_trials["init_xy"].append(torch.tensor(env.unwrapped.init_xy))
        hard_trials["goal_xy"].append(torch.tensor(env.unwrapped.cur_goal_xy))

easy_trials["init_xy"] = torch.stack(easy_trials["init_xy"])
easy_trials["goal_xy"] = torch.stack(easy_trials["goal_xy"])

medium_trials["init_xy"] = torch.stack(medium_trials["init_xy"])
medium_trials["goal_xy"] = torch.stack(medium_trials["goal_xy"])

hard_trials["init_xy"] = torch.stack(hard_trials["init_xy"])
hard_trials["goal_xy"] = torch.stack(hard_trials["goal_xy"])

torch.save(easy_trials, save_path + "/trials_easy.pt")
torch.save(medium_trials, save_path + "/trials_medium.pt")
torch.save(hard_trials, save_path + "/trials_hard.pt")
