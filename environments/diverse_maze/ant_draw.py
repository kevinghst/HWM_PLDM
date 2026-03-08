from __future__ import annotations
from typing import *

import os
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import gym
import gym.spaces

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)
import random

from gymnasium.envs.registration import register as gymnasium_register
import gymnasium

from gym.envs.registration import register as gym_register


@contextmanager
def suppress_output():
    """
    A context manager that redirects stdout and stderr to devnull
    https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def create_custom_maze2d_env(name, maze_map, max_episode_steps: int):
    from d4rl.pointmaze import MazeEnv

    class CustomMazeEnv(MazeEnv):
        def __init__(self, **kwargs):
            # Call the __init__ method of MazeEnv with the custom maze layout
            super(CustomMazeEnv, self).__init__(maze_spec=maze_map, **kwargs)

        def step(self, action):
            # Call the original step method to get next state, reward, etc.
            next_state, original_reward, done, info = super().step(action)

            # Define a binary reward based on some condition
            # Example: reward is 1 if the agent reaches the goal, otherwise 0
            if self._is_goal_reached():  # You can define your own condition here
                reward = 1
            else:
                reward = 0

            return next_state, reward, done, info

        def _is_goal_reached(self, goal_dist_threshold=0.5):
            # Check if the agent has reached the goal
            # You can define how you determine if the goal is reached, e.g.,
            # by comparing the agent's current position with the goal position
            current_position = self._get_obs()[:2]  # Custom method to get position
            goal_position = self.get_target()  # Custom method to get goal
            distance_to_goal = np.linalg.norm(current_position - goal_position)

            # Example condition: reward is 1 if agent is within a certain distance of the goal
            return distance_to_goal < goal_dist_threshold  # Adjust threshold as needed

    gym_register(
        id=name,
        entry_point=lambda: CustomMazeEnv(),  # Change this to match your file name
        max_episode_steps=max_episode_steps,
    )

    with suppress_output():
        wrapped_env: gym.Wrapper = gym.make(name)

    env = cast(gym.Env, wrapped_env.unwrapped)

    return env


def create_custom_antmaze_env(name, maze_map, max_episode_steps: int, no_legs: bool):
    gymnasium_register(
        id=name,
        entry_point="ogbench.locomaze.maze:make_maze_env",
        max_episode_steps=1000,
        kwargs=dict(
            loco_env_type="ant",
            maze_env_type="maze",
            maze_map=maze_map,
            no_legs=no_legs,
        ),
    )

    env = gymnasium.make(
        name,
        terminate_at_goal=False,
        max_episode_steps=max_episode_steps,
    )

    # test
    # env.reset()
    # from PIL import Image
    # obs = env.render()
    # image = Image.fromarray(obs)
    # image.save("ant_render_ogbench_diverse.png")'

    return env


def convert_to_binary_array(map_key):
    # Create a 2D binary numpy array by iterating over the list of strings
    binary_array = np.array(
        [[1 if char == "#" else 0 for char in string] for string in map_key.split("\\")]
    )
    return binary_array


def load_environment(
    name: str,
    map_key: str = None,
    block_dist: int = None,
    turns: int = None,
    max_episode_steps: int = 600,
    no_legs: bool = True,
):  # NB: this removes the TimeLimit wrapper
    # create custom environment if layout is provided
    if map_key is not None:
        maze_map = convert_to_binary_array(map_key)

        if "ant" in name:
            maze_map_list = maze_map.tolist()

            env = create_custom_antmaze_env(
                name, maze_map_list, max_episode_steps, no_legs=no_legs
            )
        else:
            env = create_custom_maze2d_env(name, map_key, max_episode_steps)
    else:
        with suppress_output():
            wrapped_env: gym.Wrapper = gym.make(name)
            env = cast(gym.Env, wrapped_env.unwrapped)

    env.max_episode_steps = max_episode_steps
    env.name = name
    env.maze_map = maze_map
    env.map_key = map_key
    env.block_dist = block_dist
    env.turns = turns

    env.reset()

    env.step(
        env.action_space.sample()
    )  # sometimes stepping is needed to initialize internal
    env.reset()

    return env


def antmaze_fix_timeouts(env, dataset: Mapping[str, np.ndarray]):
    # https://gist.github.com/jannerm/d5ea90f17878b3fa198daf7dec67dfde#file-diffuser_antmaze-py-L1-L66

    logging.info("[ datasets/d4rl ] Fixing timeouts")
    N = len(dataset["observations"])
    max_episode_steps = np.where(dataset["timeouts"])[0][
        0
    ]  # usually 1000, sometimes 700

    ## 1000, 2001, 3002, 4003, ...
    timeouts = [max_episode_steps] + (
        np.arange(
            max_episode_steps + 1,
            N - max_episode_steps,
            max_episode_steps + 1,
        )
        + max_episode_steps
    ).tolist()
    timeouts = np.array(timeouts)

    timeouts_bool = np.zeros_like(dataset["timeouts"])
    timeouts_bool[timeouts] = 1

    assert np.all(timeouts_bool == dataset["timeouts"]), "sanity check"

    # dataset['timeouts'] = timeouts_bool
    dataset["terminals"][:] = 0

    fixed = {
        "observations": [],
        "actions": [],
        "next_observations": [],
        "rewards": [],
        "terminals": [],
        "timeouts": [],
        "steps": [],
    }
    step = 0
    for i in range(N - 1):
        done = dataset["terminals"][i] or dataset["timeouts"][i]

        if done:
            ## this is the last observation in its trajectory,
            ## cannot add a next_observation to this transition
            # print(i, step)
            step = 0
            continue

        for key in ["observations", "actions", "rewards", "terminals"]:
            val = dataset[key][i]
            fixed[key].append(val)

        next_observation = dataset["observations"][i + 1]
        fixed["next_observations"].append(next_observation)

        timeout = dataset["timeouts"][i + 1]
        fixed["timeouts"].append(timeout)

        fixed["steps"].append(step)

        step += 1

    fixed = {key: np.asarray(val) for key, val in fixed.items()}

    data_max_episode_steps = fixed["steps"].max() + 1
    logging.info(f"[ datasets/d4rl ] Max episode length: {max_episode_steps} (env)")
    logging.info(
        f"[ datasets/d4rl ] Max episode length: {data_max_episode_steps} (data)"
    )

    return fixed


def antmaze_scale_rewards(dataset, mult=10):
    # https://gist.github.com/jannerm/d5ea90f17878b3fa198daf7dec67dfde#file-diffuser_antmaze-py-L1-L66

    dataset["rewards"] = dataset["rewards"] * mult
    logging.info(f"[ datasets/d4rl ] Scaled rewards by {mult}")
    return dataset


def antmaze_get_dataset(env, reward_scale=1):
    dataset = env.get_dataset()
    if env.name.startswith("antmaze"):
        dataset = antmaze_fix_timeouts(env, dataset)
        dataset = antmaze_scale_rewards(dataset, reward_scale)
    return dataset


def set_state(env, state):
    if "ant" in env.name:
        qpos = np.copy(env.physics.data.qpos)
        qvel = np.copy(env.physics.data.qvel)
        qpos_slice = slice(0, 15)
        qvel_slice = slice(15, None)
    else:
        qpos = np.copy(env.sim.data.qpos)
        qvel = np.copy(env.sim.data.qvel)
        qpos_slice = slice(0, 2)
        qvel_slice = slice(2, None)

    qpos[qpos_slice] = state[qpos_slice]
    qvel[: len(state[qvel_slice])] = state[qvel_slice]

    env.set_state(qpos, qvel)
