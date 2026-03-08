import torch
from environments.diverse_maze.utils import load_uniform
import environments.diverse_maze.ant_draw as ant_draw
from environments.utils.normalizer import Normalizer
from environments.diverse_maze.utils import sample_nearby_grid_location_ij
from environments.diverse_maze.evaluation.envs_generator import EnvsGenerator

from environments.diverse_ant.wrappers import DiverseAntNormEvalWrapper

import numpy as np


class AntEnvsGenerator(EnvsGenerator):
    def __init__(
        self,
        env_name: str,
        n_envs: int,
        min_block_radius: int,
        max_block_radius: int,
        seed: int = 42,
        stack_states: int = 1,
        image_obs: bool = True,
        data_path: str = None,
        trials_path: str = None,
        unique_shortest_path: bool = False,
        normalizer: Normalizer = None,
    ):
        super().__init__(
            env_name=env_name,
            n_envs=n_envs,
            min_block_radius=min_block_radius,
            max_block_radius=max_block_radius,
            seed=seed,
            stack_states=stack_states,
            image_obs=image_obs,
            data_path=data_path,
            trials_path=trials_path,
            unique_shortest_path=unique_shortest_path,
            normalizer=normalizer,
        )

    def _make_env(
        self,
        start,
        min_block_radius,
        max_block_radius,
        map_idx=None,
        map_key=None,
        ood_dist=None,
        mode=None,
        target=None,
        block_dist=None,
        turns=None,
        observations=None,
        actions=None,
    ):

        env_name = f"{self.env_name}_{map_idx}"
        env = ant_draw.load_environment(
            name=env_name,
            map_key=map_key,
            max_episode_steps=self.metadata["episode_length"],
        )

        if observations is not None:
            # we construct start and target from the trajectory

            start_obs = observations[0]
            goal_obs = observations[-1]
            start_xy = start_obs[:2]
            goal_xy = goal_obs[:2]

            options = {
                "reset_exact": True,
                "start_qpos": start_obs[:15],
                "start_qvel": start_obs[15:],
                "goal_qpos": goal_obs[:15],
                "goal_qvel": goal_obs[15:],
                "render_goal": True,
            }

        else:

            start_xy = start
            if target is None:
                start_ij = env.unwrapped.xy_to_ij(start_xy)

                goal_ij, block_dist, turns, unique_path = (
                    sample_nearby_grid_location_ij(
                        anchor=start_ij,
                        map_key=map_key,
                        min_block_radius=min_block_radius,
                        max_block_radius=max_block_radius,
                    )
                )

                goal_xy = env.unwrapped.ij_to_xy(goal_ij)
                goal_xy = env.unwrapped.add_noise(goal_xy)
            else:
                goal_xy = target

            options = {
                "init_xy": start_xy,
                "goal_xy": goal_xy,
                "render_goal": True,
            }

        env.reset(options=options)

        env = DiverseAntNormEvalWrapper(
            env=env,
            normalizer=self.normalizer,
            stack_states=self.stack_states,
            image_size=self.metadata["img_size"],
        )

        env.start_xy = start_xy
        env.trajectory = observations

        return env, goal_xy, block_dist, turns
