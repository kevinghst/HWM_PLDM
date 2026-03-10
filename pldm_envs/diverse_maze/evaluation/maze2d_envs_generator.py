from pldm_envs.diverse_maze.data_generation.maze_stats import RENDER_STATS
import pldm_envs.diverse_maze.ant_draw as ant_draw
from pldm_envs.utils.normalizer import Normalizer
from pldm_envs.diverse_maze.wrappers import ActionRepeatWrapper, NormEvalWrapper
from pldm_envs.diverse_maze.evaluation.envs_generator import EnvsGenerator
from pldm_envs.diverse_maze.utils import sample_nearby_grid_location_v2

import numpy as np


class Maze2DEnvsGenerator(EnvsGenerator):
    def __init__(
        self,
        env_name: str,
        n_envs: int,
        min_block_radius: int,
        max_block_radius: int,
        action_repeat: int,
        action_repeat_mode: str,
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
        self.action_repeat = action_repeat
        self.action_repeat_mode = action_repeat_mode

    def _sample_nearby_location(
        self, anchor, map_key, min_block_radius=-1, max_block_radius=3
    ):
        """
        Given the anchor coordinate in obs space, and a particualar map_key (layout),
        sample a target coordinate within {block_radius} blocks of the anchor
        """

        # if "small" in self.env_name:
        obs_coord, block_dist, turns, unique_path = sample_nearby_grid_location_v2(
            anchor=anchor,
            map_key=map_key,
            min_block_radius=min_block_radius,
            max_block_radius=max_block_radius,
            obs_range_total=RENDER_STATS[self.env_name]["obs_range_total"],
            obs_min_total=RENDER_STATS[self.env_name]["obs_min_total"],
            unique_shortest_path=self.unique_shortest_path,
        )
        # else:
        #     raise CustomError("medium setting no longer supported")

        return obs_coord, block_dist, turns

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
        if target is None:
            target, block_dist, turns = self._sample_nearby_location(
                anchor=start[:2],
                map_key=map_key,
                min_block_radius=min_block_radius,
                max_block_radius=max_block_radius,
            )

        if map_idx is not None:
            env_name = f"{self.env_name}_{map_idx}"
            env = ant_draw.load_environment(
                name=env_name, map_key=map_key, block_dist=block_dist, turns=turns
            )
        else:
            env_name = self.env_name
            env = ant_draw.load_environment(
                env_name, block_dist=block_dist, turns=turns
            )

        if self.action_repeat > 1:
            env = ActionRepeatWrapper(
                env,
                action_repeat=self.action_repeat,
                action_repeat_mode=self.action_repeat_mode,
            )

        env = NormEvalWrapper(env, self.normalizer, stack_states=self.stack_states)

        env.reset()
        if "ant" in env.name:
            env.set_state(qpos=start[:15], qvel=start[15:])
        else:
            env.set_state(qpos=start[:2], qvel=np.array([0, 0]))

        env.set_target(target[:2])
        env.start_xy = start[:2]
        env.ood_dist = ood_dist
        env.mode = mode

        return env, target, block_dist, turns
