import os
import sys
import argparse
import yaml

from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
import random
import math

from environments.diverse_maze.data_generation.map_generator import MapGenerator
from environments.diverse_maze import ant_draw
from environments.diverse_maze.data_generation.wrappers import OGBenchWrapper

from environments.utils.utils import sample_vector
from PIL import Image


class DataGenerator:
    def __init__(
        self,
        config,
        output_path,
        maps=None,
        exclude_maps=None,
    ):
        self.output_path = output_path
        self.config = config

        self.n_episodes = self.config["n_episodes"]
        self.episode_length = self.config["episode_length"]
        self.margin = 0.5

        self.dataset_type = self.config.get("dataset_type", "explore")
        self.noise = self.config.get("noise", 0)
        self.resample_every = self.config.get("resample_every", 1)
        self.ogbench_gather = self.config.get("ogbench_gather", False)

        os.makedirs(self.output_path, exist_ok=True)
        # generate map layouts if in diverse setting
        if "diverse" in self.config["env"]:
            if maps is None:
                train_maps_n = self.config["train_maps_n"]
                map_generator = MapGenerator(
                    width=self.config["num_blocks_width_in_img"] - 2,
                    height=self.config["num_blocks_width_in_img"] - 2,
                    num_maps=train_maps_n,
                    sparsity_low=self.config.get("sparsity_low", 53),
                    sparsity_high=self.config.get("sparsity_high", 88),
                    max_path_len=self.config.get("max_path_len", 13),
                    exclude_maps=exclude_maps,
                    wall_coords=self.config.get("wall_coords", []),
                    space_coords=self.config.get("space_coords", []),
                )
                self.diverse_maps = map_generator.generate_diverse_maps()
                save_map_path = f"{self.output_path}/train_maps.pt"
                torch.save(self.diverse_maps, save_map_path)
            else:
                self.diverse_maps = maps
                n_maps = self.config["train_maps_n"]
                if len(self.diverse_maps) > n_maps:
                    self.diverse_maps = dict(list(self.diverse_maps.items())[:n_maps])
                    save_map_path = f"{self.output_path}/train_maps.pt"
                    torch.save(self.diverse_maps, save_map_path)

    def _plot_points_over_image(
        self,
        image: np.ndarray,
        samples: np.ndarray,
    ):
        """
        params:
            image: np.ndarray of shape (height, width, 3)
            samples: np.ndarray of shape (n_samples, 2)
        output:
            output: np.ndarray of shape (height, width, 3)
        """
        image = image.copy()
        height, width = image.shape[:2]
        valid_mask = (
            (samples[:, 0] >= 0)
            & (samples[:, 0] < width)
            & (samples[:, 1] >= 0)
            & (samples[:, 1] < height)
        )
        samples = samples[valid_mask]

        image[samples[:, 0], samples[:, 1]] = [255, 0, 0]

        return image

    def pick_random_start(self, env, range_min, range_max):
        if "ant" not in env.name:
            state = env.reset()

            if self.config["qvel_norm_prior"]:
                if self.config["qvel_norm_prior_type"] == "normal":
                    qvel = self.qvel_dist.rvs(size=2)
                elif self.config["qvel_norm_prior_type"] == "uniform":
                    qvel = sample_vector(max_norm=5)

                state[2:] = qvel

            return state

        state = np.zeros(29)

        while True:
            # generate random state in the range
            state[:2] = np.random.uniform(range_min, range_max, 2)
            if not env._is_in_collision(state[:2]):
                return state

    def generate_data(self):
        cfg = self.config
        n_episodes = self.n_episodes
        episode_length = self.episode_length

        print(f"Collecting {n_episodes} episodes for length {episode_length}")

        if "diverse" in cfg["env"]:
            splits = []
            # we are creating many envs - each with unique layout
            for map_idx, map_key in self.diverse_maps.items():
                env = ant_draw.load_environment(
                    name=f"{cfg['env']}_{map_idx}",
                    map_key=map_key,
                    max_episode_steps=episode_length,
                    no_legs=cfg.get("no_legs", False),
                )

                splits += self.generate_data_for_env(env, map_idx)
                env.close()
                del env
        else:
            env = ant_draw.load_environment(
                cfg["env"],
                max_episode_steps=episode_length,
                no_legs=cfg.get("no_legs", False),
            )
            splits = self.generate_data_for_env(env)

        output_file_name = f"{self.output_path}/data.p"
        print("Saving data to", output_file_name)
        torch.save(splits, output_file_name)

        output_metadata_name = f"{self.output_path}/metadata.pt"
        torch.save(self.config, output_metadata_name)
