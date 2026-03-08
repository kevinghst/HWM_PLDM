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
import glob
import json

from scipy.stats import truncnorm
from environments.diverse_maze import ant_draw

from environments.utils.distributions import sample_tapered_distribution
from environments.utils.utils import sample_vector
from environments.diverse_maze.data_generation.data_generator import DataGenerator
from environments.ogbench.utils import PixelMapper, test_pixel_mapper
from PIL import Image

# TODO: make it less hacky
sys.path.insert(0, "/scratch/wz1232/ogbench/impls")
from utils.data_gen import gather_dataset
from torchvision import transforms
from PIL import Image
import imageio


class AntMazeDataGenerator(DataGenerator):
    def __init__(
        self,
        config,
        output_path,
        maps,
        exclude_maps,
    ):
        super().__init__(
            config=config,
            output_path=output_path,
            maps=maps,
            exclude_maps=exclude_maps,
        )

        self.actor_fn = None
        self.image_transform = transforms.Resize(tuple(self.config["img_size"][:2]))
        self.num_workers = 20

    def _resize_images(self, images_np):
        resized_images = []
        for i in range(images_np.shape[0]):
            img = Image.fromarray(images_np[i])
            resized_img = self.image_transform(img)
            resized_images.append(np.array(resized_img))
        return np.stack(resized_images)

    def _save_video(self, video_array, output_path, fps=15):
        """
        Params:
            traj: np.array: (N, H, W, 3)
        """
        imageio.mimsave(output_path, video_array, fps=fps)

    def estimate_state_ranges(self, env, num_samples=1000):
        """
        Estimate the minimum and maximum x, y positions in obs coords
        Output:
            min: (2,)
            max: (2,)
            obses: (num_samples, 2)
        """

        vals = []
        for _i in tqdm(range(num_samples), desc="Estimating state ranges"):
            ob, _ = env.reset()
            vals.append(ob[:2])

        vals = np.stack(vals)

        output = (
            np.min(vals, axis=0),
            np.max(vals, axis=0),
            vals,
        )

        return output

    def _load_expert(self, env):
        from agents import SACAgent
        from utils.flax_utils import restore_agent
        from utils.evaluation import supply_rng

        ob_dim = env.observation_space.shape[0]

        candidates = glob.glob(self.config["expert_restore_path"])
        assert len(candidates) == 1, f"Found {len(candidates)} candidates: {candidates}"

        with open(candidates[0] + "/flags.json", "r") as f:
            agent_config = json.load(f)["agent"]

        # Load agent.
        agent = SACAgent.create(
            self.config["seed"],
            np.zeros(ob_dim),
            env.action_space.sample(),
            agent_config,
        )
        agent = restore_agent(
            agent,
            self.config["expert_restore_path"],
            self.config["expert_restore_epoch"],
        )
        actor_fn = supply_rng(agent.sample_actions, rng=agent.rng)

        return actor_fn

    def gather_dataset(self, env):
        """
        Uses the ogbench function to get dataset
        Return:
        - all_obses_stacked: np.array of shape (n_episodes, episode_length, obs_dim)
        - all_actions_stacked: np.array of shape (n_episodes, episode_length - 1, action_dim)
        """
        if self.actor_fn is None:
            self.actor_fn = self._load_expert(env=env)

        dataset, _ = gather_dataset(
            env,
            self.actor_fn,
            num_episodes=self.n_episodes,
            dataset_type=self.dataset_type,
            sample_every=self.resample_every,
            noise=self.noise,
            collect_val=False,
            rtn_visual_obs=self.config["render"],
        )

        obs = np.stack(dataset["observations"])
        obs_dim = obs.shape[-1]
        obs = obs.reshape(self.n_episodes, self.episode_length, obs_dim)

        actions = np.stack(dataset["actions"])
        actions_dim = actions.shape[-1]
        actions = actions.reshape(self.n_episodes, self.episode_length, actions_dim)
        # remove last action of every episode
        actions = actions[:, :-1, :]

        if self.config["render"]:
            # resize visual observations
            obs_visual = np.stack(dataset["visual_obs"])
            resized_obs_visual = self._resize_images(obs_visual)
            img_shape = resized_obs_visual.shape[1:]
            resized_obs_visual = resized_obs_visual.reshape(
                self.n_episodes, self.episode_length, *img_shape
            )
        else:
            resized_obs_visual = None

        return obs, actions, resized_obs_visual

    def generate_data_for_env(self, env, map_idx=0):
        layout_dir = f"{self.output_path}/layout_{map_idx}"
        os.makedirs(layout_dir, exist_ok=True)

        _, _, samples = self.estimate_state_ranges(env)
        image = env.render()

        pixel_mapper = PixelMapper(
            env_name=env.name,
            img_width=image.shape[0],
            img_height=image.shape[1],
        )
        samples_xy = pixel_mapper.obs_coord_to_pixel_coord(samples, flip_coord=False)

        image = self._plot_points_over_image(image, samples_xy)

        image = Image.fromarray(image)
        image.save(f"{layout_dir}/reset_states_locations.png")

        obses, actions, visual_obs = self.gather_dataset(env)
        all_obses = obses.reshape(-1, obses.shape[-1])

        # plot all trajectories
        all_obses_xy = all_obses[:, :2]
        all_obses_xy = pixel_mapper.obs_coord_to_pixel_coord(
            all_obses_xy, flip_coord=False
        )

        image = env.render()
        image = self._plot_points_over_image(image, all_obses_xy)
        image = Image.fromarray(image)
        image.save(f"{layout_dir}/all_states_locations.png")

        # plot 5 sample trajectories
        for i in range(min(5, obses.shape[0])):
            obses_xy = obses[i, :, :2]
            obses_xy = pixel_mapper.obs_coord_to_pixel_coord(obses_xy, flip_coord=False)

            image = env.render()
            image = self._plot_points_over_image(image, obses_xy)
            image = Image.fromarray(image)
            image.save(f"{layout_dir}/sample_{i}_states_locations.png")

            # image.save(f"/scratch/wz1232/HJEPA/imgs/map_{map_idx}_sample_{i}_states_locations.png")

            # save a video for the entire trajectory
            if visual_obs is not None:
                self._save_video(
                    visual_obs[i], f"{layout_dir}/sample_{i}_trajectory.mp4"
                )

        if visual_obs is not None:
            splits = [
                {"actions": a, "observations": o, "visual_obs": vo, "map_idx": map_idx}
                for a, o, vo in zip(actions, obses, visual_obs)
            ]
        else:
            splits = [
                {"actions": a, "observations": o, "map_idx": map_idx}
                for a, o in zip(actions, obses)
            ]

        return splits
