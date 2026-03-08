from __future__ import annotations
from typing import *

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import gym
import gym.spaces

from PIL import Image

import d4rl.pointmaze
from environments.diverse_maze.transforms import select_transforms
from environments.diverse_maze.data_generation.maze_stats import RENDER_STATS
from environments.diverse_maze.utils import PixelMapper
from torchvision import transforms


class AntDrawer:
    def __init__(self, env, env_id: str, process_size: int = 64):
        self.env = env
        self.env_id = env_id
        self.image_transform = transforms.Resize((process_size, process_size))

    def render_state(self, obs):
        """
        Params:
            obs: numpy array: (29,)
        """
        qpos = obs[:15]
        qvel = obs[15:]
        self.env.set_state(qpos=qpos, qvel=qvel)
        img = self.env.render()  # (200, 200, 3)
        img_pil = Image.fromarray(img)
        img_resized_pil = self.image_transform(img_pil)
        img_resized_np = np.array(img_resized_pil)
        return img_resized_np, img_resized_pil


class D4RLMaze2DDrawer(object):
    def __init__(
        self,
        env: d4rl.pointmaze.MazeEnv,
        *,
        lookat: List[float],
        arrow_mult: float,
    ):
        self.env = env
        self.env.render(mode="rgb_array")  # initialize viewer
        self.env.viewer.cam.elevation = -90
        self.env.viewer.cam.lookat[:] = np.asarray(lookat, dtype=np.float64)

        self.arrow_mult = arrow_mult

        self.pixel_mapper = PixelMapper(env_name=env.name)

    def render_state(self, obs: Any) -> np.ndarray:
        self.env.set_state(
            *[v.numpy() for v in torch.as_tensor(obs).data.cpu()[:4].chunk(2, dim=-1)]
        )
        return self.env.render("rgb_array")

    @torch.no_grad()
    def __call__(self, obs, ax: Union[None, plt.Axes] = None) -> plt.Axes:
        obs = torch.as_tensor(obs).data.cpu()[:4]
        rgb = self.render_state(obs)
        if ax is None:
            ax = plt.gca()
        ax.imshow(rgb)
        ax.autoscale(False)
        arrow = ax.arrow(
            *self.pixel_mapper.obs_coord_to_pixel_coord(obs[:2]),
            *obs[2:].mul(torch.tensor([1, -1])).mul(self.arrow_mult),
            width=6,
            color="red",
            alpha=0.6,
        )
        return ax


def render_umaze(env, obs, set_to_obs=True, normalizer=None):
    drawer = create_drawer(env, env.name)
    og_state = env.unwrapped._get_obs()

    transforms = select_transforms(drawer.env.name)
    if len(obs.shape) == 1:
        image = Image.fromarray(np.uint8(drawer.render_state(obs)))
        image = transforms(image)
        output = torch.from_numpy(np.array(image)).permute(2, 0, 1)
    else:
        images = []
        for o in obs:
            image = Image.fromarray(np.uint8(drawer.render_state(o)))
            image = transforms(image)
            images.append(torch.from_numpy(np.array(image)).permute(2, 0, 1))
        output = torch.stack(images)

    if not set_to_obs:
        if "ant" in drawer.env.name:
            env.set_state(qpos=og_state[:15], qvel=og_state[15:])
        else:
            env.set_state(qpos=og_state[:2], qvel=og_state[2:])

    if normalizer is not None:
        output = normalizer.normalize_state(output)

    return output


def create_drawer(env, env_id):
    if env is None:
        env = gym.make(env_id)
        env.name = env_id

    if "diverse" in env_id:
        key_word = "diverse"
        index = env_id.find(key_word)
        if index != -1:
            env_id = env_id[: index + len(key_word)]

    if "ant" in env_id:
        return AntDrawer(env, env_id)
    else:
        stats = RENDER_STATS[env_id]
        return D4RLMaze2DDrawer(
            env, lookat=stats["lookat"], arrow_mult=stats["arrow_mult"]
        )
