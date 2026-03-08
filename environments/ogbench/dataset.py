from typing import NamedTuple, Optional

import torch
import numpy as np

from environments.diverse_maze.enums import D4RLDatasetConfig


class LocoMazeSample(NamedTuple):
    states: torch.Tensor  # [(batch_size), T, D] or [(batch_size), T, C, H, W]
    locations: torch.Tensor  # [(batch_size), T, 2]
    actions: torch.Tensor  # [(batch_size), T-1, 2]
    indices: torch.Tensor  # [(batch_size)] needed for prioritized replay
    proprio_pos: torch.Tensor  # [batch_size, T, D]
    proprio_vel: torch.Tensor  # [batch_size, T, D]
    view_states: torch.Tensor  # [bs, T, C, H, W] needed for visualization
    directions: torch.Tensor  # [batch_size, T-1, 2]
    chunked_locations: Optional[torch.Tensor] = None  # [batch_size, T, chunk_size, 2]
    chunked_proprio_pos: Optional[torch.Tensor] = None  # [batch_size, T, chunk_size, D]
    chunked_proprio_vel: Optional[torch.Tensor] = None  # [batch_size, T, chunk_size, D]


class LocoMazeDataset(torch.utils.data.Dataset):
    def __init__(self, config: D4RLDatasetConfig, images_tensor=None, suffix=""):
        self.config = config
        self._prepare_saved_ds()

    def _prepare_saved_ds(self):
        assert self.config.path is not None
        print("loading saved dataset from", self.config.path)

        metadata = torch.load(f"{self.config.path}/metadata.pt")
        self.episode_length = metadata["episode_length"]
        self.num_episodes = metadata["num_episodes"]

        self.qvel = np.load(f"{self.config.path}/qvel.npy")
        self.qpos = np.load(f"{self.config.path}/qpos.npy")
        self.actions = np.load(f"{self.config.path}/actions.npy")

        if self.config.chunked_actions:
            # Samples, chunk_size, D
            assert len(self.actions.shape) == 3
            assert len(self.qvel.shape) == 3
            assert len(self.qpos.shape) == 3
        else:
            # Samples, D
            assert len(self.qvel.shape) == 2
            assert len(self.qpos.shape) == 2
            assert len(self.actions.shape) == 2

        self.directions = np.load(f"{self.config.path}/direction.npy")

        if self.config.image_based:
            self.images_tensor = np.load(
                f"{self.config.path}/observations.npy", mmap_mode="r"
            )
        else:
            self.images_tensor = None

        if self.config.load_top_down_view:
            self.top_down_view_tensor = np.load(
                f"{self.config.path}/top_down_observations.npy", mmap_mode="r"
            )

        self.cum_lengths = np.cumsum(
            [
                self.episode_length
                - self.config.n_steps
                - (self.config.stack_states - 1)
                for x in range(self.num_episodes)
            ]
        )

        self.cum_lengths_total = np.cumsum(
            [self.episode_length for x in range(self.num_episodes)]
        )

    def __len__(self):
        if self.config.crop_length is not None:
            return min(self.config.crop_length, self.cum_lengths[-1])
        else:
            return self.cum_lengths[-1]

    def _load_images_tensor(self, index, length, top_down=False):
        images_tensor = self.top_down_view_tensor if top_down else self.images_tensor

        return (
            torch.from_numpy(images_tensor[index : index + length])
            .permute(0, 3, 1, 2)
            .float()
        )

    def _load_qpos(self, index, length):
        return torch.from_numpy(self.qpos[index : index + length]).float()

    def _load_qvel(self, index, length):
        return torch.from_numpy(self.qvel[index : index + length]).float()

    def _load_directions(self, index, length):
        return torch.from_numpy(self.directions[index : index + length]).float()

    def _load_actions(self, index, length):
        return torch.from_numpy(self.actions[index : index + length]).float()

    def __getitem__(self, idx):
        episode_idx = np.searchsorted(self.cum_lengths, idx, side="right")
        start_idx = idx - self.cum_lengths[episode_idx - 1] if episode_idx > 0 else idx

        if episode_idx == 0:
            global_start_idx = start_idx
        else:
            global_start_idx = self.cum_lengths_total[episode_idx - 1] + start_idx

        if self.config.load_top_down_view:
            top_down_view_states = self._load_images_tensor(
                index=global_start_idx + (self.config.stack_states - 1),
                length=self.config.n_steps,
                top_down=True,
            )
        else:
            top_down_view_states = torch.empty(0)

        qpos = self._load_qpos(
            index=global_start_idx + +(self.config.stack_states - 1),
            length=self.config.n_steps,
        )

        locations = qpos[..., :2]
        proprio_pos = qpos[..., 2:]

        proprio_vel = self._load_qvel(
            index=global_start_idx + +(self.config.stack_states - 1),
            length=self.config.n_steps,
        )

        actions = self._load_actions(
            index=global_start_idx + +(self.config.stack_states - 1),
            length=self.config.n_steps - 1,
        )

        if self.config.chunked_actions:
            # get the head of every chunk
            chunked_locations = locations
            chunked_proprio_pos = proprio_pos
            chunked_proprio_vel = proprio_vel

            locations = locations[:, 0, :]
            proprio_pos = proprio_pos[:, 0, :]
            proprio_vel = proprio_vel[:, 0, :]
        else:
            chunked_locations = torch.empty(0)
            chunked_proprio_pos = torch.empty(0)
            chunked_proprio_vel = torch.empty(0)

        directions = self._load_directions(
            index=global_start_idx + +(self.config.stack_states - 1),
            length=self.config.n_steps - 1,
        )

        if self.config.substitute_action == "direction":
            actions = directions

        if self.config.image_based:
            states = self._load_images_tensor(
                index=global_start_idx,
                length=self.config.n_steps + self.config.stack_states - 1,
            )
        else:
            # it's just the global position
            states = locations

        if self.config.stack_states > 1:
            states = torch.stack(
                [
                    states[i : i + self.config.stack_states]
                    for i in range(self.config.n_steps)
                ],
                dim=0,
            )
            states = states.flatten(1, 2)  # (sample_length, stack_states * state_dim)

        if self.config.random_actions:
            # uniformly sample values from -1 to 1
            actions = torch.rand_like(actions) * 2 - 1

        sample = LocoMazeSample(
            states=states,
            locations=locations,
            actions=actions,
            indices=idx,
            proprio_pos=proprio_pos,
            proprio_vel=proprio_vel,
            view_states=top_down_view_states,
            directions=directions,
            chunked_locations=chunked_locations,
            chunked_proprio_pos=chunked_proprio_pos,
            chunked_proprio_vel=chunked_proprio_vel,
        )

        return sample
