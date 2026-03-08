import dataclasses

import torch

from pldm_envs.wall.data.wall import WallDataset, WallDatasetConfig
import random


class WallPassingTestDataset(WallDataset):
    def __init__(self, config: WallDatasetConfig, *args, **kwargs):
        super().__init__(
            dataclasses.replace(
                config,
                size=config.batch_size,
                cross_wall_rate=0,
            ),
            *args,
            **kwargs,
        )

    def generate_state(self, wall_locs=None, door_locs=None):
        """
        Returns Tensor (bs, 2)
        Details the coordinates of starting points
        """
        # We leave 2 * self.std margin when generating state, and don't let the
        # dot approach the border.

        K = self.config.batch_size // 2
        y = torch.zeros(self.config.batch_size, device=door_locs.device)
        for i in range(len(door_locs)):
            if random.random() < 0.25:
                # with 25%, sample y position intersecting door
                y[i] = door_locs[i] - 2 + random.random() * 4
            else:
                # with 75%, sample y position intersecting wall
                if random.random() < 0.5:
                    y[i] = self.padding + random.random() * (
                        door_locs[i] - 2 - self.padding
                    )
                else:
                    y[i] = (
                        door_locs[i]
                        + 2
                        + random.random()
                        * (self.config.img_size - 1 - self.padding - (door_locs[i] + 2))
                    )

        x1 = torch.zeros(K, device=wall_locs.device)
        x2 = torch.zeros(K, device=wall_locs.device)

        for i in range(K):
            min_bound = max(self.padding, wall_locs[i] - self.config.n_steps + 2)
            max_bound = wall_locs[i] - 2

            x1[i] = min_bound + random.random() * (max_bound - min_bound)

        for i in range(K):
            min_bound = wall_locs[i + K] + 2
            max_bound = min(
                self.config.img_size - 1 - self.padding,
                wall_locs[i + K] + self.config.n_steps - 2,
            )

            x2[i] = min_bound + random.random() * (max_bound - min_bound)

        locations = torch.cat(
            [torch.stack([x1, y[:K]], dim=1), torch.stack([x2, y[K:]], dim=1)], dim=0
        )

        return locations

    def generate_actions(
        self,
        n_steps,
        bias_angle=None,
    ):
        # Note: generates n_steps - 1 actions

        # first half of actions go right, second half go left
        actions = torch.zeros(
            self.config.batch_size, n_steps - 1, 2, device=self.device
        )
        actions[: self.config.batch_size // 2, :, 0] = self.config.action_step_mean
        actions[self.config.batch_size // 2 :, :, 0] = -self.config.action_step_mean

        bias_angle = torch.zeros(self.config.batch_size, 2, device=self.device)
        bias_angle[: self.config.batch_size // 2, 0] = self.config.action_step_mean
        bias_angle[self.config.batch_size // 2 :, 0] = -self.config.action_step_mean

        return actions, bias_angle
