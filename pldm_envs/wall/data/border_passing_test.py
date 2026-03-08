import dataclasses

import torch

from pldm_envs.wall.data.wall import WallDataset, WallDatasetConfig
import random


class BorderPassingTestDataset(WallDataset):
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

        First 25% points to left wall
        Second 25% points to right wall
        Third 25% points to top wall
        Fourth 25% points to bot wall

        """
        # We leave 2 * self.std margin when generating state, and don't let the
        # dot approach the border.

        K = self.config.batch_size
        states = torch.zeros((self.config.batch_size, 2), device=door_locs.device)

        for i in range(K):
            # first pretend we're always generating a state pointing towards left wall
            min_x = self.config.border_wall_loc - 1
            max_x = min(
                self.config.border_wall_loc + self.config.n_steps - 2,
                self.config.img_size - self.config.border_wall_loc,
            )

            x = random.random() * (max_x - min_x) + min_x

            min_y = self.config.border_wall_loc - 1
            max_y = self.config.img_size - self.config.border_wall_loc

            y = random.random() * (max_y - min_y) + min_y

            if K / 4 <= i and i < K / 2:
                # flip x to the right side
                x = self.config.img_size - 1 - x
            elif K / 2 < i and i < K * 3 / 4:
                # reverse x and y to point to top
                x, y = y, x
            elif K * 3 / 4 < i and i < K:
                # flip x to the right side
                # reverse x and y to point to bot
                x = self.config.img_size - 1 - x
                x, y = y, x

            states[i][0] = x
            states[i][1] = y

        return states

    def generate_actions(
        self,
        n_steps,
        bias_angle=None,
    ):
        """
        Returns Tensor (bs, n_steps - 1, 2)
        Details the action coordinates

        First 25% points to left wall
        Second 25% points to right wall
        Third 25% points to top wall
        Fourth 25% points to bot wall

        """
        # Note: generates n_steps - 1 actions

        K = self.config.batch_size

        actions = torch.zeros(K, n_steps - 1, 2, device=self.device)
        actions[: K // 4, :, 0] = -self.config.action_step_mean
        actions[K // 4 : K // 2, :, 0] = self.config.action_step_mean
        actions[K // 2 : K * 3 // 4, :, 1] = -self.config.action_step_mean
        actions[K * 3 // 4 : K, :, 1] = self.config.action_step_mean

        bias_angle = torch.zeros(K, 2, device=self.device)
        bias_angle[: K // 4, 0] = -self.config.action_step_mean
        bias_angle[K // 4 : K // 2, 0] = self.config.action_step_mean
        bias_angle[K // 2 : K * 3 // 4, 1] = -self.config.action_step_mean
        bias_angle[K * 3 // 4 : K, 1] = self.config.action_step_mean

        return actions, bias_angle
