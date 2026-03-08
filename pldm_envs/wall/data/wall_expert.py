from dataclasses import dataclass
import dataclasses
import math
from typing import NamedTuple

import torch
from torch.utils.data import DataLoader

from pldm_envs.wall.data.wall import WallDataset, WallDatasetConfig
from pldm_envs.wall.data.single import DotDataset


@dataclass
class WallExpertDatasetConfig(WallDatasetConfig):
    expert_fraction: float = 0.5
    max_step_expert: float = 2.0
    num_workers: int = 16


class ExpertSample(NamedTuple):
    states: torch.Tensor  # [(batch_size), T, 1, 28, 28]
    locations: torch.Tensor  # [(batch_size), T, N_DOTS, 2]
    actions: torch.Tensor  # [(batch_size), T, 2]
    bias_angle: torch.Tensor  # [(batch_size), 2]
    goal: torch.Tensor  # [(batch_size), 2]


class WallExpertDataset(WallDataset):
    def __init__(self, config: WallExpertDatasetConfig, *args, **kwargs):
        self.expert_planner = ExpertPlanner(max_step=config.max_step_expert)
        super().__init__(config, *args, **kwargs)

    def generate_multistep_sample(
        self,
    ):
        walls = self.sample_walls()
        start_location = self.generate_state()
        actions, bias_angle, goal = self.generate_actions(
            start_location, walls, self.config.n_steps
        )
        sample = self.generate_transitions(
            start_location, goal, actions, bias_angle, walls
        )
        if self.config.static_noise > 0 or self.config.noise > 0:
            # static noise means just one noise overlay for all timesteps
            static_noise_overlay = (
                self.generate_static_overlay(sample.states.shape) * sample.states.max()
            )
            if self.config.static_noise_speed > 0:
                for i in range(static_noise_overlay.shape[1]):
                    static_noise_overlay[:, i] = torch.roll(
                        static_noise_overlay[:, i],
                        shifts=int(i * self.config.static_noise_speed),
                        dims=-1,
                    )
            rnd_noise_overlay = (
                self.generate_rnd_overlay(sample.states.shape) * sample.states.max()
            )
            static_noised_states = (
                sample.states
                + static_noise_overlay * self.config.static_noise
                + rnd_noise_overlay * self.config.noise
            )
            sample = ExpertSample(
                states=static_noised_states,
                locations=sample.locations,
                actions=sample.actions,
                bias_angle=bias_angle,
                goal=goal,
            )

        if self.config.action_bias_only:
            sample = ExpertSample(
                states=sample.states,
                locations=sample.locations,
                actions=sample.bias_angle.view(-1, 1, 1, 2).repeat(
                    1, sample.actions.shape[1], 1, 1
                ),
                bias_angle=sample.bias_angle,
                goal=goal,
            )

        if self.config.zero_action:
            sample = ExpertSample(
                states=sample.states,
                locations=sample.locations,
                actions=torch.zeros_like(sample.actions),
                bias_angle=sample.bias_angle,
                goal=goal,
            )
        sample = self.normalizer.normalize_sample(sample)
        return sample

    def generate_transitions(
        self,
        location,
        goals,
        actions,
        bias_angle,
        walls,
    ):
        # print("walls", walls)
        locations = [location]
        for i in range(actions.shape[1]):
            next_location = self.generate_transition(locations[-1], actions[:, i])
            # print("next_location", next_location)
            check_intersection = (
                torch.sign(locations[-1][:, 0] - walls[0])
                * torch.sign(next_location[:, 0] - walls[0])
            ) <= 0
            # print("check_intersection", check_intersection.shape)
            # print("next_location", next_location.shape)
            for j in check_intersection.nonzero():
                # print("found intersection at", i, j.item())
                d = next_location[j] - locations[-1][j]
                # a and b are the line parameters fit to the last step
                a = d[0, 1] / d[0, 0]
                b = locations[-1][j, 1] - a * locations[-1][j, 0]
                # y is the intersection point of the wall plane
                y = a * walls[0][j] + b
                # If the intersection point is in the hole, we are good
                # otherwise, we need to move the point back
                if y < walls[1][j] - 2 or y > walls[1][j] + 2:  # we're not in the hole
                    next_location[j] = locations[-1][j].clone()

            locations.append(next_location)

        # Unsqueeze for compatibility with multi-dot dataset
        locations = torch.stack(locations, dim=1).unsqueeze(dim=-2)
        actions = actions.unsqueeze(dim=-2)
        states = self.render_location(locations)
        walls = self.render_walls(*walls).unsqueeze(1).unsqueeze(1)
        walls = walls.repeat(1, states.shape[1], 1, 1, 1)
        # print('walls are', walls.shape)
        # print('states are', states.shape)
        walls *= states.max()
        states_with_walls = torch.cat([states, walls], dim=-3)
        # print('states with walls are', states_with_walls.shape)
        return ExpertSample(
            states=states_with_walls,
            locations=locations,
            actions=actions,
            bias_angle=bias_angle,
            goal=goals,
        )

    def render_walls(self, wall_locs, hole_locs):
        x = torch.arange(0, self.config.img_size, device=self.device)
        y = torch.arange(0, self.config.img_size, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        grid_x = grid_x.unsqueeze(0).repeat(self.config.batch_size, 1, 1)
        grid_y = grid_y.unsqueeze(0).repeat(self.config.batch_size, 1, 1)

        wall_locs_r = wall_locs.view(self.config.batch_size, 1, 1).repeat(
            1, self.config.img_size, self.config.img_size
        )
        hole_locs_r = hole_locs.view(self.config.batch_size, 1, 1).repeat(
            1, self.config.img_size, self.config.img_size
        )
        res = (
            (wall_locs_r == grid_x)
            * ((hole_locs_r < grid_y - 2) + (hole_locs_r > grid_y + 2))
        ).float()
        return res

    def generate_actions(
        self, positions: torch.Tensor, walls: torch.Tensor, n_steps: int
    ):
        n_random_actions = int(
            self.config.batch_size * (1 - self.config.expert_fraction)
        )
        # Note: generates n_steps - 1 actions
        x = torch.rand(n_random_actions, device=self.device) * 2 * math.pi
        bias_angle = DotDataset.angle_to_vec(x)
        actions = (
            bias_angle.view(-1, 1, 2).repeat(1, n_steps - 1, 1) * self.config.max_step
        )
        actions += (
            torch.randn_like(actions) * self.config.action_noise * self.config.max_step
        )

        expert_size = self.config.batch_size - n_random_actions
        goal = self.generate_state(size=expert_size)
        expert_actions = self.expert_planner.generate_expert_actions(
            start=positions[n_random_actions:],
            goal=goal,
            walls=(walls[0][n_random_actions:], walls[1][n_random_actions:]),
            n_steps=n_steps,
        )

        actions = torch.cat([actions, expert_actions], dim=0)

        return actions, bias_angle, goal


class ExpertPlanner:
    def __init__(self, max_step: float, wall_dist: float = 4):
        self.max_step = max_step
        self.wall_dist = wall_dist

    def generate_expert_actions(
        self,
        start: torch.Tensor,
        goal: torch.Tensor,
        walls: torch.Tensor,
        n_steps: int,
    ):
        all_actions = []
        is_different_side = (
            torch.sign(start[:, 0] - walls[0]) * torch.sign(goal[:, 0] - walls[0])
        ) <= 0

        for i in range(start.shape[0]):
            if is_different_side[i]:
                # Here we'll just plan to go to the door,
                # and from the door to the goal.
                door_loc = torch.tensor(
                    [walls[0][i], walls[1][i]], device=start.device
                )  # the door location
                if self.wall_dist == 0:
                    actions = self.plan_steps_line(start[i], door_loc)
                    actions += self.plan_steps_line(door_loc, goal[i])
                else:
                    sign = torch.sign(start[:, 0] - walls[0]).unsqueeze(1)[i]
                    delta = torch.tensor([self.wall_dist, 0], device=start.device)
                    # loc1 is on the same side as the start
                    door_loc1 = door_loc + sign * delta

                    # loc1 is on the same side as the target
                    door_loc2 = door_loc - sign * delta

                    actions = []
                    # if we need to get closer to the door
                    if (
                        sign[0] * (start[i][0] - door_loc1[0]) > 1
                        or (start[i][1] - door_loc1[1]).abs() > 1
                    ):
                        actions = self.plan_steps_line(start[i], door_loc1)
                    actions += self.plan_steps_line(door_loc1, door_loc2)
                    actions += self.plan_steps_line(door_loc2, goal[i])
            else:
                actions = self.plan_steps_line(start[i], goal[i])
            # If we have more than n_steps - 1 actions,
            # we'll just take the first n_steps - 1
            result = torch.stack(actions)[: n_steps - 1]
            pad_size = n_steps - 1 - result.shape[0]
            result = torch.nn.functional.pad(result, (0, 0, 0, pad_size))
            all_actions.append(result)
        return torch.stack(all_actions)

    def plan_steps_line(self, start: torch.Tensor, goal: torch.Tensor):
        loc = start.clone().float()
        actions = []
        direction = goal - loc
        direction /= direction.norm()
        # print('direction', direction)
        while True:
            distance = (loc - goal).norm()
            if distance < 1e-5:
                break
            action = min(distance, self.max_step) * direction
            loc += action
            actions.append(action)
        return actions


class WrappedWallExpertDataset:
    def __init__(self, config: WallExpertDatasetConfig, *args, **kwargs):
        self.orig_device = config.device

        self.ds = WallExpertDataset(
            dataclasses.replace(config, device="cpu"),
            normalizer=self.normalizer,
            *args,
            **kwargs,
        )
        self.loader = DataLoader(
            self.ds, batch_size=1, shuffle=False, num_workers=config.num_workers
        )

        self.config = self.ds.config

    def __iter__(self):
        for e in self.loader:
            e = ExpertSample(*[x[0].to(self.orig_device) for x in e])
            yield e

    def __len__(self):
        return len(self.loader)
